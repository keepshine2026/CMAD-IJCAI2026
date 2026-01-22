#!/usr/bin/env python
"""Train scorer (teacher model) for the MMLU pipeline.

Semantics:
- Per-message logits: [helpful_logit, harmful_logit]
- u_raw = harmful - helpful (higher => more likely remove)
- Run score: S = run_bias - logsumexp(u_raw over messages)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _add_repo_root_to_syspath() -> None:
    import sys

    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


_add_repo_root_to_syspath()

from experiments.mmlu.dataset import (  # noqa: E402
    FlipDataset,
    QuestionDataset,
    RoleVocab,
    SchemaConfig,
    build_all_examples,
    collate_flip,
    collate_question,
    split_by_qid,
)
from experiments.mmlu.cls_cache import SqliteClsCache  # noqa: E402


@dataclass(frozen=True)
class Config:
    logs_root: str
    seed: int
    epochs: int
    lr: float
    lr_role: float
    lr_topo: float
    batch_size: int
    flip_batch_size: int
    flip_mode: str
    pairwise_max_pairs: int
    flip_pool: str
    flip_w_neg: float
    device: str
    save_path: str
    select_metric: str
    allow_missing_parent: bool
    num_workers: int
    # embedding
    hf_model: str
    sbert_model: str
    max_length: int
    micro_batch: int
    d_model: int
    gate_init: float
    proj_seed: int
    use_cls_cache: bool
    cls_cache_path: str
    bag_only_epochs: int
    warmup_epochs: int
    # losses
    margin: float
    margin_pos: float
    margin_neg: float
    margin_sign: Optional[float]
    lambda_flip: float
    lambda_sign: float
    lambda_scale: float
    scale_huber_delta: float
    freeze_encoder: bool
    calibrate_after_train: bool
    calib_q: float
    calib_scale_q: float
    calib_clip_u: float
    calib_T: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_true = list(map(int, y_true))
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, list(map(float, y_prob))))


def compute_topo_norm(
    question_ds: QuestionDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        question_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_question,
    )
    sum_vec: Optional[torch.Tensor] = None
    sum_sq: Optional[torch.Tensor] = None
    n = 0
    for batch in loader:
        topo = batch.get("topo")
        mask = batch.get("mask")
        if topo is None or mask is None:
            continue
        valid = topo[mask]  # [M, D]
        if valid.numel() == 0:
            continue
        valid = valid.float().cpu()
        if sum_vec is None:
            sum_vec = valid.sum(dim=0)
            sum_sq = (valid * valid).sum(dim=0)
        else:
            sum_vec += valid.sum(dim=0)
            sum_sq += (valid * valid).sum(dim=0)
        n += int(valid.shape[0])
    if sum_vec is None or sum_sq is None or n == 0:
        raise RuntimeError("Failed to compute topo_norm: no valid topo vectors in dataset.")
    mean = sum_vec / float(n)
    var = (sum_sq / float(n)) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    std = torch.clamp(std, min=1e-6)
    return mean, std


def build_role_desc_texts(role_vocab: RoleVocab) -> Tuple[List[str], bool]:
    try:
        from CMAD.prompt.mmlu_prompt_set import ROLE_DESCRIPTION  # type: ignore

        role_desc = dict(ROLE_DESCRIPTION)
    except Exception:
        role_desc = {}
    had_missing = False
    texts: List[str] = []
    for role in role_vocab.id2role:
        desc = role_desc.get(role)
        if not desc:
            had_missing = True
            texts.append(str(role))
        else:
            texts.append(f"{role}: {str(desc).strip()}")
    return texts, had_missing


def init_linear_deterministic(linear: nn.Linear, *, seed: int) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    with torch.no_grad():
        w = torch.randn(linear.weight.shape, generator=gen, device="cpu", dtype=linear.weight.dtype) * 0.02
        linear.weight.copy_(w.to(device=linear.weight.device))
        if linear.bias is not None:
            linear.bias.zero_()


class FrozenEmbedder:
    """
    Frozen encoders + trainable alignment/fusion head producing per-message vector z_i:

      msg = LN(GELU(Wm * CLS(Q,MSG)))
      role = LN(GELU(Wr * SBERT(role_desc)))
      topo = LN(MLP(phi_topo_norm))
      fuse = LN( msg + gate_r*(role_src+role_dst)/2 + gate_t*topo )

    Output z_i is `d_model` and is fed into the 2-logit head.
    """

    def __init__(
        self,
        *,
        role_vocab: RoleVocab,
        hf_model: str,
        sbert_model: str,
        max_length: int,
        micro_batch: int,
        d_model: int,
        gate_init: float,
        topo_mean: torch.Tensor,
        topo_std: torch.Tensor,
        proj_seed: int,
        device: torch.device,
        freeze_encoder: bool,
        cls_cache_path: Optional[str],
    ):
        self.freeze_encoder = bool(freeze_encoder)
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError("transformers is required for cross-encoder embedding.") from e
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError("sentence-transformers is required for role embedding.") from e

        self.role_vocab = role_vocab
        self.max_length = int(max_length)
        self.micro_batch = max(1, int(micro_batch))
        self.d_model = int(d_model)
        self.device = device
        self.cache_prefix = f"{hf_model}|L{int(self.max_length)}"

        # Cross-encoder
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        self.encoder = AutoModel.from_pretrained(hf_model).to(device)
        if self.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.h_dim = int(getattr(getattr(self.encoder, "config", None), "hidden_size", 0) or 0)
        if self.h_dim <= 0:
            raise ValueError("Could not infer cross-encoder hidden size.")

        self.cls_cache: Optional[SqliteClsCache] = None
        if cls_cache_path:
            if not self.freeze_encoder:
                print("[WARN] cls_cache disabled because encoder is trainable (outputs would drift).")
            else:
                self.cls_cache = SqliteClsCache(str(cls_cache_path), dim=int(self.h_dim), dtype="f16")

        # Role descriptions -> SBERT (frozen vectors)
        self.sbert = SentenceTransformer(sbert_model, device=str(device))
        try:
            self.sbert.eval()
        except Exception:
            pass
        self._role_desc_base: Dict[str, str] = {}
        try:
            from CMAD.prompt.mmlu_prompt_set import ROLE_DESCRIPTION  # type: ignore

            self._role_desc_base = {str(k): str(v) for k, v in dict(ROLE_DESCRIPTION).items()}
        except Exception:
            self._role_desc_base = {}
        role_texts, had_missing = build_role_desc_texts(role_vocab)
        if had_missing:
            print("[WARN] role_desc missing for some roles; fallback to role_name.")
        with torch.inference_mode():
            role_h = self.sbert.encode(
                role_texts,
                batch_size=self.micro_batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            if not isinstance(role_h, torch.Tensor):
                role_h = torch.tensor(role_h)
            role_h = role_h.to(device=device, dtype=torch.float32)
        self.role_h = role_h  # [R, H_role] (frozen tensor)
        self._dynamic_role_h: Dict[str, torch.Tensor] = {}
        role_hidden = int(self.role_h.shape[1])

        # Alignment projections + fusion.
        self.msg_proj = nn.Sequential(
            nn.Linear(self.h_dim, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        ).to(device)
        init_linear_deterministic(self.msg_proj[0], seed=int(proj_seed) + 201)

        self.role_proj = nn.Sequential(
            nn.Linear(role_hidden, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        ).to(device)
        init_linear_deterministic(self.role_proj[0], seed=int(proj_seed) + 211)

        topo_mean = topo_mean.to(device=device, dtype=torch.float32).view(-1)
        topo_std = topo_std.to(device=device, dtype=torch.float32).view(-1)
        topo_std = torch.clamp(topo_std, min=1e-6)
        self.topo_mean = topo_mean
        self.topo_std = topo_std
        topo_dim = int(self.topo_mean.numel())
        hid = max(128, self.d_model * 2)
        lin1 = nn.Linear(topo_dim, hid, bias=True)
        lin2 = nn.Linear(hid, self.d_model, bias=True)
        init_linear_deterministic(lin1, seed=int(proj_seed) + 301)
        init_linear_deterministic(lin2, seed=int(proj_seed) + 302)
        self.topo_mlp = nn.Sequential(lin1, nn.GELU(), lin2, nn.LayerNorm(self.d_model)).to(device)

        self.gate_role = nn.Parameter(torch.tensor(float(gate_init), device=device, dtype=torch.float32))
        self.gate_topo = nn.Parameter(torch.tensor(float(gate_init), device=device, dtype=torch.float32))
        self.fuse_ln = nn.LayerNorm(self.d_model).to(device)

    def _format_role_text(self, role: str, *, role_desc: Optional[Dict[str, str]] = None) -> str:
        role = str(role or "Unknown")
        desc = None
        if role_desc:
            desc = role_desc.get(role)
        if not desc and self._role_desc_base:
            desc = self._role_desc_base.get(role)
        if not desc:
            return role
        return f"{role}: {str(desc).strip()}"

    def _encode_role_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, int(self.role_h.shape[1])), device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            h = self.sbert.encode(
                list(texts),
                batch_size=self.micro_batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            if not isinstance(h, torch.Tensor):
                h = torch.tensor(h)
            return h.to(device=self.device, dtype=torch.float32)

    def _role_h_for(
        self,
        role_id: torch.Tensor,
        role_name: Optional[str],
        *,
        role_desc: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        # role_id is expected to be scalar on device; role_name is optional.
        r = int(self.role_h.shape[0])
        role_id_safe = int(torch.clamp(role_id, min=0, max=max(0, r - 1)).item()) if r > 0 else 0

        if role_name:
            name = str(role_name or "Unknown")
            if name and name != "Unknown":
                vid = self.role_vocab.role2id.get(name)
                if vid is not None and 0 <= int(vid) < r:
                    return self.role_h[int(vid)]
                text = self._format_role_text(name, role_desc=role_desc)
                cached = self._dynamic_role_h.get(text)
                if cached is None:
                    h = self._encode_role_texts([text])
                    cached = h[0]
                    self._dynamic_role_h[text] = cached
                return cached

        if r <= 0:
            return torch.zeros((0,), device=self.device, dtype=torch.float32)
        return self.role_h[role_id_safe]

    def _role_h_batch(
        self,
        role_ids: torch.Tensor,
        role_names: Optional[List[str]],
        *,
        role_desc: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        # Returns [M, H_role] float32 on self.device.
        if role_ids.ndim != 1:
            role_ids = role_ids.view(-1)
        m = int(role_ids.numel())
        if m == 0:
            return torch.zeros((0, int(self.role_h.shape[1])), device=self.device, dtype=torch.float32)

        r = int(self.role_h.shape[0])
        role_ids_safe = role_ids.clamp(min=0, max=max(0, r - 1)) if r > 0 else role_ids.clamp(min=0)
        out = self.role_h[role_ids_safe]
        if not role_names:
            return out

        if len(role_names) != m:
            # Best-effort: ignore mismatch and keep id-based embeddings.
            return out

        # Fill overrides (known-by-name or dynamic-by-name).
        dyn_texts: List[str] = []
        dyn_pos: List[int] = []
        for i, name in enumerate(role_names):
            name = str(name or "Unknown")
            if not name or name == "Unknown":
                continue
            vid = self.role_vocab.role2id.get(name)
            if vid is not None and 0 <= int(vid) < r:
                out[i] = self.role_h[int(vid)]
                continue
            text = self._format_role_text(name, role_desc=role_desc)
            cached = self._dynamic_role_h.get(text)
            if cached is not None:
                out[i] = cached
            else:
                dyn_texts.append(text)
                dyn_pos.append(i)

        if dyn_texts:
            # Encode in one batch, then cache + write back.
            h = self._encode_role_texts(dyn_texts)
            for j, pos in enumerate(dyn_pos):
                vec = h[j]
                out[pos] = vec
                self._dynamic_role_h[dyn_texts[j]] = vec
        return out

    def reset_cache_stats(self) -> None:
        if self.cls_cache is not None:
            self.cls_cache.reset_stats()

    def cache_stats(self) -> Optional[Dict[str, int]]:
        if self.cls_cache is None:
            return None
        st = self.cls_cache.stats()
        return {"hits": int(st.hits), "misses": int(st.misses), "puts": int(st.puts)}

    def _mk_cache_key(self, msg_key: str) -> str:
        return f"{self.cache_prefix}|{msg_key}"

    def _encode_cross_encoder(
        self,
        *,
        q_list: List[str],
        m_list: List[str],
        cache_keys: Optional[List[str]],
    ) -> torch.Tensor:
        m = len(q_list)
        if m == 0:
            return torch.zeros((0, self.h_dim), device=self.device, dtype=torch.float32)

        # No cache: compute all.
        if self.cls_cache is None or cache_keys is None:
            h_list: List[torch.Tensor] = []
            ctx = torch.inference_mode() if self.freeze_encoder else torch.enable_grad()
            with ctx:
                for start in range(0, m, self.micro_batch):
                    end = min(m, start + self.micro_batch)
                    inputs = self.tokenizer(
                        q_list[start:end],
                        m_list[start:end],
                        truncation=True,
                        max_length=int(self.max_length),
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    out = self.encoder(**inputs)
                    h_list.append(out.last_hidden_state[:, 0, :].to(dtype=torch.float32))
            return torch.cat(h_list, dim=0)

        # Cache: read what we can, compute the rest.
        cached = self.cls_cache.get_many(cache_keys)
        h = torch.empty((m, self.h_dim), device=self.device, dtype=torch.float32)

        missing_idx: List[int] = []
        for i, ck in enumerate(cache_keys):
            arr = cached.get(ck)
            if arr is None:
                missing_idx.append(i)
            else:
                h[i] = torch.from_numpy(arr).to(device=self.device, dtype=torch.float32)

        if not missing_idx:
            return h

        to_put: List[Tuple[str, np.ndarray]] = []
        with torch.inference_mode():
            for start in range(0, len(missing_idx), self.micro_batch):
                end = min(len(missing_idx), start + self.micro_batch)
                idx_chunk = missing_idx[start:end]
                q_chunk = [q_list[i] for i in idx_chunk]
                m_chunk = [m_list[i] for i in idx_chunk]
                inputs = self.tokenizer(
                    q_chunk,
                    m_chunk,
                    truncation=True,
                    max_length=int(self.max_length),
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self.encoder(**inputs)
                h_chunk = out.last_hidden_state[:, 0, :].to(dtype=torch.float32)
                for j, i in enumerate(idx_chunk):
                    h[i] = h_chunk[j]
                    to_put.append((cache_keys[i], h_chunk[j].detach().cpu().numpy()))

        if to_put:
            self.cls_cache.put_many(to_put)
        return h

    def embed_question_batch(
        self, batch: Dict[str, Any], *, return_msg_emb: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        q_texts: List[str] = list(batch.get("q_text") or [])
        texts: List[List[str]] = list(batch.get("texts") or [])
        keys: List[List[str]] = list(batch.get("keys") or [])
        qids: List[str] = list(batch.get("qid") or [])
        run_ids: List[str] = list(batch.get("run_id") or [])

        mask: torch.Tensor = batch["mask"].to(torch.bool)
        topo: torch.Tensor = batch.get("topo")
        if topo is None:
            topo = torch.zeros((mask.shape[0], mask.shape[1], int(self.topo_mean.numel())), dtype=torch.float32)
        src_role_id: torch.Tensor = batch["src_role_id"].to(torch.long)
        dst_role_id: torch.Tensor = batch["dst_role_id"].to(torch.long)
        src_role_names: Optional[List[List[str]]] = batch.get("src_role")
        dst_role_names: Optional[List[List[str]]] = batch.get("dst_role")
        role_desc: Optional[Dict[str, str]] = batch.get("role_desc") if isinstance(batch.get("role_desc"), dict) else None

        b_idx, n_idx = mask.nonzero(as_tuple=True)
        b_list = b_idx.tolist()
        n_list = n_idx.tolist()
        m = len(b_list)
        if m == 0:
            z = torch.zeros((0, self.d_model), device=self.device, dtype=torch.float32)
            return z, {"batch_idx": b_idx.cpu(), "msg_pos": n_idx.cpu()}

        q_flat = [str(q_texts[b]) for b in b_list]
        m_flat = [str(texts[b][j]) for b, j in zip(b_list, n_list)]
        k_flat = [str(keys[b][j]) for b, j in zip(b_list, n_list)]
        src_flat_name: Optional[List[str]] = None
        dst_flat_name: Optional[List[str]] = None
        if isinstance(src_role_names, list) and isinstance(dst_role_names, list) and len(src_role_names) == len(texts):
            try:
                src_flat_name = [str(src_role_names[b][j]) for b, j in zip(b_list, n_list)]
                dst_flat_name = [str(dst_role_names[b][j]) for b, j in zip(b_list, n_list)]
            except Exception:
                src_flat_name = None
                dst_flat_name = None
        cache_keys = [self._mk_cache_key(k) for k in k_flat] if self.cls_cache is not None else None

        # Cross-encoder CLS(Q, MSG) (optionally cached)
        h = self._encode_cross_encoder(q_list=q_flat, m_list=m_flat, cache_keys=cache_keys)

        flat_src = src_role_id[b_idx, n_idx].to(self.device, dtype=torch.long)
        flat_dst = dst_role_id[b_idx, n_idx].to(self.device, dtype=torch.long)
        role_src_h = self._role_h_batch(flat_src, src_flat_name, role_desc=role_desc)
        role_dst_h = self._role_h_batch(flat_dst, dst_flat_name, role_desc=role_desc)
        role_src = self.role_proj(role_src_h)
        role_dst = self.role_proj(role_dst_h)
        role_avg = 0.5 * (role_src + role_dst)

        phi = topo[b_idx, n_idx].to(self.device, dtype=torch.float32)
        phi = (phi - self.topo_mean) / self.topo_std
        topo_emb = self.topo_mlp(phi)

        msg_emb = self.msg_proj(h)
        z = self.fuse_ln(msg_emb + self.gate_role * role_avg + self.gate_topo * topo_emb)

        dbg = {
            "gate_role": float(self.gate_role.detach().cpu().item()),
            "gate_topo": float(self.gate_topo.detach().cpu().item()),
            "msg_norm": float(msg_emb.detach().norm(dim=1).mean().cpu().item()),
            "role_norm": float(role_avg.detach().norm(dim=1).mean().cpu().item()),
            "topo_norm": float(topo_emb.detach().norm(dim=1).mean().cpu().item()),
        }
        meta = {
            "batch_idx": b_idx.cpu(),
            "msg_pos": n_idx.cpu(),
            "qid": [str(qids[b]) for b in b_list],
            "run_id": [str(run_ids[b]) for b in b_list],
            "key": k_flat,
            "dbg": dbg,
        }
        if return_msg_emb:
            # For teacher-cache usage only (no gradients); keep it compact on CPU.
            meta["msg_emb"] = msg_emb.detach().cpu().to(dtype=torch.float16)
        return z, meta

    def embed_flip_item(self, item: Dict[str, Any]) -> torch.Tensor:
        q_text = str(item.get("q_text") or "")
        msgs: List[Dict[str, Any]] = list(item.get("base_messages") or [])
        if not msgs:
            return torch.zeros((0, self.d_model), device=self.device, dtype=torch.float32)

        msg_texts = [str(m.get("text") or "") for m in msgs]
        msg_keys = [str(m.get("key") or "") for m in msgs]
        q_list = [q_text] * len(msg_texts)
        cache_keys = [self._mk_cache_key(k) for k in msg_keys] if self.cls_cache is not None else None

        h = self._encode_cross_encoder(q_list=q_list, m_list=msg_texts, cache_keys=cache_keys)

        src_role_id = torch.tensor([int(m.get("src_role_id", 0)) for m in msgs], dtype=torch.long, device=self.device)
        dst_role_id = torch.tensor([int(m.get("dst_role_id", 0)) for m in msgs], dtype=torch.long, device=self.device)
        src_role_name = [str(m.get("src_role") or "") for m in msgs]
        dst_role_name = [str(m.get("dst_role") or "") for m in msgs]
        role_src_h = self._role_h_batch(src_role_id, src_role_name, role_desc=None)
        role_dst_h = self._role_h_batch(dst_role_id, dst_role_name, role_desc=None)
        role_src = self.role_proj(role_src_h)
        role_dst = self.role_proj(role_dst_h)
        role_avg = 0.5 * (role_src + role_dst)

        phi = torch.stack([m.get("phi_topo") for m in msgs], dim=0).to(self.device, dtype=torch.float32)
        phi = (phi - self.topo_mean) / self.topo_std
        topo_emb = self.topo_mlp(phi)

        msg_emb = self.msg_proj(h)
        return self.fuse_ln(msg_emb + self.gate_role * role_avg + self.gate_topo * topo_emb)


class ScorerHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FrozenScorer:
    """
    Lightweight frozen scorer loader for checkpoints.

    Provides per-message outputs with fixed calibration:
      - u_raw = harmful - helpful
      - u_centered = clip((u_raw-u_center)/u_scale, [-clip_u, clip_u])
      - y_del = sigmoid(u_centered/T), y_keep = 1 - y_del
    """

    def __init__(
        self,
        *,
        ckpt_path: str,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        micro_batch: Optional[int] = None,
    ) -> None:
        from experiments.mmlu.dataset import PHI_TOPO_DIM

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg = ckpt.get("config") or {}

        hf_model = str(cfg.get("hf_model") or "microsoft/deberta-v3-base")
        sbert_model = str(cfg.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2")
        d_model = int(cfg.get("d_model") or 256)
        proj_seed = int(cfg.get("proj_seed") or 0)
        cfg_max_len = int(cfg.get("max_length") or 256)
        cfg_mb = int(cfg.get("micro_batch") or 64)

        max_length = cfg_max_len if max_length is None else int(max_length)
        micro_batch = cfg_mb if micro_batch is None else int(micro_batch)

        dev_str = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = torch.device(dev_str)

        id2role = (ckpt.get("role_vocab") or {}).get("id2role") or []
        self.role_vocab = RoleVocab(list(map(str, id2role)))

        topo_norm = ckpt.get("topo_norm") or {}
        mean_raw = topo_norm.get("mean", None)
        std_raw = topo_norm.get("std", None)
        if mean_raw is None:
            mean_raw = [0.0] * int(PHI_TOPO_DIM)
        if std_raw is None:
            std_raw = [1.0] * int(PHI_TOPO_DIM)
        topo_mean = torch.as_tensor(mean_raw, dtype=torch.float32).view(-1)
        topo_std = torch.as_tensor(std_raw, dtype=torch.float32).view(-1)
        if int(topo_mean.numel()) != int(PHI_TOPO_DIM):
            padded = torch.zeros((int(PHI_TOPO_DIM),), dtype=torch.float32)
            n_copy = min(int(PHI_TOPO_DIM), int(topo_mean.numel()))
            padded[:n_copy] = topo_mean[:n_copy]
            topo_mean = padded
        if int(topo_std.numel()) != int(PHI_TOPO_DIM):
            padded = torch.ones((int(PHI_TOPO_DIM),), dtype=torch.float32)
            n_copy = min(int(PHI_TOPO_DIM), int(topo_std.numel()))
            padded[:n_copy] = topo_std[:n_copy]
            topo_std = padded
        topo_std = torch.clamp(topo_std, min=1e-6)

        self.embedder = FrozenEmbedder(
            role_vocab=self.role_vocab,
            hf_model=hf_model,
            sbert_model=sbert_model,
            max_length=max_length,
            micro_batch=micro_batch,
            d_model=d_model,
            gate_init=0.0,
            topo_mean=topo_mean,
            topo_std=topo_std,
            proj_seed=proj_seed,
            device=self.device,
            freeze_encoder=True,
            cls_cache_path=None,
        )
        self.embedder.msg_proj.load_state_dict(ckpt["msg_proj_state_dict"])
        self.embedder.role_proj.load_state_dict(ckpt["role_proj_state_dict"])
        self.embedder.topo_mlp.load_state_dict(ckpt["topo_mlp_state_dict"])
        self.embedder.fuse_ln.load_state_dict(ckpt["fuse_ln_state_dict"])
        self.embedder.gate_role.data.copy_(torch.tensor(float(ckpt.get("gate_role") or 0.0), device=self.device))
        self.embedder.gate_topo.data.copy_(torch.tensor(float(ckpt.get("gate_topo") or 0.0), device=self.device))
        if "encoder_state_dict" in ckpt:
            try:
                self.embedder.encoder.load_state_dict(ckpt["encoder_state_dict"])
            except Exception as e:
                print(f"[WARN] failed to load encoder_state_dict from ckpt: {e}")

        self.head = ScorerHead(d_model).to(self.device)
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.head.eval()

        self.run_bias = float(ckpt.get("run_bias") or 0.0)

        calib = ckpt.get("calibration") or {}
        if not isinstance(calib, dict) or ("u_center" not in calib or "u_scale" not in calib):
            # Fallback (not ideal): behave like uncalibrated checkpoint.
            calib = {
                "u_center": 0.0,
                "u_scale": 1.0,
                "clip_u": float(cfg.get("calib_clip_u") or 20.0),
                "T": float(cfg.get("calib_T") or 1.5),
                "eps": 1e-6,
            }
            print("[WARN] checkpoint missing calibration; using identity calibration (may drift).")
        self.calibration: Dict[str, float] = {k: float(v) for k, v in calib.items()}

    @torch.inference_mode()
    def predict_question_batch(self, batch: Dict[str, Any], *, return_text_emb: bool = False) -> Dict[str, Any]:
        z, meta = self.embedder.embed_question_batch(batch, return_msg_emb=bool(return_text_emb))
        logits = self.head(z)  # [M,2]
        helpful_logit = logits[:, 0].view(-1)
        harmful_logit = logits[:, 1].view(-1)
        u_raw = (harmful_logit - helpful_logit).view(-1)
        u_centered, y_del, y_keep = _apply_calibration(u_raw, self.calibration)
        text_emb = None
        if bool(return_text_emb) and isinstance(meta, dict) and ("msg_emb" in meta):
            text_emb = meta.get("msg_emb")
            try:
                del meta["msg_emb"]
            except Exception:
                pass
        out = {
            "helpful_logit": helpful_logit.detach(),
            "harmful_logit": harmful_logit.detach(),
            "helpful_prob": torch.sigmoid(helpful_logit.detach()),
            "harmful_prob": torch.sigmoid(harmful_logit.detach()),
            "u_raw": u_raw.detach(),
            "u_centered": u_centered.detach(),
            "y_del": y_del.detach(),
            "y_keep": y_keep.detach(),
            "text_emb": text_emb,
            "meta": meta,
            "calibration": dict(self.calibration),
            "debug": dict(meta.get("dbg") or {}) if isinstance(meta, dict) else {},
        }
        return out


def bag_logsumexp(s: torch.Tensor, batch_idx: torch.Tensor, bsz: int) -> torch.Tensor:
    """
    Bag pooling via logsumexp.
    """
    out: List[torch.Tensor] = []
    for b in range(int(bsz)):
        sb = s[batch_idx == b]
        if sb.numel() == 0:
            out.append(torch.tensor(-1e9, device=s.device, dtype=s.dtype))
        else:
            out.append(torch.logsumexp(sb, dim=0)) 
    return torch.stack(out, dim=0)


def clean_indices(idx: Sequence[int], n: int) -> List[int]:
    return sorted({int(i) for i in idx if 0 <= int(i) < int(n)})


def _pool_1d(u: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pool a 1D tensor into a scalar.

    Supported:
      - lse: logsumexp
      - lme: logmeanexp = logsumexp - log(n)
      - max: max
      - mean: mean

    Caller must ensure u.numel() > 0.
    """
    if u.ndim != 1:
        u = u.view(-1)
    n = int(u.numel())
    if n <= 0:
        raise ValueError("Cannot pool empty tensor.")

    mode = str(mode).strip().lower()
    if mode == "lse":
        return torch.logsumexp(u, dim=0)
    if mode == "lme":
        return torch.logsumexp(u, dim=0) - torch.log(torch.tensor(float(n), device=u.device, dtype=u.dtype))
    if mode == "max":
        return u.max()
    if mode == "mean":
        return u.mean()
    if mode.startswith("topk"):
        k_text = mode[len("topk") :].strip()
        if not k_text:
            raise ValueError("flip_pool 'topkK' requires an integer K, e.g. topk3.")
        try:
            k = int(k_text)
        except ValueError as e:
            raise ValueError(f"Invalid flip_pool={mode!r}; expected e.g. topk3.") from e
        if k <= 0:
            raise ValueError(f"Invalid flip_pool={mode!r}; K must be >= 1.")
        if n <= k:
            return u.mean()
        return torch.topk(u, k, largest=True).values.mean()
    raise ValueError(f"Unknown flip_pool={mode!r}. Expected one of: lse,lme,max,mean,topkK (e.g. topk3)")


def flip_loss_terms(
    *,
    u: torch.Tensor,
    removed_idx: Sequence[int],
    delta: int,
    margin_pos: float,
    margin_neg: float,
    margin_sign: Optional[float],
    flip_pool: str,
    flip_w_neg: float,
    flip_mode: str,
    pairwise_max_pairs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(delta) == 0:
        return torch.tensor(0.0, device=u.device), torch.tensor(0.0, device=u.device)
    n = int(u.numel())
    idx = clean_indices(removed_idx, n)
    if not idx or len(idx) >= n:
        return torch.tensor(0.0, device=u.device), torch.tensor(0.0, device=u.device)

    removed_mask = torch.zeros((n,), dtype=torch.bool, device=u.device)
    removed_mask[torch.tensor(idx, dtype=torch.long, device=u.device)] = True
    keep_mask = ~removed_mask
    if keep_mask.sum().item() == 0:
        return torch.tensor(0.0, device=u.device), torch.tensor(0.0, device=u.device)

    u_r = u[removed_mask]
    u_k = u[keep_mask]
    if u_r.numel() == 0 or u_k.numel() == 0:
        return torch.tensor(0.0, device=u.device), torch.tensor(0.0, device=u.device)

    flip_mode = str(flip_mode).strip().lower()
    if flip_mode not in {"pool", "pairwise"}:
        raise ValueError(f"Unknown flip_mode={flip_mode!r}. Expected 'pool' or 'pairwise'.")

    delta_i = int(delta)
    margin_d = float(margin_pos) if delta_i == 1 else float(margin_neg)
    w = 1.0 if delta_i == 1 else float(flip_w_neg)

    if flip_mode == "pool":
        u_r_pool = _pool_1d(u_r, flip_pool)
        u_k_pool = _pool_1d(u_k, flip_pool)
        gap = (-float(delta_i)) * (u_r_pool - u_k_pool)

        margin_t = u.new_tensor(float(margin_d))
        l_flip = float(w) * F.softplus(margin_t - gap)
        if margin_sign is None:
            l_sign = F.softplus(-gap)
        else:
            l_sign = F.softplus(u.new_tensor(float(margin_sign)) - gap)
        return l_flip, l_sign

    # pairwise: all (removed_i, kept_j) pairs
    # gap_ij = -delta * (u_r_i - u_k_j)
    d = float(delta_i)
    nr = int(u_r.numel())
    nk = int(u_k.numel())
    n_pairs = nr * nk
    if n_pairs <= 0:
        return torch.tensor(0.0, device=u.device), torch.tensor(0.0, device=u.device)

    # Safety: avoid materializing gigantic matrices.
    # For typical runs, nr/nk are small; this path remains fast.
    max_materialize = 2_000_000  # pairs
    if n_pairs <= max_materialize:
        gap_mat = (-d) * (u_r.view(nr, 1) - u_k.view(1, nk))  # [nr, nk]
        gap_flat = gap_mat.reshape(-1)
    else:
        # Sample without materializing full gap matrix.
        max_pairs = max(1, int(pairwise_max_pairs))
        flat_idx = torch.randint(0, int(n_pairs), (max_pairs,), device=u.device)
        i = torch.div(flat_idx, nk, rounding_mode="floor")
        j = flat_idx - i * nk
        gap_flat = (-d) * (u_r[i] - u_k[j])

    max_pairs = max(1, int(pairwise_max_pairs))
    if gap_flat.numel() > max_pairs:
        perm = torch.randperm(int(gap_flat.numel()), device=u.device)[:max_pairs]
        gap_flat = gap_flat[perm]

    margin_t = u.new_tensor(float(margin_d))
    l_flip = float(w) * F.softplus(margin_t - gap_flat).mean()
    if margin_sign is None:
        l_sign = F.softplus(-gap_flat).mean()
    else:
        l_sign = F.softplus(u.new_tensor(float(margin_sign)) - gap_flat).mean()
    return l_flip, l_sign


@torch.inference_mode()
def _run_level_outputs(
    question_ds: QuestionDataset,
    *,
    embedder: FrozenEmbedder,
    head: ScorerHead,
    run_bias: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[int], List[float], List[float]]:
    loader = DataLoader(
        question_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_question,
    )
    y_true: List[int] = []
    y_prob: List[float] = []
    s_b_all: List[float] = []
    for batch in loader:
        z, meta = embedder.embed_question_batch(batch)
        logits = head(z)
        u = (logits[:, 1] - logits[:, 0]).view(-1)
        b_idx = meta["batch_idx"].to(device=u.device)
        bsz = len(batch["run_id"])
        U = bag_logsumexp(u, b_idx, bsz)
        S = run_bias.to(device=u.device, dtype=u.dtype) - U
        s_b_all.extend(S.detach().cpu().numpy().tolist())
        prob = torch.sigmoid(S).detach().cpu().numpy().tolist()
        y = batch["y"].detach().cpu().numpy().astype(int).tolist()
        y_true.extend(y)
        y_prob.extend(prob)
    return y_true, y_prob, s_b_all


def _choose_threshold_max_f1(y_true: Sequence[int], y_prob: Sequence[float]) -> Tuple[float, float]:
    """
    Choose threshold to maximize F1 on y_prob.
    Returns (threshold, best_f1).
    """
    y = np.asarray(list(map(int, y_true)), dtype=np.int32)
    p = np.asarray(list(map(float, y_prob)), dtype=np.float32)
    if y.size == 0:
        return 0.5, float("nan")
    if len(set(y.tolist())) < 2:
        return 0.5, float("nan")

    order = np.argsort(-p)
    p_s = p[order]
    y_s = y[order]
    tp = np.cumsum(y_s)
    pred_pos = np.arange(1, y_s.size + 1)
    fp = pred_pos - tp
    fn = tp[-1] - tp
    denom = (2 * tp + fp + fn).astype(np.float64)
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    best_k = int(np.argmax(f1)) + 1
    thr = float(p_s[best_k - 1])
    return thr, float(f1[best_k - 1])


def _run_metrics(
    *,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    s_b: Sequence[float],
    threshold: float,
    name: str,
) -> Dict[str, float]:
    y_true_l = list(map(int, y_true))
    y_prob_l = list(map(float, y_prob))
    y_pred = [1 if p >= float(threshold) else 0 for p in y_prob_l]

    pos_rate = float(np.mean(y_true_l)) if y_true_l else float("nan")
    pred_pos_rate = float(np.mean(y_pred)) if y_pred else float("nan")
    if s_b:
        q10, q50, q90 = np.percentile(np.asarray(list(map(float, s_b)), dtype=np.float32), [10, 50, 90]).tolist()
    else:
        q10 = q50 = q90 = float("nan")

    print(
        f"[Diag run {name}] thr={float(threshold):.3f} pos_rate={pos_rate:.3f} pred_pos_rate={pred_pos_rate:.3f} "
        f"S_b_q10={q10:.3f} S_b_med={q50:.3f} S_b_q90={q90:.3f}"
    )

    acc = float(accuracy_score(y_true_l, y_pred)) if y_true_l else float("nan")
    f1 = float(f1_score(y_true_l, y_pred)) if len(set(y_true_l)) >= 2 else float("nan")
    auc = safe_auc(y_true_l, y_prob_l)
    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "threshold": float(threshold),
        "pos_rate": pos_rate,
        "pred_pos_rate": pred_pos_rate,
        "S_b_q10": float(q10),
        "S_b_med": float(q50),
        "S_b_q90": float(q90),
    }


def _tensor_stats_1d(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().view(-1)
    if x.numel() == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    qs = torch.tensor([0.1, 0.5, 0.9], device=x.device, dtype=x.dtype)
    qv = torch.quantile(x, qs).detach().cpu().numpy().tolist()
    return {
        "n": float(int(x.numel())),
        "mean": float(x.mean().detach().cpu().item()),
        "std": float(x.std(unbiased=False).detach().cpu().item()),
        "p10": float(qv[0]),
        "p50": float(qv[1]),
        "p90": float(qv[2]),
    }


def _compute_calibration(
    u_raw: torch.Tensor,
    *,
    q_center: float,
    q_scale: float,
    clip_u: float,
    T: float,
) -> Dict[str, float]:
    u = u_raw.detach().float().view(-1)
    if u.numel() == 0:
        raise RuntimeError("Calibration failed: empty u_raw.")
    q_center_t = torch.tensor([float(q_center)], device=u.device, dtype=u.dtype)
    u_center = float(torch.quantile(u, q_center_t).detach().cpu().item())
    dev = (u - float(u_center)).abs()
    q_scale_t = torch.tensor([float(q_scale)], device=u.device, dtype=u.dtype)
    u_scale = float(torch.quantile(dev, q_scale_t).detach().cpu().item())
    u_scale = max(float(u_scale), 1e-6)
    return {
        "q_center": float(q_center),
        "q_scale": float(q_scale),
        "u_center": float(u_center),
        "u_scale": float(u_scale),
        "clip_u": float(clip_u),
        "T": float(T),
        "eps": 1e-6,
    }


def _apply_calibration(u_raw: torch.Tensor, calib: Dict[str, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = u_raw.detach().float().view(-1)
    u_center = float(calib.get("u_center", 0.0))
    u_scale = float(calib.get("u_scale", 1.0))
    eps = float(calib.get("eps", 1e-6))
    clip_u = float(calib.get("clip_u", 20.0))
    T = float(calib.get("T", 1.5))
    u_centered = (u - u_center) / max(u_scale, eps)
    u_centered = torch.clamp(u_centered, min=-clip_u, max=clip_u)
    y_del = torch.sigmoid(u_centered / max(T, 1e-6))
    y_keep = 1.0 - y_del
    return u_centered, y_del, y_keep


@torch.inference_mode()
def calibrate_checkpoint_inplace(
    ckpt_path: str,
    *,
    question_ds: QuestionDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    q_center: float,
    q_scale: float,
    clip_u: float,
    T: float,
) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("config") or {}
    hf_model = str(cfg.get("hf_model") or "microsoft/deberta-v3-base")
    sbert_model = str(cfg.get("sbert_model") or "sentence-transformers/all-MiniLM-L6-v2")
    d_model = int(cfg.get("d_model") or 256)
    proj_seed = int(cfg.get("proj_seed") or 0)
    max_length = int(cfg.get("max_length") or 256)
    micro_batch = int(cfg.get("micro_batch") or 64)

    id2role = (ckpt.get("role_vocab") or {}).get("id2role") or []
    role_vocab = RoleVocab(list(map(str, id2role)))

    topo_norm = ckpt.get("topo_norm") or {}
    topo_mean = torch.as_tensor(topo_norm.get("mean", None), dtype=torch.float32).view(-1)
    topo_std = torch.as_tensor(topo_norm.get("std", None), dtype=torch.float32).view(-1)
    topo_std = torch.clamp(topo_std, min=1e-6)

    embedder = FrozenEmbedder(
        role_vocab=role_vocab,
        hf_model=hf_model,
        sbert_model=sbert_model,
        max_length=max_length,
        micro_batch=micro_batch,
        d_model=d_model,
        gate_init=0.0,
        topo_mean=topo_mean,
        topo_std=topo_std,
        proj_seed=proj_seed,
        device=device,
        freeze_encoder=True,
        cls_cache_path=None,
    )
    if "msg_proj_state_dict" in ckpt:
        embedder.msg_proj.load_state_dict(ckpt["msg_proj_state_dict"])
    if "role_proj_state_dict" in ckpt:
        embedder.role_proj.load_state_dict(ckpt["role_proj_state_dict"])
    if "topo_mlp_state_dict" in ckpt:
        embedder.topo_mlp.load_state_dict(ckpt["topo_mlp_state_dict"])
    if "fuse_ln_state_dict" in ckpt:
        embedder.fuse_ln.load_state_dict(ckpt["fuse_ln_state_dict"])
    if "gate_role" in ckpt:
        embedder.gate_role.data.copy_(torch.tensor(float(ckpt["gate_role"]), device=device, dtype=torch.float32))
    if "gate_topo" in ckpt:
        embedder.gate_topo.data.copy_(torch.tensor(float(ckpt["gate_topo"]), device=device, dtype=torch.float32))
    if "encoder_state_dict" in ckpt:
        try:
            embedder.encoder.load_state_dict(ckpt["encoder_state_dict"])
        except Exception as e:
            print(f"[WARN] failed to load encoder_state_dict from ckpt: {e}")

    head = ScorerHead(d_model).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    loader = DataLoader(
        question_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_question,
    )
    u_all: List[torch.Tensor] = []
    dbg_sum: Dict[str, float] = {"msg_norm": 0.0, "role_norm": 0.0, "topo_norm": 0.0}
    dbg_n = 0
    for batch in loader:
        z, meta = embedder.embed_question_batch(batch)
        logits = head(z)
        u = (logits[:, 1] - logits[:, 0]).detach().cpu()
        u_all.append(u)
        dbg = meta.get("dbg") if isinstance(meta, dict) else None
        if isinstance(dbg, dict):
            for k in list(dbg_sum.keys()):
                if k in dbg:
                    dbg_sum[k] += float(dbg[k])
            dbg_n += 1
    if not u_all:
        raise RuntimeError("Calibration failed: no u_raw collected.")
    u_raw = torch.cat(u_all, dim=0).view(-1)

    calib = _compute_calibration(u_raw, q_center=q_center, q_scale=q_scale, clip_u=clip_u, T=T)
    u_centered, y_del, _y_keep = _apply_calibration(u_raw, calib)

    diag = {
        "u_raw": _tensor_stats_1d(u_raw),
        "u_centered": _tensor_stats_1d(u_centered),
        "y_del": _tensor_stats_1d(y_del),
        "gate_role": float(embedder.gate_role.detach().cpu().item()),
        "gate_topo": float(embedder.gate_topo.detach().cpu().item()),
        "msg_norm_mean": (dbg_sum["msg_norm"] / float(max(1, dbg_n))),
        "role_norm_mean": (dbg_sum["role_norm"] / float(max(1, dbg_n))),
        "topo_norm_mean": (dbg_sum["topo_norm"] / float(max(1, dbg_n))),
        "calibration": calib,
    }

    ckpt["calibration"] = calib
    torch.save(ckpt, str(ckpt_path))
    return diag


@torch.inference_mode()
def eval_run_level(
    question_ds: QuestionDataset,
    *,
    embedder: FrozenEmbedder,
    head: ScorerHead,
    run_bias: torch.Tensor,
    batch_size: int,
    num_workers: int,
    threshold: float,
    name: str,
) -> Dict[str, float]:
    y_true, y_prob, s_b = _run_level_outputs(
        question_ds,
        embedder=embedder,
        head=head,
        run_bias=run_bias,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return _run_metrics(y_true=y_true, y_prob=y_prob, s_b=s_b, threshold=threshold, name=name)

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    pos_rate = float(np.mean(y_true)) if y_true else float("nan")
    pred_pos_rate = float(np.mean(y_pred)) if y_pred else float("nan")
    if s_b_all:
        q10, q50, q90 = np.percentile(np.asarray(s_b_all, dtype=np.float32), [10, 50, 90]).tolist()
    else:
        q10 = q50 = q90 = float("nan")

    print(
        f"[Diag run] pos_rate={pos_rate:.3f} pred_pos_rate={pred_pos_rate:.3f} "
        f"S_b_q10={q10:.3f} S_b_med={q50:.3f} S_b_q90={q90:.3f}"
    )

    acc = float(accuracy_score(y_true, y_pred)) if y_true else float("nan")
    f1 = float(f1_score(y_true, y_pred)) if len(set(y_true)) >= 2 else float("nan")
    auc = safe_auc(y_true, y_prob)
    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "pos_rate": pos_rate,
        "pred_pos_rate": pred_pos_rate,
        "S_b_q10": float(q10),
        "S_b_med": float(q50),
        "S_b_q90": float(q90),
    }


@torch.inference_mode()
def eval_flip_level(
    flip_ds: FlipDataset,
    *,
    embedder: FrozenEmbedder,
    head: ScorerHead,
    flip_pool: str,
) -> Dict[str, float]:
    if len(flip_ds) == 0:
        return {
            "pair_acc": float("nan"),
            "pair_acc_pos": float("nan"),
            "pair_acc_neg": float("nan"),
            "n_pos": 0.0,
            "n_neg": 0.0,
            "removed_mass": float("nan"),
            "gap_mean": float("nan"),
            "gap_pos_mean": float("nan"),
            "gap_neg_mean": float("nan"),
            "mean_n_removed": float("nan"),
            "mean_n_kept": float("nan"),
            "size_bias": float("nan"),
            "diff_pos_mean": float("nan"),
            "diff_neg_mean": float("nan"),
            "pairwise_gap_mean": float("nan"),
        }
    correct = 0
    total = 0
    correct_pos = 0
    total_pos = 0
    correct_neg = 0
    total_neg = 0
    masses: List[float] = []
    gaps: List[float] = []
    gaps_pos: List[float] = []
    gaps_neg: List[float] = []
    n_removed_list: List[int] = []
    n_kept_list: List[int] = []
    size_bias_list: List[float] = []
    diffs_pos: List[float] = []
    diffs_neg: List[float] = []
    pairwise_gap_means: List[float] = []
    for i in range(len(flip_ds)):
        it = flip_ds[i]
        z = embedder.embed_flip_item(it)
        logits = head(z)
        u = (logits[:, 1] - logits[:, 0]).view(-1)
        n = int(u.numel())
        idx = clean_indices(it.get("removed_idx") or [], n)
        if not idx or len(idx) >= n:
            continue
        removed_mask = torch.zeros((n,), dtype=torch.bool, device=u.device)
        removed_mask[torch.tensor(idx, dtype=torch.long, device=u.device)] = True
        keep_mask = ~removed_mask
        u_r = u[removed_mask]
        u_k = u[keep_mask]
        if u_r.numel() == 0 or u_k.numel() == 0:
            continue
        u_r_pool = _pool_1d(u_r, flip_pool)
        u_k_pool = _pool_1d(u_k, flip_pool)
        delta = int(it.get("delta") or 0)
        diff = float((u_r_pool - u_k_pool).item())
        gap = float((-delta) * diff)
        gaps.append(gap)
        n_removed_list.append(int(u_r.numel()))
        n_kept_list.append(int(u_k.numel()))
        size_bias_list.append(float(math.log(float(u_k.numel()) / float(u_r.numel()))))
        if gap > 0:
            correct += 1
            if delta == 1:
                correct_pos += 1
            elif delta == -1:
                correct_neg += 1
        total += 1
        if delta == 1:
            total_pos += 1
            gaps_pos.append(gap)
            diffs_pos.append(diff)
        elif delta == -1:
            total_neg += 1
            gaps_neg.append(gap)
            diffs_neg.append(diff)

        # Diagnostics only: mean pairwise gap (-delta*(u_r_i-u_k_j)), approximated by sampling.
        nr = int(u_r.numel())
        nk = int(u_k.numel())
        n_pairs = nr * nk
        if nr > 0 and nk > 0 and n_pairs > 0 and n_pairs <= 2_000_000:
            max_pairs = 64
            gap_flat = ((-float(delta)) * (u_r.view(nr, 1) - u_k.view(1, nk))).reshape(-1)
            if gap_flat.numel() > max_pairs:
                perm = torch.randperm(int(gap_flat.numel()), device=u.device)[:max_pairs]
                gap_flat = gap_flat[perm]
            pairwise_gap_means.append(float(gap_flat.mean().item()))
        masses.append(float(torch.softmax(u, dim=0)[removed_mask].sum().item()))
    pair_acc = float(correct / max(1, total))
    pair_acc_pos = float(correct_pos / max(1, total_pos)) if total_pos else float("nan")
    pair_acc_neg = float(correct_neg / max(1, total_neg)) if total_neg else float("nan")
    removed_mass = float(np.mean(masses)) if masses else float("nan")
    gap_mean = float(np.mean(gaps)) if gaps else float("nan")
    gap_pos_mean = float(np.mean(gaps_pos)) if gaps_pos else float("nan")
    gap_neg_mean = float(np.mean(gaps_neg)) if gaps_neg else float("nan")
    mean_n_removed = float(np.mean(n_removed_list)) if n_removed_list else float("nan")
    mean_n_kept = float(np.mean(n_kept_list)) if n_kept_list else float("nan")
    size_bias = float(np.mean(size_bias_list)) if size_bias_list else float("nan")
    diff_pos_mean = float(np.mean(diffs_pos)) if diffs_pos else float("nan")
    diff_neg_mean = float(np.mean(diffs_neg)) if diffs_neg else float("nan")
    pairwise_gap_mean = float(np.mean(pairwise_gap_means)) if pairwise_gap_means else float("nan")
    return {
        "pair_acc": pair_acc,
        "pair_acc_pos": pair_acc_pos,
        "pair_acc_neg": pair_acc_neg,
        "n_pos": float(total_pos),
        "n_neg": float(total_neg),
        "removed_mass": removed_mass,
        "gap_mean": gap_mean,
        "gap_pos_mean": gap_pos_mean,
        "gap_neg_mean": gap_neg_mean,
        "mean_n_removed": mean_n_removed,
        "mean_n_kept": mean_n_kept,
        "size_bias": size_bias,
        "diff_pos_mean": diff_pos_mean,
        "diff_neg_mean": diff_neg_mean,
        "pairwise_gap_mean": pairwise_gap_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train scorer (stable teacher; bag-BCE + flip loss).")
    parser.add_argument("--logs_root", "--data_root", dest="logs_root", type=str, default=os.path.join("logs", "mmlu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_role", type=float, default=1e-3)
    parser.add_argument("--lr_topo", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--flip_batch_size", type=int, default=1)
    parser.add_argument(
        "--flip_mode",
        type=str,
        default="pairwise",
        choices=["pool", "pairwise"],
        help="Flip loss mode: pool uses set pooling (u_r_pool-u_k_pool); pairwise uses all removed-vs-kept pairs.",
    )
    parser.add_argument(
        "--pairwise_max_pairs",
        type=int,
        default=64,
        help="When --flip_mode pairwise, sample at most this many (removed,kept) pairs per item.",
    )
    parser.add_argument(
        "--flip_pool",
        type=str,
        default="lme",
        help=(
            "Flip pooling over removed/kept sets: lse=logsumexp, lme=logmeanexp, max=max, mean=mean, "
            "topkK=mean(top-K). Example: topk3."
        ),
    )
    parser.add_argument(
        "--flip_w_neg",
        type=float,
        default=1.0,
        help="Extra weight multiplier for delta=-1 flip pairs (default 1.0 keeps old weighting).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default=os.path.join("logs", "mmlu", "scorer", "best.pt"))
    parser.add_argument("--select_metric", type=str, default="auc", choices=["auc", "f1"])
    parser.add_argument("--allow_missing_parent", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hf_model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--micro_batch", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256, help="Unified fused embedding dim.")
    parser.add_argument("--gate_init", type=float, default=0.1, help="Init value for scalar role/topo gates.")
    parser.add_argument("--proj_seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.2, help="Default flip margin (used if margin_pos/neg unset).")
    parser.add_argument("--margin_pos", type=float, default=None, help="Flip margin for delta=+1 (default: --margin).")
    parser.add_argument("--margin_neg", type=float, default=None, help="Flip margin for delta=-1 (default: --margin).")
    parser.add_argument(
        "--margin_sign",
        type=float,
        default=None,
        help="Optional sign margin: if set, use softplus(margin_sign - gap); else softplus(-gap).",
    )
    parser.add_argument("--lambda_flip", type=float, default=0.5)
    parser.add_argument("--lambda_sign", type=float, default=0.2)
    parser.add_argument("--lambda_scale", type=float, default=0.01, help="Weight for mean(u_raw) stabilization loss.")
    parser.add_argument("--scale_huber_delta", type=float, default=1.0, help="Huber delta for mean(u_raw) stabilization.")
    parser.add_argument("--bag_only_epochs", type=int, default=5, help="First N epochs train bag loss only.")
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Linearly warm up flip/sign weights for N epochs after bag_only_epochs.",
    )
    parser.add_argument(
        "--cls_cache_path",
        type=str,
        default="",
        help="SQLite cache path for frozen cross-encoder CLS vectors (empty -> auto under save_path dir).",
    )
    parser.add_argument("--no_cls_cache", action="store_true", help="Disable CLS cache (always run cross-encoder).")
    parser.add_argument("--train_encoder", action="store_true", help="If set, unfreeze the cross-encoder (slow; disables CLS cache).")
    parser.add_argument("--no_calibrate_after_train", action="store_true", help="Disable post-train calibration writeback.")
    parser.add_argument("--calib_q", type=float, default=0.5, help="Quantile for u_center (default median).")
    parser.add_argument("--calib_scale_q", type=float, default=0.9, help="Quantile for |u-u_center| to compute u_scale.")
    parser.add_argument("--calib_clip_u", type=float, default=20.0, help="Clip range for calibrated u_centered.")
    parser.add_argument("--calib_T", type=float, default=1.5, help="Temperature for y_del=sigmoid(u_centered/T).")
    args = parser.parse_args()

    freeze_encoder = not bool(getattr(args, "train_encoder", False))

    use_cls_cache = not bool(getattr(args, "no_cls_cache", False))
    cls_cache_path = str(getattr(args, "cls_cache_path", "") or "").strip()
    if use_cls_cache and not cls_cache_path:
        model_tag = str(getattr(args, "hf_model", "hf")).replace("/", "_").replace(":", "_")
        cls_cache_path = os.path.join(
            os.path.dirname(str(getattr(args, "save_path", os.path.join("logs", "mmlu", "scorer", "best.pt")))),
            f"cls_cache_{model_tag}_L{int(getattr(args, 'max_length', 256))}.sqlite",
        )
    if not use_cls_cache:
        cls_cache_path = ""
    if use_cls_cache and not freeze_encoder:
        print("[WARN] disabling CLS cache because encoder is trainable (--train_encoder).")
        cls_cache_path = ""
        use_cls_cache = False

    cfg = Config(
        logs_root=str(args.logs_root),
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.lr),
        lr_role=float(args.lr_role),
        lr_topo=float(args.lr_topo),
        batch_size=int(args.batch_size),
        flip_batch_size=int(args.flip_batch_size),
        flip_mode=str(args.flip_mode),
        pairwise_max_pairs=int(args.pairwise_max_pairs),
        flip_pool=str(args.flip_pool),
        flip_w_neg=float(args.flip_w_neg),
        device=str(args.device),
        save_path=str(args.save_path),
        select_metric=str(args.select_metric),
        allow_missing_parent=bool(args.allow_missing_parent),
        num_workers=int(args.num_workers),
        hf_model=str(args.hf_model),
        sbert_model=str(args.sbert_model),
        max_length=int(args.max_length),
        micro_batch=int(args.micro_batch),
        d_model=int(args.d_model),
        gate_init=float(args.gate_init),
        proj_seed=int(args.proj_seed),
        use_cls_cache=bool(use_cls_cache),
        cls_cache_path=str(cls_cache_path),
        bag_only_epochs=int(args.bag_only_epochs),
        warmup_epochs=int(args.warmup_epochs),
        margin=float(args.margin),
        margin_pos=float(args.margin_pos) if args.margin_pos is not None else float(args.margin),
        margin_neg=float(args.margin_neg) if args.margin_neg is not None else float(args.margin),
        margin_sign=(float(args.margin_sign) if args.margin_sign is not None else None),
        lambda_flip=float(args.lambda_flip),
        lambda_sign=float(args.lambda_sign),
        lambda_scale=float(args.lambda_scale),
        scale_huber_delta=float(args.scale_huber_delta),
        freeze_encoder=bool(freeze_encoder),
        calibrate_after_train=not bool(getattr(args, "no_calibrate_after_train", False)),
        calib_q=float(args.calib_q),
        calib_scale_q=float(args.calib_scale_q),
        calib_clip_u=float(args.calib_clip_u),
        calib_T=float(args.calib_T),
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    strict_parent = not cfg.allow_missing_parent
    schema = SchemaConfig()
    runs_all, base_by_run_id, examples_by_run_id, nodes_by_run_id, role_vocab = build_all_examples(
        logs_root=cfg.logs_root, schema=schema, strict_parent=strict_parent
    )
    # Per user request: no test split; use all runs for training + reporting.
    train_runs = list(runs_all)
    val_runs: List[Any] = []
    test_runs: List[Any] = []

    train_q = QuestionDataset(
        runs=train_runs,
        examples_by_run_id=examples_by_run_id,
        base_by_run_id=base_by_run_id,
        nodes_by_run_id=nodes_by_run_id,
        strict_parent=strict_parent,
    )

    train_cf = [r for r in train_runs if r.parent_run_id and r.source in {"true", "false"}]
    train_flip = FlipDataset(
        cf_runs=train_cf,
        examples_by_run_id=examples_by_run_id,
        base_by_run_id=base_by_run_id,
        nodes_by_run_id=nodes_by_run_id,
        strict_parent=strict_parent,
    )

    print(f"[Data] runs={len(runs_all)} train={len(train_runs)} val=0 test=0")
    print(f"[Flip] train={len(train_flip)} val=0 test=0")

    topo_mean, topo_std = compute_topo_norm(train_q, batch_size=32, num_workers=cfg.num_workers)
    print(f"[TopoNorm] dim={int(topo_mean.numel())} mean0={float(topo_mean[0]):.4f} std0={float(topo_std[0]):.4f}")

    t0 = time.perf_counter()
    embedder = FrozenEmbedder(
        role_vocab=role_vocab,
        hf_model=cfg.hf_model,
        sbert_model=cfg.sbert_model,
        max_length=cfg.max_length,
        micro_batch=cfg.micro_batch,
        d_model=cfg.d_model,
        gate_init=cfg.gate_init,
        topo_mean=topo_mean,
        topo_std=topo_std,
        proj_seed=cfg.proj_seed,
        device=device,
        freeze_encoder=bool(cfg.freeze_encoder),
        cls_cache_path=(str(cfg.cls_cache_path) if bool(cfg.use_cls_cache) else None),
    )
    t1 = time.perf_counter()
    dz = int(cfg.d_model)
    print(f"[EmbedderV4] H={embedder.h_dim} d={dz} init_sec={(t1-t0):.2f} freeze_encoder={cfg.freeze_encoder}")
    print(
        f"[Gates] gate_init={cfg.gate_init:.3f} gate_role={float(embedder.gate_role.detach().cpu().item()):.3f} "
        f"gate_topo={float(embedder.gate_topo.detach().cpu().item()):.3f}"
    )
    if cfg.use_cls_cache and cfg.freeze_encoder:
        print(f"[CLSCache] enabled path={cfg.cls_cache_path}")
    else:
        print("[CLSCache] disabled")

    head = ScorerHead(dz).to(device)
    run_bias = nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
    param_groups = [
        {
            "params": list(head.parameters())
            + [run_bias]
            + list(embedder.msg_proj.parameters())
            + [embedder.gate_role, embedder.gate_topo]
            + list(embedder.fuse_ln.parameters()),
            "lr": float(cfg.lr),
        },
        {"params": list(embedder.role_proj.parameters()), "lr": float(cfg.lr_role)},
        {"params": list(embedder.topo_mlp.parameters()), "lr": float(cfg.lr_topo)},
    ]
    if not bool(cfg.freeze_encoder):
        enc_params = [p for p in embedder.encoder.parameters() if getattr(p, "requires_grad", False)]
        if enc_params:
            param_groups.append({"params": enc_params, "lr": float(cfg.lr) * 0.1})
            print("[Train] encoder params are trainable; CLS cache should be disabled.")
    opt = torch.optim.AdamW(param_groups)

    # Bag-level class imbalance: pos_weight = n_neg / n_pos (train).
    n_pos = sum(1 for r in train_runs if int(getattr(r, "y", 0)) == 1)
    n_neg = sum(1 for r in train_runs if int(getattr(r, "y", 0)) == 0)
    pos_weight = float(n_neg / max(1, n_pos)) if (n_pos > 0 and n_neg > 0) else 1.0
    print(f"[BagBCE] n_pos={n_pos} n_neg={n_neg} pos_weight={pos_weight:.4f}")
    loss_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

    q_loader = DataLoader(
        train_q,
        batch_size=max(1, cfg.batch_size),
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_question,
    )

    # Flip balanced sampler (WeightedRandomSampler).
    flip_loader = None
    if len(train_flip) > 0:
        deltas = [int(train_flip[i]["delta"]) for i in range(len(train_flip))]
        n_pos = sum(1 for d in deltas if d == 1)
        n_neg = sum(1 for d in deltas if d == -1)
        weights = []
        for d in deltas:
            if d == 1 and n_pos > 0:
                weights.append(1.0 / float(n_pos))
            elif d == -1 and n_neg > 0:
                weights.append(1.0 / float(n_neg))
            else:
                weights.append(1.0)
        num_samples = max(1, len(q_loader) * max(1, cfg.flip_batch_size))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        flip_loader = DataLoader(
            train_flip,
            batch_size=max(1, cfg.flip_batch_size),
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_flip,
        )
        print(f"[FlipBalance] n_pos={n_pos} n_neg={n_neg} sampler=WeightedRandomSampler")
    else:
        print("[FlipBalance] train_flip is empty; flip losses disabled.")

    best_score = -1e9

    for epoch in range(1, cfg.epochs + 1):
        embedder.reset_cache_stats()
        # Linear warmup for auxiliary losses (flip/sign).
        if epoch <= int(cfg.bag_only_epochs):
            warm = 0.0
        else:
            denom = max(1, int(cfg.warmup_epochs))
            warm = min(1.0, float(epoch - int(cfg.bag_only_epochs)) / float(denom))
        lambda_flip_eff = float(cfg.lambda_flip) * float(warm)
        lambda_sign_eff = float(cfg.lambda_sign) * float(warm)

        head.train()
        loss_bag_sum = 0.0
        loss_flip_sum = 0.0
        loss_sign_sum = 0.0
        loss_scale_sum = 0.0
        n_steps = 0
        flip_iter = iter(flip_loader) if flip_loader is not None else None

        t_ep0 = time.perf_counter()
        epoch_iter = q_loader
        if tqdm is not None:
            epoch_iter = tqdm(
                q_loader,
                total=len(q_loader),
                desc=f"Epoch {epoch}/{cfg.epochs}",
                dynamic_ncols=True,
            )
        for q_batch in epoch_iter:
            n_steps += 1
            z, meta = embedder.embed_question_batch(q_batch)
            logits = head(z)
            u = (logits[:, 1] - logits[:, 0]).view(-1)
            b_idx = meta["batch_idx"].to(device=u.device)
            bsz = len(q_batch["run_id"])
            U = bag_logsumexp(u, b_idx, bsz)
            S = run_bias.to(device=u.device, dtype=u.dtype) - U
            y = q_batch["y"].to(device=u.device, dtype=torch.float32)
            loss_bag = loss_bce(S, y)

            loss_scale = torch.tensor(0.0, device=u.device)
            if float(cfg.lambda_scale) > 0.0 and u.numel() > 0:
                mean_u = u.mean()
                loss_scale = F.huber_loss(
                    mean_u,
                    mean_u.new_tensor(0.0),
                    delta=float(cfg.scale_huber_delta),
                )

            loss_flip = torch.tensor(0.0, device=u.device)
            loss_sign = torch.tensor(0.0, device=u.device)
            if flip_iter is not None and (lambda_flip_eff > 0.0 or lambda_sign_eff > 0.0):
                try:
                    f_batch = next(flip_iter)
                except StopIteration:
                    flip_iter = iter(flip_loader)
                    f_batch = next(flip_iter)
                items = f_batch["items"]
                lf_list: List[torch.Tensor] = []
                ls_list: List[torch.Tensor] = []
                for it in items:
                    zf = embedder.embed_flip_item(it)
                    logf = head(zf)
                    uf = (logf[:, 1] - logf[:, 0]).view(-1)
                    delta = int(it.get("delta") or 0)
                    lf, ls = flip_loss_terms(
                        u=uf,
                        removed_idx=it.get("removed_idx") or [],
                        delta=delta,
                        margin_pos=cfg.margin_pos,
                        margin_neg=cfg.margin_neg,
                        margin_sign=cfg.margin_sign,
                        flip_pool=cfg.flip_pool,
                        flip_w_neg=cfg.flip_w_neg,
                        flip_mode=cfg.flip_mode,
                        pairwise_max_pairs=cfg.pairwise_max_pairs,
                    )
                    lf_list.append(lf)
                    ls_list.append(ls)
                if lf_list:
                    loss_flip = torch.stack(lf_list).mean()
                if ls_list:
                    loss_sign = torch.stack(ls_list).mean()

            loss = loss_bag + float(cfg.lambda_scale) * loss_scale + float(lambda_flip_eff) * loss_flip + float(lambda_sign_eff) * loss_sign
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params_to_clip: List[torch.Tensor] = []
            seen: set[int] = set()
            for group in opt.param_groups:
                for p in group.get("params", []):
                    if p is None or not getattr(p, "requires_grad", False):
                        continue
                    pid = id(p)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    params_to_clip.append(p)
            if params_to_clip:
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0)
            opt.step()

            loss_bag_sum += float(loss_bag.item())
            loss_flip_sum += float(loss_flip.item())
            loss_sign_sum += float(loss_sign.item())
            loss_scale_sum += float(loss_scale.item())
            if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    loss=float(loss.item()),
                    bag=float(loss_bag.item()),
                    flip=float(loss_flip.item()),
                    sign=float(loss_sign.item()),
                    scale=float(loss_scale.item()),
                    warm=float(warm),
                )

        t_ep1 = time.perf_counter()

        head.eval()
        # No val/test set; choose a threshold on train and report train@0.5 and train@bestF1.
        train_y, train_p, train_s = _run_level_outputs(
            train_q,
            embedder=embedder,
            head=head,
            run_bias=run_bias,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        best_thr, best_train_f1 = _choose_threshold_max_f1(train_y, train_p)
        print(f"[Thr] train_best_f1_thr={best_thr:.3f} train_best_f1={best_train_f1:.4f}")

        train_run_at_05 = _run_metrics(y_true=train_y, y_prob=train_p, s_b=train_s, threshold=0.5, name="Train@0.5")
        train_run = _run_metrics(y_true=train_y, y_prob=train_p, s_b=train_s, threshold=best_thr, name="Train@bestF1")
        train_flip_stats = eval_flip_level(train_flip, embedder=embedder, head=head, flip_pool=cfg.flip_pool)

        train_loss_bag = loss_bag_sum / max(1, n_steps)
        train_loss_flip = loss_flip_sum / max(1, n_steps)
        train_loss_sign = loss_sign_sum / max(1, n_steps)
        train_loss_scale = loss_scale_sum / max(1, n_steps)
        train_loss_total = (
            train_loss_bag
            + float(cfg.lambda_scale) * train_loss_scale
            + float(lambda_flip_eff) * train_loss_flip
            + float(lambda_sign_eff) * train_loss_sign
        )

        score = float(train_run.get(cfg.select_metric, float("nan")))
        improved = isinstance(score, float) and not math.isnan(score) and score > best_score
        if improved:
            best_score = score
            ensure_parent_dir(cfg.save_path)
            ckpt = {
                "version": 4,
                "head_state_dict": head.state_dict(),
                "run_bias": float(run_bias.detach().cpu().item()),
                "msg_proj_state_dict": embedder.msg_proj.state_dict(),
                "role_proj_state_dict": embedder.role_proj.state_dict(),
                "topo_mlp_state_dict": embedder.topo_mlp.state_dict(),
                "gate_role": float(embedder.gate_role.detach().cpu().item()),
                "gate_topo": float(embedder.gate_topo.detach().cpu().item()),
                "fuse_ln_state_dict": embedder.fuse_ln.state_dict(),
                "best_threshold": float(best_thr),
                "config": asdict(cfg),
                "role_vocab": {"id2role": list(role_vocab.id2role)},
                "topo_norm": {"mean": topo_mean, "std": topo_std},
                "calibration": None,
            }
            if not bool(cfg.freeze_encoder):
                ckpt["encoder_state_dict"] = {k: v.detach().cpu() for k, v in embedder.encoder.state_dict().items()}
            torch.save(ckpt, cfg.save_path)
            with open(os.path.join(os.path.dirname(cfg.save_path), "config.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss_total:.4f} "
            f"(bag={train_loss_bag:.4f} flip={train_loss_flip:.4f} sign={train_loss_sign:.4f} scale={train_loss_scale:.4f}) "
            f"warm={warm:.3f} lf={lambda_flip_eff:.3f} ls={lambda_sign_eff:.3f} "
            f"bias={float(run_bias.detach().cpu().item()):.3f} "
            f"gate_r={float(embedder.gate_role.detach().cpu().item()):.3f} "
            f"gate_t={float(embedder.gate_topo.detach().cpu().item()):.3f} sec={(t_ep1-t_ep0):.1f}"
        )
        print(
            f"  [Train run@bestF1] acc={train_run['acc']:.4f} f1={train_run['f1']:.4f} auc={train_run['auc']:.4f} "
            f"| [Train run@0.5] acc={train_run_at_05['acc']:.4f} f1={train_run_at_05['f1']:.4f} auc={train_run_at_05['auc']:.4f} "
            f"| [Train flip] pair_acc={train_flip_stats['pair_acc']:.4f} "
            f"(+1={train_flip_stats['pair_acc_pos']:.4f} -1={train_flip_stats['pair_acc_neg']:.4f} "
            f"n+={int(train_flip_stats['n_pos'])} n-={int(train_flip_stats['n_neg'])}) "
            f"gap_mean={train_flip_stats['gap_mean']:.3f} "
            f"gap(+1)={train_flip_stats['gap_pos_mean']:.3f} gap(-1)={train_flip_stats['gap_neg_mean']:.3f} "
            f"diff(+1)={train_flip_stats['diff_pos_mean']:.3f} diff(-1)={train_flip_stats['diff_neg_mean']:.3f} "
            f"|R|={train_flip_stats['mean_n_removed']:.1f} |K|={train_flip_stats['mean_n_kept']:.1f} "
            f"removed_mass={train_flip_stats['removed_mass']:.4f} size_bias={train_flip_stats['size_bias']:.3f} "
            f"pairwise_gap_mean={train_flip_stats['pairwise_gap_mean']:.3f}"
        )
        st = embedder.cache_stats()
        if st is not None:
            hits = int(st.get("hits") or 0)
            misses = int(st.get("misses") or 0)
            puts = int(st.get("puts") or 0)
            denom = max(1, hits + misses)
            hit_rate = float(hits) / float(denom)
            print(f"  [CLSCache] hit={hits} miss={misses} hit_rate={hit_rate:.3f} puts={puts}")
        if improved:
            print(f"  [CKPT] saved best {cfg.select_metric}={best_score:.4f} to {cfg.save_path}")

    if cfg.calibrate_after_train:
        try:
            diag = calibrate_checkpoint_inplace(
                cfg.save_path,
                question_ds=train_q,
                device=device,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                q_center=cfg.calib_q,
                q_scale=cfg.calib_scale_q,
                clip_u=cfg.calib_clip_u,
                T=cfg.calib_T,
            )
            diag_path = os.path.join(os.path.dirname(cfg.save_path), "calibration_diagnostics.json")
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2, ensure_ascii=False)
            print(f"[Calibration] wrote ckpt calibration + diagnostics={diag_path}")
            print("[CalibrationDiag] " + json.dumps(diag, ensure_ascii=False))
        except Exception as e:
            print(f"[WARN] calibration failed: {e}")

    print(f"[Done] best_{cfg.select_metric}={best_score:.4f} ckpt={cfg.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
