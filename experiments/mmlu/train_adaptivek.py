"""Train an Adaptive-K edge selection model for MMLU."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from CMAD.graph.graph import Graph
from CMAD.graph.graph_final import EdgeMLP, EdgeMLPConfig, GCNEmbed

from experiments.mmlu.dataset import (
    _compute_edge_position_features,
    EDGE_TYPE_SPATIAL,
    load_json,
    load_nodes_index,
    PHI_TOPO_DIM,
    SchemaConfig,
)
from experiments.mmlu.data_prep import (
    DistillRecord,
    load_baseline_records,
    load_phase1_records,
    select_balanced_records,
)
from experiments.mmlu.trainscorer import FrozenScorer


def _candidate_edge_flats(graph: Graph) -> List[int]:
    try:
        return list(getattr(graph, "_candidate_edge_flats")())  # type: ignore[misc]
    except Exception:
        n = int(graph.num_nodes)
        mask = graph.spatial_masks.detach().cpu().view(-1)
        out: List[int] = []
        for src in range(n):
            for dst in range(src + 1, n):
                flat = src * n + dst
                if flat >= int(mask.numel()):
                    continue
                if float(mask[flat].item()) <= 0.5:
                    continue
                out.append(int(flat))
        return out


def _build_teacher_cache_from_records(
    records: List[DistillRecord],
    *,
    baseline_root: str,
    phase1_root: str,
    scorer_ckpt: str,
    scorer_device: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    if cache_path and os.path.isfile(cache_path):
        try:
            cache = torch.load(cache_path)
            if cache.get("examples"):
                return cache
        except Exception:
            pass

    teacher = FrozenScorer(ckpt_path=scorer_ckpt, device=scorer_device or "cuda")
    schema = SchemaConfig()

    nodes_by_run: Dict[str, Any] = {}
    for root_dir in [baseline_root, phase1_root]:
        nodes_path = os.path.join(root_dir, "nodes.jsonl")
        if os.path.isfile(nodes_path):
            nodes_by_run.update(load_nodes_index(nodes_path))

    examples: Dict[str, Any] = {}
    it = tqdm(records, desc="TeacherCache") if tqdm is not None else records
    for rec in it:
        transcript_abs: Optional[str] = None
        for base in [baseline_root, phase1_root]:
            for subdir in ["", "transcripts"]:
                candidate = os.path.join(base, subdir, rec.transcript_path)
                if os.path.isfile(candidate):
                    transcript_abs = candidate
                    break
                candidate = os.path.join(base, subdir, os.path.basename(rec.transcript_path))
                if os.path.isfile(candidate):
                    transcript_abs = candidate
                    break
            if transcript_abs:
                break
        if not transcript_abs:
            continue

        try:
            transcript = load_json(transcript_abs)
        except Exception:
            continue

        nodes_idx = nodes_by_run.get(rec.run_id)
        if nodes_idx is None:
            continue

        raw_messages = transcript.get(schema.messages_field, [])
        if not isinstance(raw_messages, list):
            continue

        node_ids = [str(n.node_id) for n in nodes_idx.ordered]
        idx_of = {nid: i for i, nid in enumerate(node_ids)}
        n = len(node_ids)
        if n <= 0:
            continue

        edge_feat_map = _compute_edge_position_features(raw_messages, schema)
        edge_flat: List[int] = []
        texts: List[str] = []
        src_role_ids: List[int] = []
        dst_role_ids: List[int] = []
        topo_vecs: List[torch.Tensor] = []

        def _role_to_id(role: str) -> int:
            role = str(role or "Unknown")
            if role in teacher.role_vocab.role2id:
                return int(teacher.role_vocab.role2id[role])
            if "Unknown" in teacher.role_vocab.role2id:
                return int(teacher.role_vocab.role2id["Unknown"])
            return 0

        seen_flat: set[int] = set()
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            if m.get(schema.msg_edge_type_field) != EDGE_TYPE_SPATIAL:
                continue
            src_id = str(m.get(schema.msg_src_id_field) or "")
            dst_id = str(m.get(schema.msg_dst_id_field) or "")
            if not src_id or not dst_id:
                continue
            if src_id not in idx_of or dst_id not in idx_of:
                continue
            src_idx = int(idx_of[src_id])
            dst_idx = int(idx_of[dst_id])
            if src_idx == dst_idx:
                continue
            flat = int(src_idx * n + dst_idx)
            if flat in seen_flat:
                continue
            seen_flat.add(flat)

            text = m.get(schema.msg_text_field, "")
            text = "" if text is None else str(text)
            src_role = str(m.get(schema.msg_src_role_field) or getattr(nodes_idx.by_id.get(src_id), "role", "Unknown"))
            dst_role = str(m.get(schema.msg_dst_role_field) or getattr(nodes_idx.by_id.get(dst_id), "role", "Unknown"))

            edge_flat.append(flat)
            texts.append(text)
            src_role_ids.append(_role_to_id(src_role))
            dst_role_ids.append(_role_to_id(dst_role))

            phi = edge_feat_map.get((src_id, dst_id))
            if phi is None:
                phi = torch.zeros((PHI_TOPO_DIM,), dtype=torch.float32)
            topo_vecs.append(phi.to(dtype=torch.float32))

        if not edge_flat:
            continue

        keys = [f"{rec.run_id}::{i}" for i in range(len(edge_flat))]
        batch = {
            "q_text": [str(rec.question)],
            "texts": [texts],
            "keys": [keys],
            "qid": [str(rec.question_id)],
            "run_id": [str(rec.run_id)],
            "mask": torch.ones((1, len(texts)), dtype=torch.bool),
            "src_role_id": torch.tensor(src_role_ids, dtype=torch.long).view(1, -1),
            "dst_role_id": torch.tensor(dst_role_ids, dtype=torch.long).view(1, -1),
            "topo": torch.stack(topo_vecs, dim=0).view(1, -1, PHI_TOPO_DIM),
        }

        out = teacher.predict_question_batch(batch)
        u_centered = out.get("u_centered")
        if not isinstance(u_centered, torch.Tensor) or u_centered.numel() == 0:
            continue

        u_centered_t = u_centered.detach().cpu().float().view(-1)
        examples[rec.run_id] = {
            "run_id": rec.run_id,
            "question_id": rec.question_id,
            "question": rec.question,
            "agent_num": rec.agent_num,
            "source": rec.source,
            "pos_edge_flat": list(map(int, edge_flat)),
            "pos_u_centered": u_centered_t.tolist(),
            "n_nodes": n,
        }

    cache = {"examples": examples}
    if cache_path:
        try:
            torch.save(cache, cache_path)
        except Exception:
            pass
    return cache


@dataclass(frozen=True)
class _TrainRun:
    uid: str
    question: str
    pos_edge_flat: List[int]
    pos_u_centered: List[float]


def train_adaptive_k(
    graph: Graph,
    *,
    baseline_root: str,
    phase1_root: str,
    num_questions: int = 40,
    scorer_ckpt: str,
    scorer_device: Optional[str] = None,
    output_dir: str,
    mode_filter: Optional[str] = "FullConnected",
    agent_num: int = 6,
    device: str = "cuda",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    teacher_temperature: float = 0.7,
    teacher_u_alpha: float = 1.0,
    difficulty_mode: str = "score_mean",
    k_mapping_mode: str = "percentile",
    k_min: int = 3,
    k_max: int = 10,
    target_k: int = 6,
    model_score_mode: str = "edge_mlp",
    edge_mlp_hidden: int = 128,
    edge_mlp_input: str = "prod",
    sim_mode: str = "dot",
    logit_temp: float = 1.0,
    logit_clip: Optional[float] = None,
    use_gcn_embed: bool = True,
    seed: int = 42,
) -> Tuple[str, Dict[str, Any]]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, "teacher_cache_adaptivek.pt")

    baseline_records = load_baseline_records(
        baseline_root,
        mode_filter=str(mode_filter or ""),
        agent_num_filter=int(agent_num),
    )
    phase1_records = load_phase1_records(
        phase1_root,
        mode_filter=str(mode_filter or ""),
        agent_num_filter=int(agent_num),
    )
    selected = select_balanced_records(baseline_records, phase1_records, total=int(num_questions), seed=int(seed))
    cache = _build_teacher_cache_from_records(
        selected,
        baseline_root=str(baseline_root),
        phase1_root=str(phase1_root),
        scorer_ckpt=str(scorer_ckpt),
        scorer_device=scorer_device,
        cache_path=cache_path,
    )
    examples = dict(cache.get("examples") or {})
    if not examples:
        raise RuntimeError("Teacher cache is empty.")

    dev = torch.device(str(device))
    n = int(agent_num)

    candidate_idx = _candidate_edge_flats(graph)
    if not candidate_idx:
        raise RuntimeError("No candidate edges found.")
    candidate_set = set(map(int, candidate_idx))
    cand_t = torch.tensor(candidate_idx, dtype=torch.long, device=dev)
    cand_cnt = int(len(candidate_idx))

    model_score_mode = str(model_score_mode).strip().lower()
    if model_score_mode not in {"edge_mlp", "similarity"}:
        model_score_mode = "edge_mlp"
    sim_mode = str(sim_mode).strip().lower()
    if sim_mode not in {"dot", "cosine"}:
        sim_mode = "dot"

    logit_temp = float(max(1e-6, float(logit_temp)))
    if logit_clip is not None and float(logit_clip) <= 0:
        logit_clip = None

    if use_gcn_embed:
        try:
            in_ch = int(graph.gcn.conv1.in_channels)
            hid_ch = int(graph.gcn.conv1.out_channels)
            out_ch = int(graph.gcn.conv2.out_channels)
            graph.gcn = GCNEmbed(in_ch, hid_ch, out_ch, use_residual=True)
        except Exception:
            use_gcn_embed = False

    graph.gcn.to(dev)
    graph.mlp.to(dev)
    graph.gcn.train()
    graph.mlp.train()
    graph.features = graph.features.to(dev)
    graph.role_adj_matrix = graph.role_adj_matrix.to(dev)

    if not hasattr(graph, "spatial_logit_bias"):
        graph.spatial_logit_bias = nn.Parameter(torch.zeros((), dtype=torch.float32, device=dev))
    else:
        old = getattr(graph, "spatial_logit_bias")
        if isinstance(old, torch.Tensor):
            graph.spatial_logit_bias = nn.Parameter(old.detach().to(device=dev, dtype=torch.float32))
        else:
            graph.spatial_logit_bias = nn.Parameter(torch.tensor(float(old), device=dev, dtype=torch.float32))

    edge_mlp: Optional[EdgeMLP] = None
    edge_mlp_use_prod = str(edge_mlp_input).strip().lower() == "prod"
    if model_score_mode == "edge_mlp":
        with torch.no_grad():
            q0 = torch.zeros((384,), device=dev, dtype=torch.float32)
            q_rep0 = q0.view(1, -1).repeat(int(n), 1)
            x0 = torch.cat([graph.features, q_rep0], dim=1)
            h0 = graph.gcn(x0, graph.role_adj_matrix)
            z0 = graph.mlp(h0)
            node_dim = int(z0.size(1))
        edge_mlp = EdgeMLP(
            EdgeMLPConfig(
                node_dim=node_dim,
                query_dim=384,
                hidden_dim=int(edge_mlp_hidden),
                use_prod=bool(edge_mlp_use_prod),
            )
        ).to(dev)
        setattr(graph, "edge_mlp", edge_mlp)

    runs: List[_TrainRun] = []
    for uid, rec in examples.items():
        rec_agent_num = rec.get("agent_num")
        if rec_agent_num is not None and int(rec_agent_num) != int(agent_num):
            continue

        pos_edges_all = list(map(int, rec.get("pos_edge_flat") or []))
        pos_u_centered_all = list(map(float, rec.get("pos_u_centered") or []))

        pos_edges: List[int] = []
        pos_u_centered: List[float] = []
        for i, ef in enumerate(pos_edges_all):
            if int(ef) not in candidate_set:
                continue
            pos_edges.append(int(ef))
            pos_u_centered.append(float(pos_u_centered_all[i]) if i < len(pos_u_centered_all) else 0.0)

        if not pos_edges:
            continue
        runs.append(
            _TrainRun(
                uid=str(uid),
                question=str(rec.get("question") or ""),
                pos_edge_flat=pos_edges,
                pos_u_centered=pos_u_centered,
            )
        )

    if not runs:
        raise RuntimeError("No training runs found.")

    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_texts = [r.question for r in runs]
    q_emb_np = sbert.encode(q_texts, batch_size=32, show_progress_bar=False)
    q_emb = torch.tensor(np.asarray(q_emb_np), dtype=torch.float32, device=dev)

    def forward_edge_logits(q_vec: torch.Tensor) -> torch.Tensor:
        q_rep = q_vec.view(1, -1).repeat(n, 1)
        x = torch.cat([graph.features, q_rep], dim=1)
        h = graph.gcn(x, graph.role_adj_matrix)
        z = graph.mlp(h)

        if sim_mode == "cosine":
            z = F.normalize(z, dim=-1, eps=1e-6)

        if model_score_mode == "edge_mlp" and edge_mlp is not None:
            z_src = z.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
            z_dst = z.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
            q_flat = q_vec.view(1, -1).expand(n * n, -1)
            prod = (z_src * z_dst) if edge_mlp_use_prod else None
            edge_logits = edge_mlp(z_src, z_dst, q_flat, prod=prod).view(-1)
        else:
            edge_logits = (z @ z.t()).flatten()

        edge_logits = edge_logits / logit_temp
        edge_logits = edge_logits + graph.spatial_logit_bias
        if logit_clip is not None:
            edge_logits = torch.clamp(edge_logits, min=-float(logit_clip), max=float(logit_clip))
        return edge_logits

    params: List[nn.Parameter] = []
    params += list(graph.gcn.parameters())
    params += list(graph.mlp.parameters())
    if edge_mlp is not None:
        params += list(edge_mlp.parameters())
    params += [graph.spatial_logit_bias]
    optimizer = torch.optim.Adam(params, lr=float(lr))

    teacher_T = float(max(1e-6, float(teacher_temperature)))
    steps_per_epoch = int(math.ceil(len(runs) / float(max(1, int(batch_size)))))

    metrics_log: List[Dict[str, Any]] = []
    for epoch in range(int(epochs)):
        order = list(range(len(runs)))
        random.shuffle(order)

        epoch_loss = 0.0
        it = range(steps_per_epoch)
        if tqdm is not None:
            it = tqdm(it, total=steps_per_epoch, desc=f"AdaptiveK {epoch+1}/{epochs}")

        for step in it:
            batch_ids = order[step * int(batch_size) : (step + 1) * int(batch_size)]
            if not batch_ids:
                continue

            loss_list: List[torch.Tensor] = []
            for j in batch_ids:
                run = runs[j]
                q_vec = q_emb[j]
                edge_logits = forward_edge_logits(q_vec)

                ef_all = list(map(int, run.pos_edge_flat))
                u_c = torch.tensor(run.pos_u_centered, dtype=torch.float32, device=dev)
                u_eff = torch.clamp(u_c * float(teacher_u_alpha), min=-30.0, max=30.0)
                y_del = torch.sigmoid(u_eff / teacher_T).clamp(0.0, 1.0)
                y_keep = (1.0 - y_del).clamp(0.0, 1.0)

                logits_pos = edge_logits[torch.tensor(ef_all, dtype=torch.long, device=dev)]
                loss_list.append(F.binary_cross_entropy_with_logits(logits_pos, y_keep, reduction="mean"))

            loss_batch = torch.stack(loss_list).mean()
            optimizer.zero_grad()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss_batch.detach().cpu().item())
            if hasattr(it, "set_postfix"):
                it.set_postfix({"loss": f"{float(loss_batch.item()):.4f}"})

        metrics = {"epoch": epoch + 1, "loss_mean": epoch_loss / max(1, steps_per_epoch)}
        metrics_log.append(metrics)

    graph.gcn.eval()
    graph.mlp.eval()
    if edge_mlp is not None:
        edge_mlp.eval()

    difficulty_mode = str(difficulty_mode).strip().lower()
    if difficulty_mode not in {"score_sum", "score_mean", "top_score_sum"}:
        difficulty_mode = "score_mean"

    scores: List[float] = []
    with torch.no_grad():
        for j in range(len(runs)):
            logits_all = forward_edge_logits(q_emb[j])
            cand_logits = logits_all[cand_t]
            if difficulty_mode == "score_sum":
                score = float(cand_logits.sum().item())
            elif difficulty_mode == "top_score_sum":
                topk_vals, _ = torch.topk(cand_logits, k=min(int(target_k), int(cand_logits.numel())))
                score = float(topk_vals.sum().item())
            else:
                score = float(cand_logits.mean().item())
            scores.append(score)

    scores_arr = np.asarray(scores, dtype=np.float64)
    percentiles = {f"p{p}": float(np.percentile(scores_arr, p)) for p in (10, 20, 30, 40, 50, 60, 70, 80, 90)}
    stats = {
        "mean": float(scores_arr.mean()),
        "std": float(scores_arr.std()),
        "min": float(scores_arr.min()),
        "max": float(scores_arr.max()),
        "count": int(scores_arr.size),
    }

    with open(os.path.join(output_dir, "difficulty_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "percentiles": percentiles,
                "stats": stats,
                "difficulty_mode": difficulty_mode,
                "k_mapping_mode": str(k_mapping_mode),
                "k_min": int(k_min),
                "k_max": int(k_max),
                "target_k": int(target_k),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    gcn_config: Dict[str, Any] = {}
    try:
        gcn_config = {
            "in_channels": int(graph.gcn.conv1.in_channels),
            "hidden_channels": int(graph.gcn.conv1.out_channels),
            "out_channels": int(graph.gcn.conv2.out_channels),
        }
    except Exception:
        gcn_config = {}

    ckpt_data: Dict[str, Any] = {
        "gcn_state_dict": graph.gcn.state_dict(),
        "mlp_state_dict": graph.mlp.state_dict(),
        "spatial_logit_bias": graph.spatial_logit_bias.detach().cpu(),
        "model_score_mode": model_score_mode,
        "sim_mode": sim_mode,
        "logit_temp": float(logit_temp),
        "logit_clip": logit_clip,
        "use_gcn_embed": bool(use_gcn_embed),
        "gcn_config": gcn_config,
        "difficulty_mode": difficulty_mode,
        "k_mapping_mode": str(k_mapping_mode),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "target_k": int(target_k),
        "percentiles": percentiles,
        "stats": stats,
    }

    if edge_mlp is not None:
        ckpt_data["edge_mlp_state_dict"] = edge_mlp.state_dict()
        ckpt_data["edge_mlp_config"] = {
            "node_dim": int(edge_mlp.config.node_dim),
            "query_dim": int(edge_mlp.config.query_dim),
            "hidden_dim": int(edge_mlp.config.hidden_dim),
            "use_prod": bool(edge_mlp.config.use_prod),
        }

    ckpt_path = os.path.join(output_dir, "adaptive_k_model.pt")
    torch.save(ckpt_data, ckpt_path)

    return output_dir, {
        "metrics": metrics_log,
        "difficulty_stats": {"percentiles": percentiles, "stats": stats},
        "num_runs": int(len(runs)),
        "num_candidates": int(cand_cnt),
    }
