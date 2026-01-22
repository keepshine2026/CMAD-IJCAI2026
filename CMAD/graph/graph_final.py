from __future__ import annotations

"""Graph utilities used by the MMLU Adaptive-K pipeline."""

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from CMAD.graph.graph import Graph

@dataclass(frozen=True)
class EdgeMLPConfig:
    node_dim: int
    query_dim: int
    hidden_dim: int = 128
    use_prod: bool = True  # include z_i*z_j term


class EdgeMLP(torch.nn.Module):
    """Query-conditioned edge scorer."""

    def __init__(self, config: EdgeMLPConfig):
        super().__init__()
        self.config = config

        in_dim = int(config.node_dim) * 2 + int(config.query_dim)
        if bool(config.use_prod):
            in_dim += int(config.node_dim)

        self.fc1 = torch.nn.Linear(in_dim, int(config.hidden_dim))
        self.fc2 = torch.nn.Linear(int(config.hidden_dim), 1)

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        q: torch.Tensor,
        *,
        prod: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_src.ndim != 2 or z_dst.ndim != 2 or q.ndim != 2:
            raise ValueError("EdgeMLP expects 2D tensors: z_src[M,D], z_dst[M,D], q[M,Dq].")
        if z_src.shape != z_dst.shape:
            raise ValueError(f"z_src and z_dst shape mismatch: {tuple(z_src.shape)} vs {tuple(z_dst.shape)}")
        if z_src.shape[0] != q.shape[0]:
            raise ValueError(f"batch size mismatch: z_src M={z_src.shape[0]} vs q M={q.shape[0]}")

        parts = [z_src, z_dst]
        if self.config.use_prod:
            if prod is None:
                prod = z_src * z_dst
            parts.append(prod)
        parts.append(q)
        x = torch.cat(parts, dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).view(-1)

class GCNEmbed(torch.nn.Module):
    """Embedding GCN used by offline trainers."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, use_residual: bool = True):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.use_residual = bool(use_residual)
        if self.use_residual and int(in_channels) != int(out_channels):
            self.residual_proj = torch.nn.Linear(int(in_channels), int(out_channels))
        else:
            self.residual_proj = None

    def reset_parameters(self) -> None:
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.residual_proj is not None:
            self.residual_proj.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity
        return x

class GraphFinal(Graph):
    """Graph + deterministic TopK + dagsample ordering."""

    spatial_select_mode: str = "topk"

    candidate_mode: str = "upper_triangle"
    acyclic: bool = False

    edge_score_mode: str = "logit"
    edge_score_source: str = "spatial_logits"

    model_score_mode: str = "sim"
    sim_mode: str = "dot"
    logit_temp: float = 1.0
    logit_clip: Optional[float] = None

    k_source: str = "head"
    keep_ratio_fixed: float = 0.35
    keep_ratio_min: float = 0.20
    keep_ratio_max: float = 0.80
    rule_k0: float = 0.35
    rule_alpha: float = 0.25
    dagsample_tau: float = 1.0
    keep_ratio_head: Optional[torch.nn.Module] = None
    edge_mlp: Optional[torch.nn.Module] = None
    edge_mlp_input: str = "prod"

    last_keep_ratio: Optional[float] = None
    last_keep_k: Optional[int] = None
    last_num_candidates: Optional[int] = None
    last_selected_edges: Optional[List[Tuple[int, int]]] = None
    _last_query_embedding: Optional[torch.Tensor] = None  # [Dq]

    def construct_new_features(self, query: str) -> torch.Tensor:
        features = super().construct_new_features(query)
        try:
            d_role = int(self.features.size(1))
            self._last_query_embedding = features[0, d_role:].detach()
        except Exception:
            self._last_query_embedding = None
        return features

    def _compute_edge_scores_from_model(self) -> Optional[torch.Tensor]:
        q_embed = getattr(self, "_last_query_embedding", None)
        if not isinstance(q_embed, torch.Tensor) or q_embed.numel() <= 0:
            return None

        n = int(self.num_nodes)
        q_node = q_embed.view(1, -1).repeat(n, 1)
        x = torch.cat([self.features, q_node.to(device=self.features.device, dtype=self.features.dtype)], dim=1)

        h = self.gcn(x, self.role_adj_matrix)
        z = self.mlp(h)

        model_score_mode = str(getattr(self, "model_score_mode", "sim")).strip().lower()
        if model_score_mode not in {"sim", "edge_mlp"}:
            model_score_mode = "sim"

        logits: torch.Tensor
        if model_score_mode == "edge_mlp":
            use_prod = str(getattr(self, "edge_mlp_input", "prod")).strip().lower() == "prod"
            edge_mlp = getattr(self, "edge_mlp", None)
            if edge_mlp is None:
                print("[GraphFinal][WARN] model_score_mode=edge_mlp but graph.edge_mlp is None; falling back to sim.")
                model_score_mode = "sim"
            else:
                if isinstance(edge_mlp, torch.nn.Module):
                    edge_mlp = edge_mlp.to(device=z.device, dtype=z.dtype)

                z_src = z.unsqueeze(1).expand(n, n, -1).reshape(n * n, -1)
                z_dst = z.unsqueeze(0).expand(n, n, -1).reshape(n * n, -1)
                q_flat = q_embed.view(1, -1).expand(n * n, -1).to(device=z.device, dtype=z.dtype)
                prod = (z_src * z_dst) if use_prod else None
                logits = edge_mlp(z_src, z_dst, q_flat, prod=prod).view(-1)

        if model_score_mode == "sim":
            sim_mode = str(getattr(self, "sim_mode", "dot")).strip().lower()
            if sim_mode == "cosine":
                z = F.normalize(z, dim=-1, eps=1e-6)
            sim = z @ z.t()
            logits = sim.flatten()

        temp = float(getattr(self, "logit_temp", 1.0))
        if temp <= 0:
            temp = 1.0
        logits = logits / float(temp)

        bias = getattr(self, "spatial_logit_bias", None)
        if isinstance(bias, torch.Tensor) and bias.numel() == 1:
            logits = logits + bias.to(device=logits.device, dtype=logits.dtype)
        elif bias is not None:
            try:
                logits = logits + float(bias)
            except Exception:
                pass

        clip = getattr(self, "logit_clip", None)
        if clip is not None:
            logits = torch.clamp(logits, min=-float(clip), max=float(clip))

        return logits

    def _candidate_edge_flats(self) -> List[int]:
        n = int(self.num_nodes)
        mask = self.spatial_masks.detach().view(-1)
        out: List[int] = []
        mode = str(getattr(self, "candidate_mode", "upper_triangle")).strip().lower()
        if mode not in {"upper_triangle", "full"}:
            mode = "upper_triangle"

        if mode == "upper_triangle":
            for src in range(n):
                for dst in range(src + 1, n):
                    flat = src * n + dst
                    if flat >= int(mask.numel()):
                        continue
                    if float(mask[flat].item()) <= 0.5:
                        continue
                    out.append(int(flat))
        else:
            for src in range(n):
                for dst in range(n):
                    if src == dst:
                        continue
                    flat = src * n + dst
                    if flat >= int(mask.numel()):
                        continue
                    if float(mask[flat].item()) <= 0.5:
                        continue
                    out.append(int(flat))
        return out

    def _compute_keep_ratio(self, *, edge_scores: torch.Tensor) -> float:
        k_source = str(getattr(self, "k_source", "head")).strip().lower()
        k_min = float(getattr(self, "keep_ratio_min", 0.20))
        k_max = float(getattr(self, "keep_ratio_max", 0.80))
        if k_max < k_min:
            k_min, k_max = k_max, k_min

        keep_ratio: Optional[float] = None
        if k_source == "head":
            head = getattr(self, "keep_ratio_head", None)
            q = getattr(self, "_last_query_embedding", None)
            if head is not None and isinstance(q, torch.Tensor) and q.numel() > 0:
                try:
                    head_device = next(head.parameters()).device
                except Exception:
                    head_device = q.device
                q_in = q.to(device=head_device).view(1, -1)
                with torch.no_grad():
                    pred = head(q_in).view(-1)[0]
                keep_ratio = float(pred.detach().cpu().item())
            else:
                k_source = "rule"

        if keep_ratio is None and k_source == "rule":
            p = torch.sigmoid(edge_scores.detach())
            eps = 1e-8
            ent = -(p * (p + eps).log() + (1.0 - p) * (1.0 - p + eps).log()).mean()
            keep_ratio = float(getattr(self, "rule_k0", 0.35)) + float(getattr(self, "rule_alpha", 0.25)) * float(
                ent.detach().cpu().item()
            )

        if keep_ratio is None:
            keep_ratio = float(getattr(self, "keep_ratio_fixed", 0.35))

        keep_ratio = float(max(k_min, min(k_max, keep_ratio)))
        return keep_ratio

    def _gumbel_noise(self, shape: torch.Size, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        eps = 1e-8
        u = torch.rand(shape, device=device, dtype=dtype)
        u = u.clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(u))

    def _dagsample_order(self, *, candidate_flats: List[int], scores: torch.Tensor) -> torch.Tensor:
        if not candidate_flats or scores.numel() <= 0:
            return torch.empty((0,), device=scores.device, dtype=torch.long)

        tau = float(getattr(self, "dagsample_tau", 1.0))
        tau = float(max(0.0, tau))
        noise = self._gumbel_noise(scores.shape, device=scores.device, dtype=scores.dtype)
        noisy = scores + noise * tau
        return torch.argsort(noisy, descending=True)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None) -> torch.Tensor:
        mode = str(getattr(self, "spatial_select_mode", "topk")).strip().lower()
        if mode not in {"topk", "dagsample"}:
            return super().construct_spatial_connection(temperature=temperature, threshold=threshold)

        self.clear_spatial_connection()

        score_source = str(getattr(self, "edge_score_source", "spatial_logits")).strip().lower()
        spatial_logits: Optional[torch.Tensor]
        if score_source == "model":
            spatial_logits = self._compute_edge_scores_from_model()
        else:
            spatial_logits = getattr(self, "spatial_logits", None)
        if not isinstance(spatial_logits, torch.Tensor) or spatial_logits.numel() == 0:
            self.last_keep_ratio = None
            self.last_keep_k = None
            self.last_num_candidates = 0
            self.last_selected_edges = []
            return torch.tensor(0.0)

        device = spatial_logits.device
        dtype = spatial_logits.dtype

        candidate_flats = self._candidate_edge_flats()
        self.last_num_candidates = int(len(candidate_flats))
        if not candidate_flats:
            self.last_keep_ratio = 0.0
            self.last_keep_k = 0
            self.last_selected_edges = []
            return torch.zeros((), device=device, dtype=dtype)

        cand_t = torch.tensor(candidate_flats, dtype=torch.long, device=device)
        scores = spatial_logits[cand_t]
        if str(getattr(self, "edge_score_mode", "logit")).strip().lower() == "prob":
            scores = torch.sigmoid(scores)

        keep_ratio = self._compute_keep_ratio(edge_scores=scores)
        k = int(math.ceil(float(keep_ratio) * float(len(candidate_flats))))
        k = max(1, min(int(len(candidate_flats)), int(k)))
        self.last_keep_ratio = float(keep_ratio)
        self.last_keep_k = int(k)

        node_ids: List[str] = list(self.nodes.keys())
        n = int(self.num_nodes)

        selected_edges: List[Tuple[int, int]] = []
        candidate_mode = str(getattr(self, "candidate_mode", "upper_triangle")).strip().lower()
        acyclic = bool(getattr(self, "acyclic", False)) and candidate_mode == "full"

        if mode == "topk":
            if not acyclic:
                top = torch.topk(scores, k=int(k), largest=True).indices
                chosen_flats = [int(candidate_flats[int(i)]) for i in top.detach().cpu().tolist()]
                for flat in chosen_flats:
                    src = int(flat // n)
                    dst = int(flat % n)
                    if src == dst:
                        continue
                    out_node = self.find_node(node_ids[src])
                    in_node = self.find_node(node_ids[dst])
                    out_node.add_successor(in_node, "spatial")
                    selected_edges.append((src, dst))
            else:
                order = torch.argsort(scores, descending=True)
                added = 0
                for idx in order.detach().cpu().tolist():
                    if added >= int(k):
                        break
                    flat = int(candidate_flats[int(idx)])
                    src = int(flat // n)
                    dst = int(flat % n)
                    if src == dst:
                        continue
                    out_node = self.find_node(node_ids[src])
                    in_node = self.find_node(node_ids[dst])
                    if self.check_cycle(in_node, {out_node}):
                        continue
                    out_node.add_successor(in_node, "spatial")
                    selected_edges.append((src, dst))
                    added += 1
                if added < int(k):
                    print(f"[GraphFinal][WARN] acyclic=True could not reach K={k}; added={added} (candidate_mode=full).")
        else:
            order = self._dagsample_order(candidate_flats=candidate_flats, scores=scores)
            if not acyclic:
                top = order[: int(k)]
                for idx in top.detach().cpu().tolist():
                    flat = int(candidate_flats[int(idx)])
                    src = int(flat // n)
                    dst = int(flat % n)
                    if src == dst:
                        continue
                    out_node = self.find_node(node_ids[src])
                    in_node = self.find_node(node_ids[dst])
                    out_node.add_successor(in_node, "spatial")
                    selected_edges.append((src, dst))
            else:
                added = 0
                for idx in order.detach().cpu().tolist():
                    if added >= int(k):
                        break
                    flat = int(candidate_flats[int(idx)])
                    src = int(flat // n)
                    dst = int(flat % n)
                    if src == dst:
                        continue
                    out_node = self.find_node(node_ids[src])
                    in_node = self.find_node(node_ids[dst])
                    if self.check_cycle(in_node, {out_node}):
                        continue
                    out_node.add_successor(in_node, "spatial")
                    selected_edges.append((src, dst))
                    added += 1
                if added < int(k):
                    print(f"[GraphFinal][WARN] dagsample acyclic=True could not reach K={k}; added={added}.")

        self.last_selected_edges = selected_edges
        return torch.zeros((), device=device, dtype=dtype)

    def get_run_metadata(self) -> dict[str, Any]:
        meta = super().get_run_metadata() if hasattr(super(), "get_run_metadata") else {}
        meta = dict(meta or {})
        meta["keep_ratio"] = self.last_keep_ratio
        meta["keep_k"] = self.last_keep_k
        meta["num_candidate_edges"] = self.last_num_candidates
        meta["selected_edges"] = self.last_selected_edges
        meta["candidate_mode"] = str(getattr(self, "candidate_mode", "upper_triangle"))
        meta["k_source"] = str(getattr(self, "k_source", "head"))
        meta["spatial_select_mode"] = str(getattr(self, "spatial_select_mode", "topk"))
        meta["dagsample_tau"] = float(getattr(self, "dagsample_tau", 1.0))
        return meta


__all__ = [
    "EdgeMLP",
    "EdgeMLPConfig",
    "GCNEmbed",
    "GraphFinal",
]
