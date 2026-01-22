#!/usr/bin/env python
"""Dataset utilities for MMLU experiment logs.

Builds run-level and flip-level samples from log folders under `logs/mmlu/*`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


EDGE_TYPE_SPATIAL = "spatial"
PHI_TOPO_DIM = 13  # matches message_contrib_features.compute_position_features_from_transcript()

DEFAULT_ROLE_ORDER = [
    "Knowlegable Expert",
    "Critic",
    "Mathematician",
    "Psychologist",
    "Historian",
    "Doctor",
    "Lawyer",
    "Economist",
    "Programmer",
]


def _default_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strtobool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().strip('"').lower()
    return text in {"1", "true", "t", "yes", "y"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _task_num(qid: str) -> int:
    m = re.search(r"(\d+)", qid or "")
    return int(m.group(1)) if m else 10**18


def _resolve_existing_path(candidates: Sequence[str]) -> str:
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"File not found. Tried: {list(candidates)}")


def resolve_transcript_path(runs_csv_path: str, transcript_rel_path: str) -> str:
    base_dir = os.path.dirname(runs_csv_path)
    candidates = [
        os.path.normpath(os.path.join(base_dir, transcript_rel_path)),
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", transcript_rel_path)),
    ]
    if os.path.isabs(transcript_rel_path):
        candidates.insert(0, os.path.normpath(transcript_rel_path))
    return _resolve_existing_path(candidates)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class SchemaConfig:
    # runs.csv
    run_id_field: str = "run_id"
    parent_run_id_field: str = "parent_run_id"
    question_id_field: str = "question_id"
    question_text_field: str = "question"
    is_correct_field: str = "is_correct"
    predicted_answer_field: str = "predicted_answer"
    correct_answer_field: str = "correct_answer"
    mode_field: str = "mode"
    agent_num_field: str = "agent_num"
    transcript_path_field: str = "transcript_path"
    removed_edges_field: str = "removed_edges"
    # transcript messages
    messages_field: str = "messages"
    msg_text_field: str = "content"
    msg_edge_type_field: str = "edge_type"
    msg_src_id_field: str = "src_id"
    msg_dst_id_field: str = "dst_id"
    msg_src_role_field: str = "src_role"
    msg_dst_role_field: str = "dst_role"
    msg_idx_field: str = "sequence_id"


@dataclass(frozen=True)
class NodeTopo:
    node_id: str
    role: str
    in_degree: int
    out_degree: int


@dataclass(frozen=True)
class NodeTopoIndex:
    ordered: List[NodeTopo]
    by_id: Dict[str, NodeTopo]

    @property
    def ordered_node_ids(self) -> List[str]:
        return [n.node_id for n in self.ordered]


def load_nodes_index(nodes_jsonl_path: str) -> Dict[str, NodeTopoIndex]:
    by_run: Dict[str, List[NodeTopo]] = {}
    for rec in load_jsonl(nodes_jsonl_path):
        if _safe_int(rec.get("round"), default=-1) != 0:
            continue
        run_id = str(rec.get("run_id") or "")
        if not run_id:
            continue
        node_id = str(rec.get("node_id") or "")
        role = str(rec.get("role") or "Unknown")
        by_run.setdefault(run_id, []).append(
            NodeTopo(
                node_id=node_id,
                role=role,
                in_degree=_safe_int(rec.get("in_degree"), default=0),
                out_degree=_safe_int(rec.get("out_degree"), default=0),
            )
        )
    out: Dict[str, NodeTopoIndex] = {}
    for run_id, ordered in by_run.items():
        out[run_id] = NodeTopoIndex(ordered=ordered, by_id={n.node_id: n for n in ordered})
    return out


class RoleVocab:
    def __init__(self, initial_roles: Optional[Sequence[str]] = None):
        self.role2id: Dict[str, int] = {}
        self.id2role: List[str] = []
        for r in initial_roles or []:
            self.get_id(r)

    def get_id(self, role: str) -> int:
        role = str(role or "Unknown")
        if role not in self.role2id:
            self.role2id[role] = len(self.id2role)
            self.id2role.append(role)
        return self.role2id[role]

    def __len__(self) -> int:
        return len(self.id2role)


@dataclass(frozen=True)
class RunMeta:
    source: str  # baseline / true / false
    run_id: str
    parent_run_id: Optional[str]
    question_id: str
    question: str
    mode: str
    agent_num: int
    y: int
    predicted_answer: str
    correct_answer: str
    transcript_abs_path: str
    removed_edges: List[Dict[str, Any]]


@dataclass(frozen=True)
class MessageItem:
    key: str
    text: str
    src_id: str
    dst_id: str
    src_role: str
    dst_role: str
    src_role_id: int
    dst_role_id: int
    phi_topo: torch.Tensor  # float32 [d_topo]
    msg_idx: int


@dataclass(frozen=True)
class RunExample:
    meta: RunMeta
    messages: List[MessageItem]
    key2idx: Dict[str, int]
    edge_to_keys_id: Dict[Tuple[str, str], List[str]]
    edge_to_keys_role: Dict[Tuple[str, str], List[str]]
    topo_missing: int


def _parse_removed_edges(payload: Any) -> List[Dict[str, Any]]:
    if not payload:
        return []
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return []
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, list):
        return []
    return [x for x in payload if isinstance(x, dict)]


def _make_msg_key(qid: str, run_id: str, src_id: str, dst_id: str, msg_idx: int) -> str:
    return f"{qid}::{run_id}::{src_id}::{dst_id}::{msg_idx}"


def _multi_source_shortest_path(adj: List[List[int]], sources: List[int]) -> List[int]:
    from collections import deque

    n = len(adj)
    inf = 10**9
    dist = [inf] * n
    dq: deque[int] = deque()
    for s in sources:
        if 0 <= s < n:
            dist[s] = 0
            dq.append(s)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            cand = dist[u] + 1
            if cand < dist[v]:
                dist[v] = cand
                dq.append(v)
    max_finite = max((d for d in dist if d < inf), default=0)
    return [d if d < inf else max_finite + 1 for d in dist]


def _reachable_count(adj: List[List[int]], start: int) -> int:
    n = len(adj)
    if not (0 <= start < n):
        return 0
    visited = [False] * n
    stack = [start]
    visited[start] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    return sum(visited) - 1


def _compute_edge_position_features(
    raw_messages: Sequence[Dict[str, Any]],
    schema: SchemaConfig,
) -> Dict[Tuple[str, str], torch.Tensor]:
    """
    Same spirit as message_contrib_features.compute_position_features_from_transcript().
    Returns edge features keyed by (src_id, dst_id), each vector dim=PHI_TOPO_DIM.
    """
    edges_by_id: Set[Tuple[str, str]] = set()
    node_ids: Set[str] = set()
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        if msg.get(schema.msg_edge_type_field) != EDGE_TYPE_SPATIAL:
            continue
        src_id = msg.get(schema.msg_src_id_field)
        dst_id = msg.get(schema.msg_dst_id_field)
        if not src_id or not dst_id:
            continue
        src_id = str(src_id)
        dst_id = str(dst_id)
        edges_by_id.add((src_id, dst_id))
        node_ids.add(src_id)
        node_ids.add(dst_id)

    if not node_ids or not edges_by_id:
        return {}

    node_list = sorted(node_ids)
    idx_of = {nid: i for i, nid in enumerate(node_list)}
    n = len(node_list)

    edges = [(idx_of[s], idx_of[d]) for (s, d) in edges_by_id if s in idx_of and d in idx_of]
    if not edges:
        return {}

    adj: List[List[int]] = [[] for _ in range(n)]
    rev: List[List[int]] = [[] for _ in range(n)]
    for src_idx, dst_idx in edges:
        adj[src_idx].append(dst_idx)
        rev[dst_idx].append(src_idx)

    in_deg = [len(rev[i]) for i in range(n)]
    out_deg = [len(adj[i]) for i in range(n)]

    sources = [i for i in range(n) if in_deg[i] == 0] or list(range(n))
    sinks = [i for i in range(n) if out_deg[i] == 0] or list(range(n))

    depth = _multi_source_shortest_path(adj, sources)
    dist_to_sink = _multi_source_shortest_path(rev, sinks)

    denom = max(1, n - 1)
    node_feat: List[List[float]] = []
    for i in range(n):
        anc = _reachable_count(rev, i)
        desc = _reachable_count(adj, i)
        node_feat.append(
            [
                in_deg[i] / denom,
                out_deg[i] / denom,
                depth[i] / denom,
                dist_to_sink[i] / denom,
                anc / denom,
                desc / denom,
            ]
        )

    edge_feat: Dict[Tuple[str, str], torch.Tensor] = {}
    for src_id, dst_id in edges_by_id:
        src_idx = idx_of[src_id]
        dst_idx = idx_of[dst_id]
        delta_depth = (depth[dst_idx] - depth[src_idx]) / denom
        vec = torch.tensor(node_feat[src_idx] + node_feat[dst_idx] + [delta_depth], dtype=torch.float32)
        if int(vec.numel()) != PHI_TOPO_DIM:
            # Should never happen, but keep it robust.
            padded = torch.zeros((PHI_TOPO_DIM,), dtype=torch.float32)
            n_copy = min(PHI_TOPO_DIM, int(vec.numel()))
            padded[:n_copy] = vec[:n_copy]
            vec = padded
        edge_feat[(src_id, dst_id)] = vec
    return edge_feat


def build_run_example(
    *,
    meta: RunMeta,
    transcript: Dict[str, Any],
    schema: SchemaConfig,
    role_vocab: RoleVocab,
    nodes_index: Optional[NodeTopoIndex],
) -> RunExample:
    raw_messages = transcript.get(schema.messages_field, [])
    if not isinstance(raw_messages, list):
        raw_messages = []

    edge_feat_map = _compute_edge_position_features(raw_messages, schema)
    topo_missing = 0

    msgs: List[MessageItem] = []
    edge_to_keys_id: Dict[Tuple[str, str], List[str]] = {}
    edge_to_keys_role: Dict[Tuple[str, str], List[str]] = {}

    for pos, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            continue
        if m.get(schema.msg_edge_type_field) != EDGE_TYPE_SPATIAL:
            continue

        msg_idx = _safe_int(m.get(schema.msg_idx_field), default=pos)
        src_id = str(m.get(schema.msg_src_id_field) or "")
        dst_id = str(m.get(schema.msg_dst_id_field) or "")
        src_role = str(m.get(schema.msg_src_role_field) or "Unknown")
        dst_role = str(m.get(schema.msg_dst_role_field) or "Unknown")
        text = m.get(schema.msg_text_field, "")
        text = "" if text is None else str(text)

        src_role_id = role_vocab.get_id(src_role)
        dst_role_id = role_vocab.get_id(dst_role)

        phi_topo = edge_feat_map.get((src_id, dst_id))
        if phi_topo is None:
            topo_missing += 1
            phi_topo = torch.zeros((PHI_TOPO_DIM,), dtype=torch.float32)
        key = _make_msg_key(meta.question_id, meta.run_id, src_id or src_role, dst_id or dst_role, msg_idx)
        msgs.append(
            MessageItem(
                key=key,
                text=text,
                src_id=src_id,
                dst_id=dst_id,
                src_role=src_role,
                dst_role=dst_role,
                src_role_id=src_role_id,
                dst_role_id=dst_role_id,
                phi_topo=phi_topo,
                msg_idx=msg_idx,
            )
        )
        edge_to_keys_id.setdefault((src_id, dst_id), []).append(key)
        edge_to_keys_role.setdefault((src_role, dst_role), []).append(key)

    key2idx = {m.key: i for i, m in enumerate(msgs)}
    return RunExample(
        meta=meta,
        messages=msgs,
        key2idx=key2idx,
        edge_to_keys_id=edge_to_keys_id,
        edge_to_keys_role=edge_to_keys_role,
        topo_missing=topo_missing,
    )


def map_removed_edges_to_keys(
    *,
    removed_edges: Sequence[Dict[str, Any]],
    edge_to_keys_id: Dict[Tuple[str, str], List[str]],
    edge_to_keys_role: Dict[Tuple[str, str], List[str]],
    nodes_index: Optional[NodeTopoIndex],
) -> Tuple[List[str], Dict[str, int]]:
    stats = {"n_edges": 0, "mapped_by_index": 0, "mapped_by_role": 0, "unmapped": 0}
    keys: List[str] = []
    node_order = nodes_index.ordered_node_ids if nodes_index is not None else []

    for e in removed_edges or []:
        if not isinstance(e, dict):
            continue
        stats["n_edges"] += 1
        src_idx = e.get("src_idx")
        dst_idx = e.get("dst_idx")
        src_role = e.get("src_role")
        dst_role = e.get("dst_role")

        mapped = False
        if isinstance(src_idx, int) and isinstance(dst_idx, int) and node_order:
            if 0 <= src_idx < len(node_order) and 0 <= dst_idx < len(node_order):
                src_id = node_order[src_idx]
                dst_id = node_order[dst_idx]
                k = edge_to_keys_id.get((src_id, dst_id), [])
                if k:
                    keys.extend(k)
                    stats["mapped_by_index"] += 1
                    mapped = True
        if not mapped and src_role and dst_role:
            k = edge_to_keys_role.get((str(src_role), str(dst_role)), [])
            if k:
                keys.extend(k)
                stats["mapped_by_role"] += 1
                mapped = True
        if not mapped:
            stats["unmapped"] += 1

    # dedupe keep order
    seen: Set[str] = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out, stats


def discover_log_dirs(logs_root: str) -> Dict[str, str]:
    root = os.path.abspath(logs_root)
    dirs = {
        "baseline": os.path.join(root, "baseline"),
        "true": os.path.join(root, "true_conterfactual"),
        "false": os.path.join(root, "false_conterfactual"),
    }
    missing = [k for k, v in dirs.items() if not os.path.isdir(v)]
    if missing:
        raise FileNotFoundError(f"Missing expected log dirs under {root}: {missing}")
    return dirs


def load_run_metas(log_dir: str, source: str, schema: SchemaConfig) -> Tuple[str, List[RunMeta]]:
    runs_csv = os.path.join(log_dir, "runs.csv")
    if not os.path.isfile(runs_csv):
        raise FileNotFoundError(f"runs.csv not found: {runs_csv}")

    baseline_fieldnames = [
        schema.run_id_field,
        "dataset",
        schema.question_id_field,
        schema.question_text_field,
        schema.mode_field,
        schema.agent_num_field,
        "role_mode",
        schema.is_correct_field,
        schema.predicted_answer_field,
        schema.correct_answer_field,
        "topology_id",
        "graph_path",
        schema.transcript_path_field,
    ]
    cf_fieldnames = [
        schema.run_id_field,
        schema.parent_run_id_field,
        "parent_is_correct",
        "parent_predicted_answer",
        "parent_correct_answer",
        "dataset",
        schema.question_id_field,
        schema.question_text_field,
        schema.mode_field,
        schema.agent_num_field,
        schema.is_correct_field,
        schema.predicted_answer_field,
        schema.correct_answer_field,
        "topology_id",
        "graph_path",
        schema.transcript_path_field,
        schema.removed_edges_field,
    ]

    def has_header(first_row: List[str]) -> bool:
        return any(str(x).strip().lstrip("\ufeff") == schema.run_id_field for x in first_row)

    with open(runs_csv, "r", encoding="utf-8", newline="") as f:
        reader0 = csv.reader(f)
        first_row = next(reader0, None)

    if first_row is None:
        rows: List[Dict[str, Any]] = []
    elif has_header(first_row):
        with open(runs_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        fieldnames = baseline_fieldnames if source == "baseline" else cf_fieldnames
        with open(runs_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, fieldnames=fieldnames))

    metas: List[RunMeta] = []
    for row in rows:
        run_id = str(row.get(schema.run_id_field) or "").strip()
        if not run_id:
            continue
        parent = row.get(schema.parent_run_id_field)
        parent_run_id = str(parent).strip() if parent else None

        qid = str(row.get(schema.question_id_field) or "").strip()
        qtext = str(row.get(schema.question_text_field) or "")
        mode = str(row.get(schema.mode_field) or "")
        agent_num = _safe_int(row.get(schema.agent_num_field), default=0)
        y = 1 if _strtobool(row.get(schema.is_correct_field)) else 0
        pred = str(row.get(schema.predicted_answer_field) or "")
        gold = str(row.get(schema.correct_answer_field) or "")

        t_rel = str(row.get(schema.transcript_path_field) or "").strip()
        t_abs = resolve_transcript_path(runs_csv, t_rel)

        removed_edges = _parse_removed_edges(row.get(schema.removed_edges_field))

        metas.append(
            RunMeta(
                source=source,
                run_id=run_id,
                parent_run_id=parent_run_id,
                question_id=qid,
                question=qtext,
                mode=mode,
                agent_num=agent_num,
                y=y,
                predicted_answer=pred,
                correct_answer=gold,
                transcript_abs_path=t_abs,
                removed_edges=removed_edges,
            )
        )
    return runs_csv, metas


def split_by_qid(
    runs: Sequence[RunMeta],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[RunMeta], List[RunMeta], List[RunMeta], Dict[str, List[str]]]:
    qids = sorted({r.question_id for r in runs})
    rng = random.Random(seed)
    rng.shuffle(qids)

    n = len(qids)
    n_train = max(0, min(n, int(round(n * train_ratio))))
    n_val = max(0, min(n - n_train, int(round(n * val_ratio))))

    train_q = set(qids[:n_train])
    val_q = set(qids[n_train : n_train + n_val])
    test_q = set(qids[n_train + n_val :])

    train = [r for r in runs if r.question_id in train_q]
    val = [r for r in runs if r.question_id in val_q]
    test = [r for r in runs if r.question_id in test_q]
    return train, val, test, {"train": sorted(train_q, key=_task_num), "val": sorted(val_q, key=_task_num), "test": sorted(test_q, key=_task_num)}


class QuestionDataset(Dataset):
    def __init__(
        self,
        *,
        runs: Sequence[RunMeta],
        examples_by_run_id: Dict[str, RunExample],
        base_by_run_id: Dict[str, RunMeta],
        nodes_by_run_id: Dict[str, NodeTopoIndex],
        strict_parent: bool,
    ):
        self.runs = list(runs)
        self.examples_by_run_id = examples_by_run_id
        self.base_by_run_id = base_by_run_id
        self.nodes_by_run_id = nodes_by_run_id
        self.strict_parent = strict_parent

    def __len__(self) -> int:
        return len(self.runs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.runs[idx]
        ex = self.examples_by_run_id[meta.run_id]

        neutral_removed: Optional[List[str]] = None
        neutral_removed_idx: Optional[List[int]] = None
        neutral_removed_base_run_id: Optional[str] = None
        if meta.parent_run_id:
            base = self.base_by_run_id.get(meta.parent_run_id)
            if base is None:
                if self.strict_parent:
                    raise KeyError(f"Missing parent baseline run_id={meta.parent_run_id} (cf_run_id={meta.run_id})")
            else:
                delta = int(base.y) - int(meta.y)
                if delta == 0 and meta.removed_edges:
                    # IMPORTANT:
                    # `removed_edges` are selected from the BASELINE topology (indices/roles).
                    # For Δ=0 neutral constraints, we want to identify which BASE messages were removed,
                    # otherwise the cf transcript may contain empty content on muted edges and the neutral
                    # signal becomes meaningless.
                    base_ex = self.examples_by_run_id[base.run_id]
                    keys, _ = map_removed_edges_to_keys(
                        removed_edges=meta.removed_edges,
                        edge_to_keys_id=base_ex.edge_to_keys_id,
                        edge_to_keys_role=base_ex.edge_to_keys_role,
                        nodes_index=self.nodes_by_run_id.get(base.run_id),
                    )
                    neutral_removed = keys or None
                    if neutral_removed:
                        neutral_removed_base_run_id = base.run_id
                        neutral_removed_idx = [base_ex.key2idx[k] for k in neutral_removed if k in base_ex.key2idx] or None

        return {
            "qid": meta.question_id,
            "run_id": meta.run_id,
            "parent_run_id": meta.parent_run_id,
            "q_text": meta.question,
            "y": int(meta.y),
            "messages": [
                {
                    "key": m.key,
                    "text": m.text,
                    "src_id": m.src_id,
                    "dst_id": m.dst_id,
                    "src_role": m.src_role,
                    "dst_role": m.dst_role,
                    "src_role_id": int(m.src_role_id),
                    "dst_role_id": int(m.dst_role_id),
                    "phi_topo": m.phi_topo,
                    "msg_idx": int(m.msg_idx),
                }
                for m in ex.messages
            ],
            "neutral_removed": neutral_removed,
            "neutral_removed_base_run_id": neutral_removed_base_run_id,
            "neutral_removed_idx": neutral_removed_idx,
        }


class FlipDataset(Dataset):
    def __init__(
        self,
        *,
        cf_runs: Sequence[RunMeta],
        examples_by_run_id: Dict[str, RunExample],
        base_by_run_id: Dict[str, RunMeta],
        nodes_by_run_id: Dict[str, NodeTopoIndex],
        strict_parent: bool,
    ):
        self.examples_by_run_id = examples_by_run_id
        self.base_by_run_id = base_by_run_id
        self.nodes_by_run_id = nodes_by_run_id
        self.strict_parent = strict_parent

        self.items: List[Dict[str, Any]] = []
        for cf in cf_runs:
            if not cf.parent_run_id:
                continue
            base = base_by_run_id.get(cf.parent_run_id)
            if base is None:
                if strict_parent:
                    raise KeyError(f"Missing parent baseline run_id={cf.parent_run_id} (cf_run_id={cf.run_id})")
                continue
            delta = int(base.y) - int(cf.y)
            if delta == 0:
                continue
            if not cf.removed_edges:
                continue
            self.items.append(
                {
                    "qid": cf.question_id,
                    "base_run_id": base.run_id,
                    "cf_run_id": cf.run_id,
                    "y_base": int(base.y),
                    "y_cf": int(cf.y),
                    "delta": int(delta),
                    "removed_edges": cf.removed_edges,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        base_run_id = it["base_run_id"]
        base_meta = self.base_by_run_id[base_run_id]
        base_ex = self.examples_by_run_id[base_run_id]

        removed_keys, stats = map_removed_edges_to_keys(
            removed_edges=it["removed_edges"],
            edge_to_keys_id=base_ex.edge_to_keys_id,
            edge_to_keys_role=base_ex.edge_to_keys_role,
            nodes_index=self.nodes_by_run_id.get(base_run_id),
        )
        removed_idx = [base_ex.key2idx[k] for k in removed_keys if k in base_ex.key2idx]

        return {
            "qid": it["qid"],
            "base_run_id": base_run_id,
            "cf_run_id": it["cf_run_id"],
            "q_text": base_meta.question,
            "y_base": it["y_base"],
            "y_cf": it["y_cf"],
            "delta": it["delta"],
            "base_messages": [
                {
                    "key": m.key,
                    "text": m.text,
                    "src_id": m.src_id,
                    "dst_id": m.dst_id,
                    "src_role": m.src_role,
                    "dst_role": m.dst_role,
                    "src_role_id": int(m.src_role_id),
                    "dst_role_id": int(m.dst_role_id),
                    "phi_topo": m.phi_topo,
                    "msg_idx": int(m.msg_idx),
                }
                for m in base_ex.messages
            ],
            "removed_keys": removed_keys,
            "removed_idx": removed_idx,
            "removed_map_stats": stats,
        }


def collate_question(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    bsz = len(batch)
    n_msgs = [len(x["messages"]) for x in batch]
    max_n = max(n_msgs) if n_msgs else 0
    topo_dim = 0
    for x in batch:
        if x["messages"]:
            topo_dim = int(x["messages"][0]["phi_topo"].numel())
            break

    topo = torch.zeros((bsz, max_n, topo_dim), dtype=torch.float32) if topo_dim > 0 else None
    src_role_id = torch.full((bsz, max_n), -1, dtype=torch.long)
    dst_role_id = torch.full((bsz, max_n), -1, dtype=torch.long)
    mask = torch.zeros((bsz, max_n), dtype=torch.bool)
    y = torch.tensor([int(x["y"]) for x in batch], dtype=torch.long)

    texts: List[List[str]] = [[""] * max_n for _ in range(bsz)]
    keys: List[List[str]] = [[""] * max_n for _ in range(bsz)]
    src_role: List[List[str]] = [[""] * max_n for _ in range(bsz)]
    dst_role: List[List[str]] = [[""] * max_n for _ in range(bsz)]

    for i, x in enumerate(batch):
        for j, m in enumerate(x["messages"]):
            if j >= max_n:
                break
            mask[i, j] = True
            texts[i][j] = m["text"]
            keys[i][j] = m["key"]
            src_role[i][j] = str(m.get("src_role") or "")
            dst_role[i][j] = str(m.get("dst_role") or "")
            src_role_id[i, j] = int(m["src_role_id"])
            dst_role_id[i, j] = int(m["dst_role_id"])
            if topo is not None:
                topo[i, j] = m["phi_topo"]

    return {
        "qid": [x["qid"] for x in batch],
        "run_id": [x["run_id"] for x in batch],
        "parent_run_id": [x.get("parent_run_id") for x in batch],
        "q_text": [x["q_text"] for x in batch],
        "y": y,
        "n_messages": n_msgs,
        "keys": keys,
        "texts": texts,
        "src_role": src_role,
        "dst_role": dst_role,
        "mask": mask,
        "src_role_id": src_role_id,
        "dst_role_id": dst_role_id,
        "topo": topo,
        "neutral_removed": [x.get("neutral_removed") for x in batch],
        "neutral_removed_base_run_id": [x.get("neutral_removed_base_run_id") for x in batch],
        "neutral_removed_idx": [x.get("neutral_removed_idx") for x in batch],
    }


def collate_flip(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")
    return {"items": list(batch)}


def _load_default_role_descriptions() -> Dict[str, str]:
    """
    Best-effort role description source for embedding tests.

    Uses mas_framework's MMLU prompt set if available; falls back to empty dict.
    """
    try:
        from CMAD.prompt.mmlu_prompt_set import ROLE_DESCRIPTION  # type: ignore

        return {str(k): str(v) for k, v in dict(ROLE_DESCRIPTION).items()}
    except Exception:
        return {}


def _encode_cls(
    *,
    tokenizer: Any,
    encoder: Any,
    texts_a: Sequence[str],
    texts_b: Optional[Sequence[str]] = None,
    max_length: int,
    micro_batch: int,
    device: torch.device,
) -> torch.Tensor:
    if texts_b is not None and len(texts_a) != len(texts_b):
        raise ValueError(f"texts_a/texts_b length mismatch: {len(texts_a)} vs {len(texts_b)}")
    if micro_batch <= 0:
        micro_batch = 64

    all_cls: List[torch.Tensor] = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(texts_a), micro_batch):
            end = min(len(texts_a), start + micro_batch)
            a = list(texts_a[start:end])
            if texts_b is None:
                inputs = tokenizer(
                    a,
                    truncation=True,
                    max_length=int(max_length),
                    padding=True,
                    return_tensors="pt",
                )
            else:
                b = list(texts_b[start:end])
                inputs = tokenizer(
                    a,
                    b,
                    truncation=True,
                    max_length=int(max_length),
                    padding=True,
                    return_tensors="pt",
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = encoder(**inputs)
            # DeBERTa(-v3) doesn't have pooler_output; use first token (CLS/<s>).
            cls = out.last_hidden_state[:, 0, :]
            all_cls.append(cls)
    return torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0, 0), device=device)


def _init_linear_deterministic(linear: nn.Linear, *, seed: int) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    with torch.no_grad():
        w = torch.randn(linear.weight.shape, generator=gen, device="cpu", dtype=linear.weight.dtype) * 0.02
        linear.weight.copy_(w.to(device=linear.weight.device))
        if linear.bias is not None:
            linear.bias.zero_()


def _encode_sbert(
    *,
    model_name: str,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for SBERT embedding; install it in your environment."
        ) from e

    if batch_size <= 0:
        batch_size = 64
    model = SentenceTransformer(model_name, device=str(device))
    emb = model.encode(
        list(texts),
        batch_size=int(batch_size),
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    return emb.to(device=device, dtype=torch.float32)


def embed_messages(
    batch: Dict[str, Any],
    *,
    role_vocab: RoleVocab,
    role_desc: Optional[Dict[str, str]] = None,
    model_name: str = "microsoft/deberta-v3-base",
    text_encoder: str = "sbert",  # "hf" or "sbert"
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 256,
    micro_batch: int = 64,
    d_role: int = 32,
    d_topo: int = 64,
    proj_seed: int = 0,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Embed messages in a collated QuestionDataset batch.

    Scheme (fixed):
      1) Cross-encoder CLS(Q, MSG) -> h_i in R^H
      2) Role description CLS(role_desc_text) -> r_role in R^H, then proj_role -> R^32
      3) Topology MLP(phi_topo) -> R^32
      4) z_i = concat(h_i, src_role_emb, dst_role_emb, topo_emb) in R^(H+96)

    Returns:
      z: Tensor[M, H + 2*d_role + d_topo]
      meta: mapping aligned with z rows (traceable back to original messages).
    """
    dev = _default_device(device)

    q_texts: List[str] = list(batch.get("q_text") or [])
    texts: List[List[str]] = list(batch.get("texts") or [])
    keys: List[List[str]] = list(batch.get("keys") or [])
    qids: List[str] = list(batch.get("qid") or [])
    run_ids: List[str] = list(batch.get("run_id") or [])

    mask: torch.Tensor = batch.get("mask")
    if mask is None or not isinstance(mask, torch.Tensor):
        raise ValueError("batch['mask'] is required (Tensor[B,N] bool).")
    if mask.dtype != torch.bool:
        mask = mask.bool()

    src_role_id: torch.Tensor = batch.get("src_role_id")
    dst_role_id: torch.Tensor = batch.get("dst_role_id")
    if src_role_id is None or dst_role_id is None:
        raise ValueError("batch['src_role_id'] and batch['dst_role_id'] are required (Tensor[B,N]).")

    topo: Optional[torch.Tensor] = batch.get("topo")
    if topo is None:
        topo = torch.zeros((mask.shape[0], mask.shape[1], 0), dtype=torch.float32)

    # Flatten valid messages.
    b_idx, n_idx = mask.nonzero(as_tuple=True)
    b_list = b_idx.tolist()
    n_list = n_idx.tolist()
    m = len(b_list)
    if m == 0:
        z = torch.zeros((0, 0), dtype=torch.float32, device=dev)
        return z, {"batch_idx": [], "msg_pos": [], "qid": [], "run_id": [], "key": []}

    flat_q: List[str] = [str(q_texts[b]) for b in b_list]
    flat_msg: List[str] = [str(texts[b][j]) for b, j in zip(b_list, n_list)]
    flat_key: List[str] = [str(keys[b][j]) for b, j in zip(b_list, n_list)]
    flat_qid: List[str] = [str(qids[b]) for b in b_list]
    flat_run_id: List[str] = [str(run_ids[b]) for b in b_list]

    flat_src_role_id = src_role_id[b_idx, n_idx].clamp(min=0).to(torch.long)
    flat_dst_role_id = dst_role_id[b_idx, n_idx].clamp(min=0).to(torch.long)
    flat_topo = topo[b_idx, n_idx].to(torch.float32)

    # Prepare SBERT model once (used for role desc; optionally for message embedding).
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for SBERT embedding; install it in your environment."
        ) from e

    t_sbert0 = time.perf_counter()
    sbert = SentenceTransformer(sbert_model_name, device=str(dev))
    t_sbert1 = time.perf_counter()

    def encode_sbert(texts_in: Sequence[str]) -> torch.Tensor:
        emb = sbert.encode(
            list(texts_in),
            batch_size=max(1, int(micro_batch)),
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        return emb.to(device=dev, dtype=torch.float32)

    # Role descriptions -> SBERT -> proj_role (d_role).
    role_desc = dict(role_desc or _load_default_role_descriptions())
    role_texts: List[str] = []
    for role in role_vocab.id2role:
        desc = role_desc.get(role)
        role_texts.append(f"{role}: {str(desc).strip()}" if desc else str(role))

    t_role0 = time.perf_counter()
    role_h = encode_sbert(role_texts)
    t_role1 = time.perf_counter()
    role_hidden = int(role_h.shape[1]) if role_h.ndim == 2 else 0
    if role_hidden <= 0:
        raise ValueError("Failed to compute SBERT role embedding dimension.")
    with torch.inference_mode():
        proj_role = nn.Linear(role_hidden, int(d_role), bias=True).to(dev)
        _init_linear_deterministic(proj_role, seed=int(proj_seed) + 11)
        src_role_emb = proj_role(role_h[flat_src_role_id.to(dev)])
        dst_role_emb = proj_role(role_h[flat_dst_role_id.to(dev)])

    # Topology -> fixed MLP -> d_topo.
    topo_dim = int(flat_topo.shape[-1])
    if topo_dim == 0:
        topo_emb = torch.zeros((m, int(d_topo)), dtype=torch.float32, device=dev)
    else:
        topo_hid = max(64, int(d_topo) * 2)
        lin1 = nn.Linear(topo_dim, topo_hid)
        lin2 = nn.Linear(topo_hid, int(d_topo))
        _init_linear_deterministic(lin1, seed=int(proj_seed) + 101)
        _init_linear_deterministic(lin2, seed=int(proj_seed) + 102)
        mlp_topo = nn.Sequential(lin1, nn.ReLU(), lin2).to(dev)
        t_topo0 = time.perf_counter()
        with torch.inference_mode():
            topo_emb = mlp_topo(flat_topo.to(dev))
        t_topo1 = time.perf_counter()

    # Message text embedding.
    t_load0 = time.perf_counter()
    t_load1 = t_load0
    t_msg0 = time.perf_counter()
    if str(text_encoder).lower() == "sbert":
        pair_texts = [f"Question: {q}\nMessage: {m}" for q, m in zip(flat_q, flat_msg)]
        h = encode_sbert(pair_texts)
        hidden_size = int(h.shape[1])
    elif str(text_encoder).lower() == "hf":
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for text_encoder='hf'; install it in your environment (e.g. conda/pip)."
            ) from e
        t_load0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        encoder = AutoModel.from_pretrained(model_name).to(dev)
        t_load1 = time.perf_counter()
        h = _encode_cls(
            tokenizer=tokenizer,
            encoder=encoder,
            texts_a=flat_q,
            texts_b=flat_msg,
            max_length=int(max_length),
            micro_batch=int(micro_batch),
            device=dev,
        )
        hidden_size = int(getattr(getattr(encoder, "config", None), "hidden_size", 0) or int(h.shape[1]))
    else:
        raise ValueError("text_encoder must be one of: 'sbert', 'hf'")
    t_msg1 = time.perf_counter()

    with torch.inference_mode():
        z = torch.cat([h, src_role_emb, dst_role_emb, topo_emb], dim=1)

    # Traceable metadata aligned with z rows.
    src_role_name = [role_vocab.id2role[int(i)] if int(i) < len(role_vocab) else "Unknown" for i in flat_src_role_id]
    dst_role_name = [role_vocab.id2role[int(i)] if int(i) < len(role_vocab) else "Unknown" for i in flat_dst_role_id]

    meta = {
        "batch_idx": b_idx.cpu(),
        "msg_pos": n_idx.cpu(),
        "qid": flat_qid,
        "run_id": flat_run_id,
        "key": flat_key,
        "src_role_id": flat_src_role_id.cpu(),
        "dst_role_id": flat_dst_role_id.cpu(),
        "src_role": src_role_name,
        "dst_role": dst_role_name,
        "dims": {
            "h": int(h.shape[1]) if h.ndim == 2 else 0,
            "d_role": int(d_role),
            "d_topo": int(d_topo),
            "dz": int(z.shape[1]) if z.ndim == 2 else 0,
        },
        "timing_sec": {
            "load_model": float(t_load1 - t_load0),
            "load_sbert": float(t_sbert1 - t_sbert0),
            "encode_roles": float(t_role1 - t_role0),
            "encode_messages": float(t_msg1 - t_msg0),
            "embed_topo": float((t_topo1 - t_topo0) if topo_dim > 0 else 0.0),
        },
    }
    return z, meta


def build_all_examples(
    *,
    logs_root: str,
    schema: SchemaConfig,
    strict_parent: bool,
) -> Tuple[List[RunMeta], Dict[str, RunMeta], Dict[str, RunExample], Dict[str, NodeTopoIndex], RoleVocab]:
    dirs = discover_log_dirs(logs_root)

    # nodes.jsonl -> run_id -> NodeTopoIndex
    nodes_by_run_id: Dict[str, NodeTopoIndex] = {}
    for _, log_dir in dirs.items():
        nodes_path = os.path.join(log_dir, "nodes.jsonl")
        if os.path.isfile(nodes_path):
            for rid, idx in load_nodes_index(nodes_path).items():
                nodes_by_run_id[rid] = idx

    # runs.csv
    _, baseline = load_run_metas(dirs["baseline"], "baseline", schema)
    _, true_runs = load_run_metas(dirs["true"], "true", schema)
    _, false_runs = load_run_metas(dirs["false"], "false", schema)
    runs_all = list(baseline) + list(true_runs) + list(false_runs)

    base_by_run_id: Dict[str, RunMeta] = {m.run_id: m for m in baseline}
    if strict_parent:
        for m in runs_all:
            if m.parent_run_id and m.parent_run_id not in base_by_run_id:
                raise KeyError(f"Missing parent baseline run_id={m.parent_run_id} for cf_run_id={m.run_id}")

    role_vocab = RoleVocab(DEFAULT_ROLE_ORDER)
    examples_by_run_id: Dict[str, RunExample] = {}
    for m in runs_all:
        transcript = load_json(m.transcript_abs_path)
        ex = build_run_example(
            meta=m,
            transcript=transcript,
            schema=schema,
            role_vocab=role_vocab,
            nodes_index=nodes_by_run_id.get(m.run_id),
        )
        examples_by_run_id[m.run_id] = ex

    return runs_all, base_by_run_id, examples_by_run_id, nodes_by_run_id, role_vocab


def inspect_logs(logs_root: str) -> None:
    schema = SchemaConfig()
    dirs = discover_log_dirs(logs_root)
    for name, log_dir in dirs.items():
        runs_csv = os.path.join(log_dir, "runs.csv")
        if not os.path.isfile(runs_csv):
            continue
        _, metas = load_run_metas(log_dir, name if name != "baseline" else "baseline", schema)
        row = None
        if metas:
            # Rebuild a minimal row-like dict for printing
            m0 = metas[0]
            row = {
                "run_id": m0.run_id,
                "question_id": m0.question_id,
                "transcript_path": os.path.relpath(m0.transcript_abs_path, start=os.path.dirname(runs_csv)),
            }
        print(f"\n== {name} ==")
        if not row:
            print("(empty)")
            continue
        print("example:", row)
        t_abs = metas[0].transcript_abs_path
        tr = load_json(t_abs)
        print("transcript keys:", list(tr.keys()))
        msgs = tr.get(schema.messages_field, [])
        print(f"messages={len(msgs)}")
        if msgs:
            print("first message keys:", list(msgs[0].keys()))


def inspect_dataset(data_path: str, n: int = 3) -> None:
    """
    Inspect current log folders under `data_path` (logs_root).
    """
    schema = SchemaConfig()
    dirs = discover_log_dirs(data_path)
    for name, log_dir in dirs.items():
        _, metas = load_run_metas(log_dir, name if name != "baseline" else "baseline", schema)
        print(f"\n== {name} ==")
        print(f"- runs={len(metas)} dir={os.path.abspath(log_dir)}")
        if not metas:
            continue
        for m in metas[: max(1, int(n))]:
            tr = load_json(m.transcript_abs_path)
            msgs = tr.get(schema.messages_field, [])
            first_keys = list(msgs[0].keys()) if isinstance(msgs, list) and msgs else []
            print(
                f"  - run_id={m.run_id} qid={m.question_id} y={m.y} "
                f"messages={len(msgs) if isinstance(msgs, list) else 0} first_msg_keys={first_keys}"
            )


def _print_question_batch(batch: Dict[str, Any]) -> None:
    topo_dim = int(batch["topo"].shape[-1]) if batch.get("topo") is not None else 0
    print(f"[QuestionBatch] B={len(batch['run_id'])} topo_dim={topo_dim}")
    for i, rid in enumerate(batch["run_id"]):
        has_neutral = batch["neutral_removed"][i] is not None
        print(f"  - {batch['qid'][i]} run_id={rid} n={batch['n_messages'][i]} neutral_removed={has_neutral}")


def _print_flip_batch(batch: Dict[str, Any]) -> None:
    items = batch["items"]
    deltas = [int(x["delta"]) for x in items]
    print(f"[FlipBatch] B={len(items)} delta(+1)={sum(1 for d in deltas if d==1)} delta(-1)={sum(1 for d in deltas if d==-1)}")
    for it in items:
        st = it.get("removed_map_stats") or {}
        print(
            f"  - {it['qid']} base={it['base_run_id']} cf={it['cf_run_id']} "
            f"delta={it['delta']} removed_idx={len(it.get('removed_idx') or [])} "
            f"removed_edges={st.get('n_edges')} unmapped={st.get('unmapped')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset pipeline (no training).")
    parser.add_argument("--logs_root", type=str, default=os.path.join("logs", "mmlu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--embed_test", action="store_true", help="Run one embedding forward pass on a sampled batch.")
    parser.add_argument("--embed_model", type=str, default="microsoft/deberta-v3-base", help="HF model name for text_encoder=hf")
    parser.add_argument("--embed_text_encoder", type=str, default="sbert", choices=["sbert", "hf"])
    parser.add_argument("--embed_sbert_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed_max_length", type=int, default=256)
    parser.add_argument("--embed_micro_batch", type=int, default=64)
    parser.add_argument("--embed_d_role", type=int, default=32)
    parser.add_argument("--embed_d_topo", type=int, default=64)
    parser.add_argument("--embed_proj_seed", type=int, default=0)
    parser.add_argument("--embed_device", type=str, default="", help="e.g. cuda, cuda:0, cpu (default: auto)")
    parser.add_argument(
        "--allow_missing_parent",
        action="store_true",
        help="If set, skip runs whose parent baseline run is missing (default: error).",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.logs_root, n=3)
        return 0

    strict_parent = not bool(args.allow_missing_parent)
    schema = SchemaConfig()
    runs_all, base_by_run_id, examples_by_run_id, nodes_by_run_id, role_vocab = build_all_examples(
        logs_root=args.logs_root, schema=schema, strict_parent=strict_parent
    )

    train_runs, val_runs, test_runs, split = split_by_qid(
        runs_all, seed=int(args.seed), train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio)
    )

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

    # stats
    qids_all = sorted({r.question_id for r in runs_all}, key=_task_num)
    n_cf = sum(1 for r in runs_all if r.parent_run_id)
    topo_missing = sum(ex.topo_missing for ex in examples_by_run_id.values())
    n_msgs = sum(len(ex.messages) for ex in examples_by_run_id.values())
    print(f"[Data] roles={len(role_vocab)} runs={len(runs_all)} qids={len(qids_all)} cf_runs={n_cf}")
    print(f"[Split] train_qids={len(split['train'])} val_qids={len(split['val'])} test_qids={len(split['test'])}")
    # Note: FlipDataset builds delta per sample.
    delta_pos = sum(1 for i in range(len(train_flip)) if int(train_flip[i]['delta']) == 1)
    delta_neg = sum(1 for i in range(len(train_flip)) if int(train_flip[i]['delta']) == -1)
    print(f"[Flip] train_flip={len(train_flip)} delta(+1)={delta_pos} delta(-1)={delta_neg}")
    print(
        f"[Topo] phi_topo_dim={PHI_TOPO_DIM} topo_missing_rate={(topo_missing/max(1,n_msgs)):.4f} "
        f"(missing={topo_missing} msgs={n_msgs})"
    )

    # Removed-edges -> keys mapping success (FlipDataset only).
    mapped_edges = 0
    total_removed_edges = 0
    for i in range(len(train_flip)):
        st = train_flip[i].get("removed_map_stats") or {}
        total_removed_edges += int(st.get("n_edges") or 0)
        mapped_edges += int(st.get("mapped_by_index") or 0) + int(st.get("mapped_by_role") or 0)
    if total_removed_edges > 0:
        print(f"[RemovedMap] mapped_edge_ratio={mapped_edges/total_removed_edges:.4f} (mapped={mapped_edges} total={total_removed_edges})")
    else:
        print("[RemovedMap] mapped_edge_ratio=N/A (no removed_edges in flip samples)")

    q_loader = DataLoader(train_q, batch_size=4, shuffle=True, collate_fn=collate_question)
    q_batch = next(iter(q_loader))
    _print_question_batch(q_batch)

    if args.embed_test:
        role_desc = _load_default_role_descriptions()
        t0 = time.perf_counter()
        z, meta = embed_messages(
            q_batch,
            role_vocab=role_vocab,
            role_desc=role_desc,
            model_name=str(args.embed_model),
            text_encoder=str(args.embed_text_encoder),
            sbert_model_name=str(args.embed_sbert_model),
            max_length=int(args.embed_max_length),
            micro_batch=int(args.embed_micro_batch),
            d_role=int(args.embed_d_role),
            d_topo=int(args.embed_d_topo),
            proj_seed=int(args.embed_proj_seed),
            device=str(args.embed_device or "") or None,
        )
        t1 = time.perf_counter()

        dz = int(z.shape[1]) if z.ndim == 2 else 0
        dims = meta.get("dims") or {}
        h = int(dims.get("h") or 0)
        timing = meta.get("timing_sec") or {}
        t_load = float(timing.get("load_model") or 0.0)
        t_load_sbert = float(timing.get("load_sbert") or 0.0)
        t_roles = float(timing.get("encode_roles") or 0.0)
        t_topo = float(timing.get("embed_topo") or 0.0)
        t_msgs = float(timing.get("encode_messages") or 0.0)
        print(
            f"[EmbedTest] text_encoder={args.embed_text_encoder} hf_model={args.embed_model} sbert_model={args.embed_sbert_model} "
            f"device={z.device} max_length={args.embed_max_length} micro_batch={args.embed_micro_batch} "
            f"d_role={args.embed_d_role} d_topo={args.embed_d_topo} seed={args.embed_proj_seed}"
        )
        print(f"[EmbedTest] M={int(z.shape[0])} H={h} Dz={dz} dtype={z.dtype}")
        print(
            f"[EmbedTest] timing_sec={{load_model:{t_load:.3f} roles:{t_roles:.3f} "
            f"load_sbert:{t_load_sbert:.3f} topo:{t_topo:.3f} msgs:{t_msgs:.3f} total:{(t1-t0):.3f}}}"
        )

    if len(train_flip) > 0:
        f_loader = DataLoader(train_flip, batch_size=1, shuffle=True, collate_fn=collate_flip)
        f_batch = next(iter(f_loader))
        _print_flip_batch(f_batch)
    else:
        print("[FlipBatch] (empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
