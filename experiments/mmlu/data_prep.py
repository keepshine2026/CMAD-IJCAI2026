"""Data preparation utilities for the MMLU Adaptive-K pipeline.

This module contains dataset/log parsing and small sampling helpers.
It does not train a model.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from experiments.mmlu.dataset import load_nodes_index


def _strtobool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class DistillRecord:
    run_id: str
    question_id: str
    question: str
    correct_answer: str
    is_correct: bool
    agent_num: int
    transcript_path: str
    source: str  # "baseline" | "phase1"


def load_baseline_records(
    baseline_root: str,
    *,
    mode_filter: str = "FullConnected",
    agent_num_filter: int = 6,
) -> List[DistillRecord]:
    runs_csv = Path(str(baseline_root)) / "runs.csv"
    if not runs_csv.is_file():
        print(f"[LoadBaseline] runs.csv not found: {runs_csv}")
        return []

    records: List[DistillRecord] = []
    with open(runs_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = str(row.get("mode") or "")
            if mode_filter and mode != str(mode_filter):
                continue

            try:
                agent_num = int(row.get("agent_num") or 0)
            except Exception:
                agent_num = 0
            if agent_num_filter and agent_num != int(agent_num_filter):
                continue

            records.append(
                DistillRecord(
                    run_id=str(row.get("run_id") or ""),
                    question_id=str(row.get("question_id") or ""),
                    question=str(row.get("question") or ""),
                    correct_answer=str(row.get("correct_answer") or ""),
                    is_correct=_strtobool(row.get("is_correct")),
                    agent_num=int(agent_num),
                    transcript_path=str(row.get("transcript_path") or ""),
                    source="baseline",
                )
            )

    print(f"[LoadBaseline] Loaded {len(records)} records from {baseline_root}")
    return records


def load_phase1_records(
    phase1_root: str,
    *,
    mode_filter: str = "FullConnected",
    agent_num_filter: int = 6,
) -> List[DistillRecord]:
    runs_csv = Path(str(phase1_root)) / "runs.csv"
    if not runs_csv.is_file():
        print(f"[LoadPhase1] runs.csv not found: {runs_csv}")
        return []

    records: List[DistillRecord] = []
    with open(runs_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = str(row.get("mode") or "")
            if mode_filter and mode != str(mode_filter):
                continue

            try:
                agent_num = int(row.get("agent_num") or 0)
            except Exception:
                agent_num = 0
            if agent_num_filter and agent_num != int(agent_num_filter):
                continue

            records.append(
                DistillRecord(
                    run_id=str(row.get("run_id") or ""),
                    question_id=str(row.get("question_id") or ""),
                    question=str(row.get("question") or ""),
                    correct_answer=str(row.get("correct_answer") or ""),
                    is_correct=_strtobool(row.get("is_correct")),
                    agent_num=int(agent_num),
                    transcript_path=str(row.get("transcript_path") or ""),
                    source="phase1",
                )
            )

    print(f"[LoadPhase1] Loaded {len(records)} records from {phase1_root}")
    return records


def select_balanced_records(
    baseline_records: List[DistillRecord],
    phase1_records: List[DistillRecord],
    *,
    total: int = 40,
    seed: int = 42,
) -> List[DistillRecord]:
    random.seed(int(seed))
    total = max(0, int(total))
    half = total // 2

    baseline_sample = list(baseline_records or [])
    phase1_sample = list(phase1_records or [])
    random.shuffle(baseline_sample)
    random.shuffle(phase1_sample)

    selected = baseline_sample[:half] + phase1_sample[:half]
    random.shuffle(selected)

    n_baseline = sum(1 for r in selected if r.source == "baseline")
    n_phase1 = sum(1 for r in selected if r.source == "phase1")
    n_correct = sum(1 for r in selected if bool(r.is_correct))
    n_wrong = sum(1 for r in selected if not bool(r.is_correct))

    print(f"[SelectBalanced] Total={len(selected)}, baseline={n_baseline}, phase1={n_phase1}")
    print(f"[SelectBalanced] Correct={n_correct}, Wrong={n_wrong}")
    return selected


def infer_role_order_from_nodes_jsonl(
    *,
    data_root: str,
    agent_num: int,
    mode_filter: Optional[str] = None,
) -> List[str]:
    root = Path(str(data_root))
    for runs_csv in sorted(root.glob("*/runs.csv")):
        base_dir = runs_csv.parent
        nodes_path = base_dir / "nodes.jsonl"
        if not nodes_path.is_file():
            continue
        nodes_by_run = load_nodes_index(str(nodes_path))
        with open(runs_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if mode_filter and str(row.get("mode") or "") != str(mode_filter):
                    continue
                try:
                    n = int(row.get("agent_num") or 0)
                except Exception:
                    continue
                if int(n) != int(agent_num):
                    continue
                run_id = str(row.get("run_id") or "")
                if not run_id:
                    continue
                idx = nodes_by_run.get(run_id)
                if idx is None:
                    continue
                roles = [str(n.role) for n in idx.ordered]
                if len(roles) == int(agent_num):
                    return roles

    raise FileNotFoundError(
        f"Could not infer role order from {data_root}. Need nodes.jsonl + runs.csv with agent_num={agent_num}."
    )


__all__ = [
    "DistillRecord",
    "load_baseline_records",
    "load_phase1_records",
    "select_balanced_records",
    "infer_role_order_from_nodes_jsonl",
]
