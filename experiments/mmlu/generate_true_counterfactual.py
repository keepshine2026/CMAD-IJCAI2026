import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import csv
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no progress bar if tqdm is unavailable.
    def tqdm(iterable, **kwargs):
        return iterable

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.mmlu_dataset import MMLUDataset
from experiments.mmlu.exp_utils import (
    Accuracy,
    append_jsonl,
    build_topology_signature,
    dump_json,
    make_run_id,
    prepare_edge_records,
    prepare_node_snapshots,
)

# Note: user requested the folder name "true_conterfactual" (typo kept for compatibility).
LOG_ROOT = os.path.join("logs", "mmlu", "true_conterfactual")
RUN_TABLE_PATH = os.path.join(LOG_ROOT, "runs.csv")
EDGE_LOG_PATH = os.path.join(LOG_ROOT, "edges.jsonl")
NODE_LOG_PATH = os.path.join(LOG_ROOT, "nodes.jsonl")
TRANSCRIPT_DIR = os.path.join(LOG_ROOT, "transcripts")
MAX_PREVIEW_CHARS = 1000

RUN_FIELDNAMES = [
    "run_id",
    "parent_run_id",
    "parent_is_correct",
    "parent_predicted_answer",
    "parent_correct_answer",
    "dataset",
    "question_id",
    "question",
    "mode",
    "agent_num",
    "is_correct",
    "predicted_answer",
    "correct_answer",
    "topology_id",
    "graph_path",
    "transcript_path",
    "removed_edges",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate true-counterfactual runs by muting spatial edges, "
            "sampling baseline runs that were answered incorrectly."
        )
    )
    parser.add_argument(
        "--runs_csv",
        type=str,
        default=os.path.join("logs", "mmlu", "baseline", "runs.csv"),
        help="Path to baseline runs.csv (e.g., logs/mmlu/baseline/runs.csv).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=LOG_ROOT,
        help="Directory to store true-counterfactual logs.",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=0,
        help="How many baseline-incorrect runs to sample. 0 means all.",
    )
    parser.add_argument(
        "--sample_questions",
        type=int,
        default=None,
        help="Alias of --num_questions (e.g., 40 means randomly sample 40 questions).",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of inference rounds for each run.",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-4o-2024-08-06",
        help="LLM name for rebuilding TestGraph nodes.",
    )
    parser.add_argument(
        "--edges_to_mute",
        type=int,
        required=True,
        help="How many spatial edges to mute each try (k).",
    )
    parser.add_argument(
        "--num_tries",
        type=int,
        required=True,
        help="How many different random edge-removal tries per baseline run.",
    )
    args = parser.parse_args()
    if args.sample_questions is not None:
        args.num_questions = int(args.sample_questions)
    return args


def ensure_log_dirs(root: str):
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "transcripts"), exist_ok=True)


def load_baseline_rows(runs_csv: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(runs_csv):
        raise FileNotFoundError(f"Cannot find baseline runs at {runs_csv}")
    with open(runs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def dedupe_by_question(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    first_seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id")
        if qid and qid not in first_seen:
            first_seen[qid] = row
    return list(first_seen.values())


def _strtobool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def truncate_message(content: Any, limit: int = MAX_PREVIEW_CHARS) -> Tuple[str, bool]:
    text = "" if content is None else str(content)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def compute_topology_id(flow_graph) -> str:
    node_roles = []
    for idx, feat in enumerate(flow_graph.x):
        role = feat.get("role", "Unknown") if isinstance(feat, dict) else getattr(feat, "role", "Unknown")
        node_roles.append({"id": str(idx), "role": role})
    if getattr(flow_graph, "edge_index", None) is None or flow_graph.edge_index.numel() == 0:
        edges = []
    else:
        edges = flow_graph.edge_index.t().cpu().tolist()
    return build_topology_signature(node_roles, edges)


def build_edge_payloads(run_id: str, edge_logs: List[Dict[str, Any]], transcript_rel_path: str) -> List[Dict[str, Any]]:
    payloads = []
    for entry in edge_logs or []:
        raw_content = entry.get("message", "")
        full_content = "" if raw_content is None else str(raw_content)
        preview, truncated = truncate_message(full_content)
        content_hash = None if not full_content else hashlib.sha256(full_content.encode("utf-8")).hexdigest()
        payloads.append(
            {
                "edge_type": entry.get("edge_type"),
                "src_id": entry.get("src_id"),
                "src_role": entry.get("src_role"),
                "dst_id": entry.get("dst_id"),
                "dst_role": entry.get("dst_role"),
                "content": preview,
                "content_truncated": truncated,
                "content_sha256": content_hash,
                "timestamp": entry.get("timestamp"),
                "transcript_path": transcript_rel_path,
            }
        )
    return payloads


def build_transcript_payload(
    *,
    run_id: str,
    topology_id: str,
    dataset_name: str,
    question_id: str,
    question_text: str,
    mode: str,
    agent_num: int,
    edge_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = []
    for idx, entry in enumerate(edge_logs or []):
        raw_content = entry.get("message", "")
        full_content = "" if raw_content is None else str(raw_content)
        preview, truncated = truncate_message(full_content)
        content_hash = None if not full_content else hashlib.sha256(full_content.encode("utf-8")).hexdigest()
        messages.append(
            {
                "sequence_id": idx,
                "message_id": f"{run_id}_{idx}",
                "round": entry.get("round", 0),
                "edge_type": entry.get("edge_type"),
                "src_id": entry.get("src_id"),
                "src_role": entry.get("src_role"),
                "dst_id": entry.get("dst_id"),
                "dst_role": entry.get("dst_role"),
                "timestamp": entry.get("timestamp"),
                "content": preview,
                "content_truncated": truncated,
                "content_sha256": content_hash,
            }
        )
    return {
        "run_id": run_id,
        "topology_id": topology_id,
        "dataset": dataset_name,
        "question_id": question_id,
        "question": question_text,
        "mode": mode,
        "agent_num": agent_num,
        "messages": messages,
    }


def parse_question_index(question_id: str) -> int:
    if not question_id.startswith("task_"):
        raise ValueError(f"Unexpected question_id format: {question_id}")
    return int(question_id.split("_", 1)[1])


def list_spatial_edges(flow_graph) -> List[Dict[str, Any]]:
    roles = [
        feat.get("role", "Unknown") if isinstance(feat, dict) else getattr(feat, "role", "Unknown")
        for feat in flow_graph.x
    ]
    edge_indices = flow_graph.edge_index.t().tolist() if flow_graph.edge_index.numel() else []
    result = []
    for idx, (src_idx, dst_idx) in enumerate(edge_indices):
        result.append(
            {
                "edge_idx": idx,
                "src_idx": src_idx,
                "dst_idx": dst_idx,
                "src_role": roles[src_idx],
                "dst_role": roles[dst_idx],
            }
        )
    return result


def choose_edges_to_mute(
    edges: List[Dict[str, Any]],
    edges_to_mute: int,
    seen: set,
    max_retry: int = 50,
) -> List[Dict[str, Any]]:
    """Sample a set of edges to mute, trying to avoid duplicates within one question/run."""
    if not edges:
        return []
    edge_count = min(max(1, int(edges_to_mute)), len(edges))

    attempt = 0
    candidate = None
    while attempt < max_retry:
        candidate = random.sample(edges, k=edge_count)
        key = tuple(sorted(edge["edge_idx"] for edge in candidate))
        if key not in seen:
            seen.add(key)
            return candidate
        attempt += 1

    if candidate is None:
        candidate = random.sample(edges, k=edge_count)
    key = tuple(sorted(edge["edge_idx"] for edge in candidate))
    seen.add(key)
    return candidate


def normalize_graph_path(runs_csv: str, relative_path: str) -> str:
    candidates = []

    if os.path.isabs(relative_path):
        candidates.append(os.path.normpath(relative_path))

    base_dir = os.path.dirname(runs_csv)
    candidates.append(os.path.normpath(os.path.join(base_dir, relative_path)))

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.normpath(os.path.join(project_root, relative_path)))

    fallbacks = [
        os.path.join("logs", "mmlu", "baseline", relative_path),
        os.path.join(project_root, "logs", "mmlu", "baseline", relative_path),
    ]
    candidates.extend(os.path.normpath(path) for path in fallbacks)

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(f"Graph file not found. Tried: {candidates}")


def run_true_counterfactual(
    baseline_row: Dict[str, Any],
    flow_graph,
    dataset: MMLUDataset,
    args,
    accuracy_tracker: Accuracy,
) -> List[Dict[str, Any]]:
    from CMAD.graph.graph import TestGraph

    question_id = baseline_row["question_id"]
    question_idx = parse_question_index(question_id)
    record = dataset[question_idx]
    input_dict = dataset.record_to_input(record)
    correct_answer = dataset.record_to_target_answer(record)

    topology_id = compute_topology_id(flow_graph)
    edges = list_spatial_edges(flow_graph)
    if not edges:
        print(f"[Skip] No spatial edges for question {question_id}, run {baseline_row.get('run_id')}")
        return []

    parent_pred = baseline_row.get("predicted_answer")
    parent_correct = baseline_row.get("correct_answer")
    parent_is_correct = baseline_row.get("is_correct")

    cf_rows: List[Dict[str, Any]] = []
    seen_edge_sets: set = set()
    for try_id in range(args.num_tries):
        selected_edges = choose_edges_to_mute(edges, args.edges_to_mute, seen_edge_sets)
        if not selected_edges:
            continue

        tg = TestGraph(
            domain="mmlu",
            llm_name=args.llm_name,
            decision_method="FinalRefer",
            pyg_data=flow_graph,
        )
        index_pairs = [(edge["src_idx"], edge["dst_idx"]) for edge in selected_edges]
        tg.set_blocked_spatial_edges_by_indices(index_pairs)

        answers = asyncio.run(tg.arun(input_dict, num_rounds=args.num_rounds))
        predicted = dataset.postprocess_answer(answers)
        is_correct = accuracy_tracker.update(predicted, correct_answer)

        run_id = make_run_id(
            dataset="mmlu",
            question_id=question_id,
            mode=baseline_row["mode"],
            agent_num=int(baseline_row["agent_num"]),
            llm_name=args.llm_name,
            topology_id=topology_id,
            extra=f"tcf{try_id}",
        )

        transcript_rel_path = os.path.join("transcripts", f"{run_id}.json")
        transcript_abs_path = os.path.join(args.output_root, transcript_rel_path)

        run_meta = tg.get_run_metadata()
        edge_logs = run_meta.get("edge_logs") or []
        node_logs = run_meta.get("node_topology") or []

        transcript_payload = build_transcript_payload(
            run_id=run_id,
            topology_id=topology_id,
            dataset_name="mmlu",
            question_id=question_id,
            question_text=input_dict["task"],
            mode=baseline_row["mode"],
            agent_num=int(baseline_row["agent_num"]),
            edge_logs=edge_logs,
        )
        dump_json(transcript_abs_path, transcript_payload)

        edge_payloads = build_edge_payloads(run_id, edge_logs, transcript_rel_path)
        append_jsonl(EDGE_LOG_PATH, prepare_edge_records(run_id, topology_id, edge_payloads))
        append_jsonl(NODE_LOG_PATH, prepare_node_snapshots(run_id, topology_id, node_logs))

        removed_edges_payload = json.dumps(selected_edges, ensure_ascii=False)

        cf_rows.append(
            {
                "run_id": run_id,
                "parent_run_id": baseline_row["run_id"],
                "parent_is_correct": parent_is_correct,
                "parent_predicted_answer": parent_pred,
                "parent_correct_answer": parent_correct,
                "dataset": "mmlu",
                "question_id": question_id,
                "question": input_dict["task"],
                "mode": baseline_row["mode"],
                "agent_num": baseline_row["agent_num"],
                "is_correct": is_correct,
                "predicted_answer": predicted,
                "correct_answer": correct_answer,
                "topology_id": topology_id,
                "graph_path": baseline_row["graph_path"],
                "transcript_path": transcript_rel_path,
                "removed_edges": removed_edges_payload,
            }
        )

    return cf_rows


def write_run_rows(rows: List[Dict[str, Any]]):
    if not rows:
        return
    ensure_log_dirs(LOG_ROOT)
    file_exists = os.path.isfile(RUN_TABLE_PATH)
    with open(RUN_TABLE_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RUN_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_log_dirs(args.output_root)

    global LOG_ROOT, RUN_TABLE_PATH, EDGE_LOG_PATH, NODE_LOG_PATH, TRANSCRIPT_DIR
    LOG_ROOT = args.output_root
    RUN_TABLE_PATH = os.path.join(LOG_ROOT, "runs.csv")
    EDGE_LOG_PATH = os.path.join(LOG_ROOT, "edges.jsonl")
    NODE_LOG_PATH = os.path.join(LOG_ROOT, "nodes.jsonl")
    TRANSCRIPT_DIR = os.path.join(LOG_ROOT, "transcripts")

    baseline_rows = load_baseline_rows(args.runs_csv)
    baseline_rows = [row for row in baseline_rows if not _strtobool(row.get("is_correct"))]
    baseline_rows = dedupe_by_question(baseline_rows)
    if not baseline_rows:
        print("No baseline-incorrect rows available.")
        return

    random.shuffle(baseline_rows)
    if args.num_questions and args.num_questions > 0:
        total_rows = len(baseline_rows)
        if args.num_questions <= total_rows:
            selected = baseline_rows[: args.num_questions]
        else:
            selected = []
            while len(selected) < args.num_questions:
                random.shuffle(baseline_rows)
                selected.extend(baseline_rows)
            selected = selected[: args.num_questions]
    else:
        selected = baseline_rows

    dataset = MMLUDataset("dev")
    accuracy_tracker = Accuracy()
    all_rows: List[Dict[str, Any]] = []

    for row in tqdm(selected, desc="TrueCounterfactual", unit="run", total=len(selected)):
        try:
            graph_abs_path = normalize_graph_path(args.runs_csv, row["graph_path"])
        except FileNotFoundError as e:
            print(f"[Skip] graph file missing for run {row.get('run_id')}: {e}")
            continue
        try:
            flow_graph = torch.load(graph_abs_path, weights_only=False)
        except Exception as e:
            print(f"[Skip] failed to load graph {graph_abs_path} for run {row.get('run_id')}: {e}")
            continue

        rows_generated = run_true_counterfactual(row, flow_graph, dataset, args, accuracy_tracker)
        if rows_generated:
            write_run_rows(rows_generated)
            all_rows.extend(rows_generated)

    if all_rows:
        print(f"Generated {len(all_rows)} true-counterfactual runs from {len(selected)} baseline-incorrect runs.")
    else:
        print("No true-counterfactual runs were generated.")


if __name__ == "__main__":
    main()
