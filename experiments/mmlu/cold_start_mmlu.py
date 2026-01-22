import argparse
import asyncio
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import find_dotenv, load_dotenv  # noqa: E402

_ = load_dotenv(find_dotenv())

from datasets.MMLU.download import download  # noqa: E402
from datasets.mmlu_dataset import MMLUDataset  # noqa: E402
from experiments.mmlu.exp_utils import (  # noqa: E402
    Accuracy,
    append_jsonl,
    build_topology_signature,
    dump_json,
    get_kwargs,
    make_run_id,
    prepare_edge_records,
    prepare_node_snapshots,
    save_graph_with_features,
)
from CMAD.graph.graph import Graph, TestGraph  # noqa: E402

# IMPORTANT:
# - Do NOT import experiment/prompt/mmlu_prompt_set (it used to override MMLU prompts).
# - Use CMAD's role order (same as CMAD/prompt/mmlu_prompt_set.py).
MAS_FRAMEWORK_ROLE_ORDER = [
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

MAX_PREVIEW_CHARS = 1000

RUN_FIELDNAMES = [
    "run_id",
    "dataset",
    "question_id",
    "question",
    "mode",
    "agent_num",
    "role_mode",
    "is_correct",
    "predicted_answer",
    "correct_answer",
    "topology_id",
    "graph_path",
    "transcript_path",
]


def _log_paths(output_root: str) -> Dict[str, str]:
    log_root = output_root
    return {
        "log_root": log_root,
        "runs_csv": os.path.join(log_root, "runs.csv"),
        "edges_jsonl": os.path.join(log_root, "edges.jsonl"),
        "nodes_jsonl": os.path.join(log_root, "nodes.jsonl"),
        "transcripts_dir": os.path.join(log_root, "transcripts"),
        "graphs_dir": os.path.join(log_root, "graphs"),
        "task_split_json": os.path.join(log_root, "task_split_mmlu.json"),
    }


def _ensure_dirs(paths: Dict[str, str]) -> None:
    os.makedirs(paths["log_root"], exist_ok=True)
    os.makedirs(paths["transcripts_dir"], exist_ok=True)
    os.makedirs(paths["graphs_dir"], exist_ok=True)


def _write_run_rows(runs_csv: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(runs_csv), exist_ok=True)
    file_exists = os.path.isfile(runs_csv)
    with open(runs_csv, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RUN_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _truncate_message(content: Any, limit: int = MAX_PREVIEW_CHARS) -> Tuple[str, bool]:
    text = "" if content is None else str(content)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _compute_topology_id(flow_graph) -> str:
    node_roles: List[Dict[str, Any]] = []
    for idx, feat in enumerate(getattr(flow_graph, "x", [])):
        role = feat.get("role", "Unknown") if isinstance(feat, dict) else getattr(feat, "role", "Unknown")
        node_roles.append({"id": str(idx), "role": role})
    if getattr(flow_graph, "edge_index", None) is None or flow_graph.edge_index.numel() == 0:
        edges: List[List[int]] = []
    else:
        edges = flow_graph.edge_index.t().cpu().tolist()
    return build_topology_signature(node_roles, edges)


def _build_edge_payloads(run_id: str, edge_logs: List[Dict[str, Any]], transcript_rel_path: str) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for idx, entry in enumerate(edge_logs or []):
        raw_content = entry.get("message", "")
        full_content = "" if raw_content is None else str(raw_content)
        preview, truncated = _truncate_message(full_content)
        content_hash = hashlib.sha256(full_content.encode("utf-8")).hexdigest() if full_content else None
        payloads.append(
            {
                "message_id": f"{run_id}_{idx}",
                "round": entry.get("round"),
                "edge_type": entry.get("edge_type"),
                "src_id": entry.get("src_id"),
                "src_role": entry.get("src_role"),
                "dst_id": entry.get("dst_id"),
                "dst_role": entry.get("dst_role"),
                "timestamp": entry.get("timestamp"),
                "content_preview": preview,
                "content_truncated": truncated,
                "content_hash": content_hash,
                "transcript_path": transcript_rel_path,
            }
        )
    return payloads


def _build_transcript_payload(
    run_id: str,
    topology_id: str,
    dataset_name: str,
    question_id: Any,
    question_text: str,
    mode: str,
    agent_num: int,
    edge_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = []
    for idx, entry in enumerate(edge_logs or []):
        raw_content = entry.get("message", "")
        full_content = "" if raw_content is None else str(raw_content)
        messages.append(
            {
                "sequence_id": idx,
                "message_id": f"{run_id}_{idx}",
                "round": entry.get("round"),
                "edge_type": entry.get("edge_type"),
                "src_id": entry.get("src_id"),
                "src_role": entry.get("src_role"),
                "dst_id": entry.get("dst_id"),
                "dst_role": entry.get("dst_role"),
                "timestamp": entry.get("timestamp"),
                "content": full_content,
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


def _unique_complex_configs() -> List[Tuple[str, int]]:
    configs = set()
    for agent_num in range(2, 7):
        if agent_num == 2:
            continue
        if agent_num == 3:
            configs.add(("FullConnected", 3))
        else:
            configs.add(("FullConnected", agent_num))
            configs.add(("Mesh", agent_num))
    return list(configs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-start MMLU data generation (phase1-style logging).")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"], help="Agent class names (single template)")
    parser.add_argument("--num_iterations", type=int, default=10, help="(Legacy) used when not --run_all_dev")
    parser.add_argument("--num_rounds", type=int, default=1, help="Dialogue rounds for each query")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-2024-08-06", help="LLM model name")
    parser.add_argument("--domain", type=str, default="mmlu", help="Domain / dataset name")
    parser.add_argument("--decision_method", type=str, default="FinalRefer", help="Decision node type")
    parser.add_argument("--role_mode", type=str, default="random", choices=["random", "cycle"], help="Role assignment mode")

    parser.add_argument("--output_root", type=str, default=os.path.join("logs", "mmlu", "phase1"))
    parser.add_argument("--run_all_dev", action="store_true", help="Run ALL dev questions (ignore num_iterations).")
    parser.add_argument("--fixed_fullconnected", action="store_true", help="Only run FullConnected configurations.")
    parser.add_argument("--fixed_agent_num", type=int, default=None, help="Only run configurations with this agent_num (e.g. 6).")

    args = parser.parse_args()
    if len(args.agent_names) != 1:
        parser.error("The number of agent names must be 1.")
    return args


async def evaluate(
    *,
    graph_configs: List[Dict[str, Any]],
    dataset: Any,
    num_rounds: int,
    limit_questions: Optional[int],
    eval_batch_size: int,
    args: argparse.Namespace,
    log_paths: Dict[str, str],
) -> float:
    accuracy = Accuracy()

    original_dataset = getattr(dataset, "dataset", dataset)

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records: List[Any] = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = len(dataset) if limit_questions is None else min(len(dataset), limit_questions)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        tasks: List[asyncio.Task] = []
        job_meta: List[Dict[str, Any]] = []

        for local_idx, record in enumerate(record_batch):
            record_id = record.get("id", f"task_{i_batch * eval_batch_size + local_idx}")
            input_dict = original_dataset.record_to_input(record)
            question_text = input_dict["task"]

            for cfg in graph_configs:
                g_copy = copy.deepcopy(cfg["graph"])
                flow_graph = g_copy.to_pyg_graph(input_dict)
                tg = TestGraph(
                    domain=args.domain,
                    llm_name=args.llm_name,
                    decision_method=args.decision_method,
                    pyg_data=flow_graph,
                )
                tasks.append(asyncio.create_task(tg.arun(input_dict, num_rounds)))
                job_meta.append(
                    {
                        "record": record,
                        "record_id": record_id,
                        "question_text": question_text,
                        "flow_graph": flow_graph,
                        "tg": tg,
                        "mode": cfg["mode"],
                        "agent_num": cfg["agent_num"],
                    }
                )

        raw_results = await asyncio.gather(*tasks)
        run_rows: List[Dict[str, Any]] = []
        edge_records: List[Dict[str, Any]] = []
        node_records: List[Dict[str, Any]] = []

        for raw_answer, meta in zip(raw_results, job_meta):
            record = meta["record"]
            record_id = meta["record_id"]
            question_text = meta["question_text"]
            flow_graph = meta["flow_graph"]
            tg_inst = meta["tg"]
            current_mode = meta["mode"]
            current_agent_num = meta["agent_num"]

            answer = original_dataset.postprocess_answer(raw_answer)
            correct_answer = original_dataset.record_to_target_answer(record)
            is_correct = accuracy.update(answer, correct_answer)

            topology_id = _compute_topology_id(flow_graph)
            run_id = make_run_id(
                "mmlu",
                record_id,
                current_mode,
                current_agent_num,
                llm_name=args.llm_name,
                topology_id=topology_id,
            )

            transcript_rel_path = os.path.join("transcripts", f"{run_id}.json")
            transcript_abs_path = os.path.join(log_paths["transcripts_dir"], f"{run_id}.json")

            run_meta = tg_inst.get_run_metadata() if hasattr(tg_inst, "get_run_metadata") else {}
            edge_logs = run_meta.get("edge_logs") or []
            node_logs = run_meta.get("node_topology") or []

            transcript_payload = _build_transcript_payload(
                run_id=run_id,
                topology_id=topology_id,
                dataset_name="mmlu",
                question_id=record_id,
                question_text=question_text,
                mode=current_mode,
                agent_num=current_agent_num,
                edge_logs=edge_logs,
            )
            dump_json(transcript_abs_path, transcript_payload)

            edge_payloads = _build_edge_payloads(run_id, edge_logs, transcript_rel_path)
            edge_records.extend(prepare_edge_records(run_id, topology_id, edge_payloads))
            node_records.extend(prepare_node_snapshots(run_id, topology_id, node_logs))

            name = "_".join(map(str, ["mmlu", record_id, current_mode, current_agent_num, is_correct]))
            graph_rel_path = os.path.join(log_paths["log_root"], "graphs", f"{name}.pt")
            graph_abs_path = os.path.join(log_paths["graphs_dir"], f"{name}.pt")
            save_graph_with_features(
                flow_graph,
                graph_abs_path,
                {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_correct,
                    "question": question_text,
                },
            )

            run_rows.append(
                {
                    "run_id": run_id,
                    "dataset": "mmlu",
                    "question_id": record_id,
                    "question": question_text,
                    "mode": current_mode,
                    "agent_num": current_agent_num,
                    "role_mode": args.role_mode,
                    "is_correct": bool(is_correct),
                    "predicted_answer": answer,
                    "correct_answer": correct_answer,
                    "topology_id": topology_id,
                    "graph_path": graph_rel_path,
                    "transcript_path": transcript_rel_path,
                }
            )

        append_jsonl(log_paths["edges_jsonl"], edge_records)
        append_jsonl(log_paths["nodes_jsonl"], node_records)
        _write_run_rows(log_paths["runs_csv"], run_rows)

        accuracy.print()

    accuracy.print()
    return accuracy.get()


async def main() -> None:
    args = parse_args()

    log_paths = _log_paths(args.output_root)
    _ensure_dirs(log_paths)

    download()
    dataset = MMLUDataset("dev")
    all_indices = list(range(len(dataset)))

    if args.run_all_dev:
        base_task_indices = all_indices
        finetune_task_indices: List[int] = []
    else:
        train_set_size = int(args.num_iterations) * int(args.batch_size)
        train_indices = all_indices[:train_set_size]
        finetune_candidates = list(train_indices)
        random.shuffle(finetune_candidates)
        base_task_indices = finetune_candidates
        finetune_task_indices = []

    with open(log_paths["task_split_json"], "w", encoding="utf-8") as f:
        json.dump({"base_tasks": base_task_indices, "finetune_tasks": finetune_task_indices}, f, ensure_ascii=False, indent=2)
    print(f"[ColdStart] Selected {len(base_task_indices)} tasks (dev={len(dataset)}). Split saved to {log_paths['task_split_json']}")

    cold_start_dataset = torch.utils.data.Subset(dataset, base_task_indices)

    configs = _unique_complex_configs()
    if args.fixed_fullconnected:
        configs = [(mode, n) for (mode, n) in configs if mode == "FullConnected"]
    if args.fixed_agent_num is not None:
        configs = [(mode, n) for (mode, n) in configs if int(n) == int(args.fixed_agent_num)]

    print(f"[ColdStart] Generating for {len(configs)} configurations; output_root={os.path.abspath(log_paths['log_root'])}")

    graph_configs: List[Dict[str, Any]] = []
    for mode, agent_num in configs:
        current_agent_names = [args.agent_names[0]] * int(agent_num)
        kwargs = get_kwargs(mode, int(agent_num))

        if args.role_mode == "cycle":
            role_cycle = itertools.cycle(MAS_FRAMEWORK_ROLE_ORDER)
            roles = [next(role_cycle) for _ in range(int(agent_num))]
        else:
            roles = random.choices(MAS_FRAMEWORK_ROLE_ORDER, k=int(agent_num))
        kwargs["node_kwargs"] = [{"role": r} for r in roles]

        base_graph = Graph(
            domain=args.domain,
            llm_name=args.llm_name,
            agent_names=current_agent_names,
            decision_method=args.decision_method,
            **kwargs,
        )
        graph_configs.append({"mode": mode, "agent_num": int(agent_num), "graph": base_graph})

    await evaluate(
        graph_configs=graph_configs,
        dataset=cold_start_dataset,
        num_rounds=int(args.num_rounds),
        limit_questions=None,
        eval_batch_size=int(args.batch_size),
        args=args,
        log_paths=log_paths,
    )
    print("[ColdStart] All cold-start data generation complete.")


if __name__ == "__main__":
    asyncio.run(main())
