"""Train and evaluate Adaptive-K edge selection on MMLU."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch

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


def _pick_mmlu_logs_dir(preferred_name: str, *, prefix: str) -> str:
    preferred = os.path.join(PROJECT_ROOT, "logs", "mmlu", str(preferred_name))
    if os.path.isdir(preferred):
        return preferred
    try:
        mmlu_logs = os.path.join(PROJECT_ROOT, "logs", "mmlu")
        candidates: List[str] = []
        for entry in os.listdir(mmlu_logs):
            if not str(entry).startswith(str(prefix)):
                continue
            p = os.path.join(mmlu_logs, str(entry))
            if os.path.isdir(p):
                candidates.append(p)
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    except Exception:
        pass
    return preferred

def _candidate_edge_indices_upper_triangle(spatial_masks: torch.Tensor, num_nodes: int) -> List[int]:
    """Return candidate edges from the upper triangle only (i < j)."""
    n = num_nodes
    mask = spatial_masks.detach().cpu().view(-1)
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


def _write_run_rows_with_fields(run_table_path: str, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Append run rows to a CSV file (with explicit fieldnames)."""
    import csv
    if not rows:
        return
    file_exists = os.path.isfile(run_table_path)
    os.makedirs(os.path.dirname(run_table_path), exist_ok=True)
    with open(run_table_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def score_to_k(
    score: float,
    percentiles: Dict[str, float],
    k_min: int = 3,
    k_max: int = 10,
    target_k: int = 6,
) -> int:
    """Map a difficulty score to k."""
    percentile_keys = ["p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90"]
    thresholds = [percentiles.get(key, 0.0) for key in percentile_keys]

    k_range = k_max - k_min

    segment = 0
    for i, thresh in enumerate(thresholds):
        if score >= thresh:
            segment = i + 1
        else:
            break

    ratio = segment / 9.0  # 0.0..1.0
    k = k_max - ratio * k_range
    
    return int(max(k_min, min(k_max, round(k))))


def score_to_k_linear(
    score: float,
    stats: Dict[str, float],
    k_min: int = 3,
    k_max: int = 10,
) -> int:
    """Linear mapping: higher score -> smaller k."""
    s_min = stats.get("min", 0)
    s_max = stats.get("max", 1)
    
    if s_max <= s_min:
        return (k_min + k_max) // 2
    
    ratio = (score - s_min) / (s_max - s_min)
    ratio = max(0.0, min(1.0, ratio))
    
    k = k_max - ratio * (k_max - k_min)
    
    return int(max(k_min, min(k_max, round(k))))


def run_eval_adaptive_k(
    ckpt_path: str,
    *,
    split: str = "val",
    limit: int = 153,
    batch_size: int = 4,
    tau: float = 0.1,
    spatial_select_mode: str = "topk",
    dagsample_tau: Optional[float] = None,
    device: str = "cuda",
    log_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate with dynamic k."""
    from CMAD.graph.graph_final import GraphFinal, EdgeMLP, EdgeMLPConfig, GCNEmbed
    from CMAD.utils.globals import Cost, PromptTokens, CompletionTokens
    from datasets.mmlu_dataset import MMLUDataset
    from experiments.mmlu.exp_utils import (
        get_kwargs,
        append_jsonl,
        build_topology_signature,
        dump_json,
        make_run_id,
        prepare_node_snapshots,
    )
    from experiments.mmlu.data_prep import infer_role_order_from_nodes_jsonl
    
    print(f"[EvalAdaptiveK] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Load config
    difficulty_mode = ckpt.get("difficulty_mode", "score_mean")
    k_mapping_mode = ckpt.get("k_mapping_mode", "percentile")
    k_min = int(ckpt.get("k_min", 3))
    k_max = int(ckpt.get("k_max", 10))
    target_k = int(ckpt.get("target_k", 6))
    percentiles = ckpt.get("percentiles", {})
    stats = ckpt.get("stats", {})
    
    model_score_mode = str(ckpt.get("model_score_mode", "edge_mlp"))
    sim_mode = str(ckpt.get("sim_mode", "dot"))
    logit_temp = float(ckpt.get("logit_temp", 1.0))
    logit_clip = ckpt.get("logit_clip", None)
    spatial_logit_bias = ckpt.get("spatial_logit_bias", None)
    use_gcn_embed = ckpt.get("use_gcn_embed", True)
    
    print(f"[EvalAdaptiveK] Config: k_min={k_min}, k_max={k_max}, target_k={target_k}")
    print(f"[EvalAdaptiveK] difficulty_mode={difficulty_mode}, k_mapping_mode={k_mapping_mode}")
    print(f"[EvalAdaptiveK] Percentiles: {percentiles}")
    
    # Build evaluation graph.
    agent_names = ["AnalyzeAgent"]
    agent_nums = [6]
    n_agents = 6
    kwargs = get_kwargs("FullConnected", n_agents)
    train_root_for_roles = _pick_mmlu_logs_dir("traindataset", prefix="traindataset")
    role_order = infer_role_order_from_nodes_jsonl(
        data_root=train_root_for_roles,
        agent_num=n_agents,
        mode_filter="FullConnected",
    )
    kwargs["node_kwargs"] = [{"role": r} for r in role_order]

    graph = GraphFinal(
        domain="mmlu",
        llm_name="gpt-4o-2024-08-06",
        agent_names=(agent_names * n_agents if len(agent_names) == 1 else agent_names),
        decision_method="FinalRefer",
        optimized_spatial=True,
        optimized_temporal=False,
        **kwargs,
    )
    if hasattr(graph, "clear_blocked_spatial_edges"):
        graph.clear_blocked_spatial_edges()
    
    dev = torch.device(device)
    n = graph.num_nodes
    
    # Optionally replace the GCN backbone (for checkpoint compatibility).
    if use_gcn_embed:
        gcn_config = ckpt.get("gcn_config", {})
        in_ch = gcn_config.get("in_channels", graph.gcn.conv1.in_channels)
        hid_ch = gcn_config.get("hidden_channels", graph.gcn.conv1.out_channels)
        out_ch = gcn_config.get("out_channels", graph.gcn.conv2.out_channels)
        graph.gcn = GCNEmbed(in_ch, hid_ch, out_ch, use_residual=True)
    
    # Load weights
    graph.gcn.load_state_dict(ckpt["gcn_state_dict"])
    graph.mlp.load_state_dict(ckpt["mlp_state_dict"])
    
    graph.gcn.to(dev)
    graph.mlp.to(dev)
    graph.gcn.eval()
    graph.mlp.eval()
    graph.features = graph.features.to(dev)
    graph.role_adj_matrix = graph.role_adj_matrix.to(dev)

    # Configure inference edge selection (topk or dagsample).
    graph.edge_score_source = "model"
    graph.model_score_mode = "edge_mlp" if model_score_mode == "edge_mlp" else "sim"
    graph.sim_mode = str(sim_mode)
    graph.logit_temp = float(logit_temp)
    if logit_clip is None:
        graph.logit_clip = None
    else:
        try:
            graph.logit_clip = None if float(logit_clip) <= 0 else float(logit_clip)
        except Exception:
            graph.logit_clip = None
    graph.edge_score_mode = "logit"
    graph.candidate_mode = "upper_triangle"
    graph.spatial_select_mode = str(spatial_select_mode)
    # Backward compatible: if dagsample_tau is not set, reuse tau.
    graph.dagsample_tau = float(tau if dagsample_tau is None else dagsample_tau)

    # Optional scalar bias on edge logits (backward compatible).
    try:
        bias_f = float(spatial_logit_bias) if spatial_logit_bias is not None else 0.0
    except Exception:
        bias_f = 0.0
    graph.spatial_logit_bias = torch.tensor(bias_f, dtype=torch.float32, device=dev)
    
    # EdgeMLP
    edge_mlp: Optional[EdgeMLP] = None
    if "edge_mlp_state_dict" in ckpt:
        cfg = ckpt.get("edge_mlp_config", {})
        edge_mlp = EdgeMLP(EdgeMLPConfig(
            node_dim=cfg.get("node_dim", 32),
            query_dim=cfg.get("query_dim", 384),
            hidden_dim=cfg.get("hidden_dim", 128),
            use_prod=cfg.get("use_prod", True),
        )).to(dev)
        edge_mlp.load_state_dict(ckpt["edge_mlp_state_dict"])
        edge_mlp.eval()
    
    # Candidate edges (upper triangle only).
    try:
        candidate_idx = list(getattr(graph, "_candidate_edge_flats")())  # type: ignore[misc]
    except Exception:
        candidate_idx = _candidate_edge_indices_upper_triangle(graph.spatial_masks, n)
    
    # Load dataset
    dataset = MMLUDataset(split=split)
    limit = min(len(dataset), limit) if limit else len(dataset)
    
    # Collect questions and answers
    questions: List[str] = []
    answers: List[str] = []
    records: List[Any] = []
    for i in range(limit):
        record = dataset[i]
        input_dict = dataset.record_to_input(record)
        questions.append(input_dict["task"])
        answers.append(dataset.record_to_target_answer(record))
        records.append(record)
    
    print(f"[EvalAdaptiveK] Loaded {len(questions)} questions")
    
    # ===== Logging directory =====
    log_paths: Optional[Dict[str, str]] = None
    if log_root:
        transcripts_dir = os.path.join(log_root, "transcripts")
        graphs_dir = os.path.join(log_root, "graphs")
        os.makedirs(transcripts_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        log_paths = {
            "log_root": log_root,
            "transcripts_dir": transcripts_dir,
            "graphs_dir": graphs_dir,
            "runs_csv": os.path.join(log_root, "runs.csv"),
            "edges_jsonl": os.path.join(log_root, "edges.jsonl"),
            "nodes_jsonl": os.path.join(log_root, "nodes.jsonl"),
        }
        csv_fieldnames = [
            "run_id", "dataset", "question_id", "question", "mode", "agent_num",
            "is_correct", "predicted_answer", "correct_answer", "topology_id",
            "selected_edges", "k", "tau", "transcript_path",
        ]
        print(f"[EvalAdaptiveK] Logging to: {log_root}")
    
    correct = 0
    total = 0
    k_values: List[int] = []
    
    # Reset token counters
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    
    async def eval_batch(batch_indices: List[int]) -> List[Tuple[bool, int]]:
        """Return a list of (is_correct, k)."""
        nonlocal k_values
        tasks = []
        batch_info = []

        cand_t = torch.tensor(candidate_idx, dtype=torch.long, device=dev)
        cand_cnt = int(len(candidate_idx))
        
        for idx in batch_indices:
            q = questions[idx]
            ans = answers[idx]

            # Deep-copy the graph (isolate per-question state) but share model weights.
            g = copy.deepcopy(graph)
            g.gcn = graph.gcn
            g.mlp = graph.mlp
            if edge_mlp is not None:
                g.edge_mlp = edge_mlp
            if hasattr(graph, "spatial_logit_bias"):
                g.spatial_logit_bias = graph.spatial_logit_bias

            # Compute this question's difficulty score -> k
            with torch.no_grad():
                g.construct_new_features(q)
                logits_all = g._compute_edge_scores_from_model() if hasattr(g, "_compute_edge_scores_from_model") else None
                if not isinstance(logits_all, torch.Tensor) or logits_all.numel() == 0:
                    raise RuntimeError("[EvalAdaptiveK] Failed to compute model edge scores.")
                cand_logits = logits_all[cand_t]

                if difficulty_mode == "score_sum":
                    score = float(cand_logits.sum().item())
                elif difficulty_mode == "top_score_sum":
                    topk_vals, _ = torch.topk(cand_logits, k=min(target_k, int(cand_logits.numel())))
                    score = float(topk_vals.sum().item())
                else:
                    score = float(cand_logits.mean().item())

                if k_mapping_mode == "percentile":
                    k = score_to_k(score, percentiles, k_min, k_max, target_k)
                else:
                    k = score_to_k_linear(score, stats, k_min, k_max)

            k = int(max(1, min(cand_cnt, int(k))))
            k_values.append(k)

            # Inject k via keep_ratio_fixed so the graph builds the topology inside arun().
            g.k_source = "fixed"
            g.keep_ratio_fixed = float(max(0.0, min(1.0, (float(k) - 1e-6) / float(max(1, cand_cnt)))))
            g.spatial_select_mode = str(getattr(graph, "spatial_select_mode", "topk"))
            g.dagsample_tau = float(getattr(graph, "dagsample_tau", 1.0))

            input_dict = {"task": q}
            tasks.append(asyncio.create_task(g.arun(input_dict, 1)))
            batch_info.append({
                "idx": idx,
                "question_id": str(idx),
                "question": q,
                "answer": ans,
                "k": k,
                "realized_graph": g,
            })
        
        # Await all tasks
        raw_results = await asyncio.gather(*tasks)
        
        # Logging
        run_rows: List[Dict[str, Any]] = []
        edge_records: List[Dict[str, Any]] = []
        node_records: List[Dict[str, Any]] = []
        
        results_list = []
        for raw_result, info in zip(raw_results, batch_info):
            # arun returns (answer, log_prob, graph) for some implementations
            if isinstance(raw_result, tuple):
                raw_answer = raw_result[0]
            else:
                raw_answer = raw_result
            
            # Postprocess answer
            pred = dataset.postprocess_answer(raw_answer)
            is_correct = (pred == info["answer"])
            results_list.append((is_correct, info["k"]))
            
            # ===== Save transcript =====
            if log_paths is not None:
                try:
                    realized_g = info["realized_graph"]
                    task_text = info["question"]
                    question_id = info["question_id"]
                    k_used = info["k"]
                    
                    # Build topology signature
                    agent_nodes = [node for node in realized_g.nodes.values() if node != getattr(realized_g, "decision_node", None)]
                    node_id_to_idx = {node.id: idx for idx, node in enumerate(agent_nodes)}
                    node_roles = [
                        {"id": str(i), "role": getattr(node, "role", "Unknown")}
                        for i, node in enumerate(agent_nodes)
                    ]
                    edge_indices_for_topo: List[Tuple[int, int]] = []
                    for src_node in agent_nodes:
                        src_i = node_id_to_idx[src_node.id]
                        for dst_node in getattr(src_node, "spatial_successors", []) or []:
                            if dst_node.id in node_id_to_idx:
                                edge_indices_for_topo.append((src_i, node_id_to_idx[dst_node.id]))
                    
                    topology_id = build_topology_signature(node_roles, edge_indices_for_topo)
                    
                    run_id = make_run_id(
                        "mmlu",
                        question_id,
                        f"adaptivek{k_used}",
                        len(realized_g.nodes),
                        llm_name="gpt-4o-2024-08-06",
                        topology_id=topology_id,
                        extra=f"i{info['idx']}",
                    )
                    
                    transcript_rel_path = os.path.join("transcripts", f"{run_id}.json")
                    transcript_abs_path = os.path.join(log_paths["transcripts_dir"], f"{run_id}.json")
                    
                    # Get run metadata
                    run_meta = realized_g.get_run_metadata() if hasattr(realized_g, "get_run_metadata") else {}
                    edge_logs = run_meta.get("edge_logs") or []
                    node_logs = run_meta.get("node_topology") or []
                    
                    # Build transcript payload
                    messages: List[Dict[str, Any]] = []
                    for msg_idx, entry in enumerate(edge_logs or []):
                        raw_content = entry.get("message", "")
                        full_content = "" if raw_content is None else str(raw_content)
                        messages.append({
                            "sequence_id": msg_idx,
                            "message_id": f"{run_id}_{msg_idx}",
                            "round": entry.get("round"),
                            "edge_type": entry.get("edge_type"),
                            "src_id": entry.get("src_id"),
                            "src_role": entry.get("src_role"),
                            "dst_id": entry.get("dst_id"),
                            "dst_role": entry.get("dst_role"),
                            "timestamp": entry.get("timestamp"),
                            "content": full_content,
                        })
                    
                    transcript_payload = {
                        "run_id": run_id,
                        "topology_id": topology_id,
                        "dataset": "mmlu",
                        "question_id": question_id,
                        "question": task_text,
                        "mode": f"adaptivek{k_used}",
                        "agent_num": len(realized_g.nodes),
                        "predicted_answer": str(pred),
                        "correct_answer": str(info["answer"]),
                        "is_correct": bool(is_correct),
                        "selected_edges": edge_indices_for_topo,
                        "k": k_used,
                        "messages": messages,
                    }
                    dump_json(transcript_abs_path, transcript_payload)
                    
                    # Prepare edge records
                    for msg_idx, entry in enumerate(edge_logs or []):
                        raw_content = entry.get("message", "")
                        full_content = "" if raw_content is None else str(raw_content)
                        preview = full_content[:500] if len(full_content) > 500 else full_content
                        edge_records.append({
                            "run_id": run_id,
                            "topology_id": topology_id,
                            "message_id": f"{run_id}_{msg_idx}",
                            "round": entry.get("round"),
                            "edge_type": entry.get("edge_type"),
                            "src_id": entry.get("src_id"),
                            "src_role": entry.get("src_role"),
                            "dst_id": entry.get("dst_id"),
                            "dst_role": entry.get("dst_role"),
                            "timestamp": entry.get("timestamp"),
                            "content_preview": preview,
                            "content_truncated": len(full_content) > 500,
                            "transcript_path": transcript_rel_path,
                        })
                    
                    # Prepare node records
                    node_records.extend(prepare_node_snapshots(run_id, topology_id, node_logs))
                    
                    # Save graph
                    graph_abs_path = os.path.join(log_paths["graphs_dir"], f"{run_id}.pt")
                    
                    if edge_indices_for_topo:
                        edge_index = torch.tensor(edge_indices_for_topo, dtype=torch.long).t().contiguous()
                    else:
                        edge_index = torch.zeros((2, 0), dtype=torch.long)
                    
                    pyg_data = {
                        "x": node_roles,
                        "edge_index": edge_index,
                        "y": task_text,
                        "metadata": {
                            "mode": f"adaptivek{k_used}",
                            "agent_nums": len(realized_g.nodes),
                            "is_correct": bool(is_correct),
                            "question": task_text[:200],
                            "selected_edges": edge_indices_for_topo,
                            "k": k_used,
                        },
                    }
                    torch.save(pyg_data, graph_abs_path)
                    
                    run_rows.append({
                        "run_id": run_id,
                        "dataset": "mmlu",
                        "question_id": question_id,
                        "question": task_text[:200] + "..." if len(task_text) > 200 else task_text,
                        "mode": f"adaptivek{k_used}",
                        "agent_num": len(realized_g.nodes),
                        "is_correct": bool(is_correct),
                        "predicted_answer": str(pred),
                        "correct_answer": str(info["answer"]),
                        "topology_id": topology_id,
                        "selected_edges": str(edge_indices_for_topo),
                        "k": k_used,
                        "tau": tau,
                        "transcript_path": transcript_rel_path,
                    })
                    
                except Exception as e:
                    print(f"[WARN] Transcript logging failed: {e}")
        
        # Flush this batch's logs
        if log_paths is not None:
            if run_rows:
                _write_run_rows_with_fields(log_paths["runs_csv"], run_rows, csv_fieldnames)
            if edge_records:
                append_jsonl(log_paths["edges_jsonl"], edge_records)
            if node_records:
                append_jsonl(log_paths["nodes_jsonl"], node_records)
        
        return results_list
    
    # Evaluation loop
    num_batches = (len(questions) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating") if tqdm else range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        batch_indices = list(range(start_idx, end_idx))
        
        results = asyncio.run(eval_batch(batch_indices))
        
        for is_correct, k in results:
            total += 1
            if is_correct:
                correct += 1
        
        # Progress logging
        print(f"[Batch {batch_idx + 1}/{num_batches}] Acc={correct}/{total} ({100*correct/max(1,total):.1f}%) | "
              f"Tokens: P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})")
    
    # K distribution summary
    k_arr = np.array(k_values)

    accuracy = correct / max(1, total)

    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "k_distribution": {
            "mean": float(k_arr.mean()),
            "std": float(k_arr.std()),
            "min": int(k_arr.min()),
            "max": int(k_arr.max()),
        },
        "token_usage": {
            "prompt_tokens": int(PromptTokens.instance().value),
            "completion_tokens": int(CompletionTokens.instance().value),
            "cost": float(Cost.instance().value),
        },
        "config": {
            "k_min": k_min,
            "k_max": k_max,
            "target_k": target_k,
            "difficulty_mode": difficulty_mode,
            "k_mapping_mode": k_mapping_mode,
            "tau": tau,
        },
    }

    print(f"\n[EvalAdaptiveK] === Final Results ===")
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print(f"K distribution: mean={k_arr.mean():.2f}, std={k_arr.std():.2f}, min={k_arr.min()}, max={k_arr.max()}")
    print(f"Tokens consumed: prompt={result['token_usage']['prompt_tokens']}, completion={result['token_usage']['completion_tokens']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Adaptive-K edge selection")
    
    # Mode
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for evaluation (default: use training output)")
    
    # Run config
    parser.add_argument("--run_name", type=str, default="adaptive_k_v1", help="Run name")
    parser.add_argument("--mode", type=str, default="FullConnected", help="Topology mode")
    parser.add_argument("--agent_nums", type=int, default=6, help="Number of agents")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["AnalyzeAgent"], help="Agent names")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-2024-08-06", help="LLM name")
    parser.add_argument("--decision_method", type=str, default="FinalRefer", help="Decision node method")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # K range
    parser.add_argument("--k_min", type=int, default=3, help="Minimum number of edges")
    parser.add_argument("--k_max", type=int, default=10, help="Maximum number of edges")
    parser.add_argument("--target_k", type=int, default=6, help="Target average k")
    
    # Difficulty scoring
    parser.add_argument("--difficulty_mode", type=str, default="score_mean",
                        choices=["score_sum", "score_mean", "top_score_sum"],
                        help="Difficulty scoring mode")
    parser.add_argument("--k_mapping_mode", type=str, default="percentile",
                        choices=["percentile", "linear"],
                        help="k mapping mode")
    
    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Teacher
    parser.add_argument("--teacher_temperature", type=float, default=0.7, help="Teacher temperature")
    parser.add_argument("--teacher_u_alpha", type=float, default=1.0, help="Teacher U scaling")
    
    # Model
    parser.add_argument("--model_score_mode", type=str, default="edge_mlp",
                        choices=["similarity", "edge_mlp"], help="Edge scoring mode")
    parser.add_argument("--edge_mlp_hidden", type=int, default=128, help="EdgeMLP hidden dim")
    parser.add_argument("--sim_mode", type=str, default="dot", choices=["dot", "cosine"], help="Similarity mode")
    parser.add_argument("--logit_temp", type=float, default=1.0, help="Logit temperature")
    
    # Gumbel sampling
    parser.add_argument("--tau", type=float, default=0.1, help="Gumbel temperature")

    # Inference edge selection
    parser.add_argument("--spatial_select_mode", type=str, default="topk", choices=["topk", "dagsample"], help="Edge selection method during inference")
    parser.add_argument("--dagsample_tau", type=float, default=0.0, help="Dagsample temperature (0 uses --tau)")
    
    # Evaluation
    parser.add_argument("--eval_split", type=str, default="val", help="Eval split")
    parser.add_argument("--limit_questions", type=int, default=153, help="Max number of eval questions")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--evaluatedataset_root", type=str, default=os.path.join("logs", "mmlu", "evaluatedataset"),
                        help="Eval log root directory")
    parser.add_argument("--no_evaluatedataset_logging", action="store_true", help="Disable eval log writing")
    
    # Data paths
    parser.add_argument("--data_root", type=str, default=None, help="Data root used to infer role order")
    parser.add_argument("--baseline_root", type=str, default=None, help="Baseline data directory")
    parser.add_argument("--phase1_root", type=str, default=None, help="Phase-1 data directory")
    parser.add_argument("--scorer_ckpt", type=str, default=None, help="Scorer checkpoint path")
    parser.add_argument("--num_questions", type=int, default=40, help="Number of training questions (half baseline, half phase-1)")
    
    args = parser.parse_args()
    
    # Set default paths
    if args.data_root is None:
        args.data_root = _pick_mmlu_logs_dir("traindataset", prefix="traindataset")
    if args.baseline_root is None:
        args.baseline_root = os.path.join(PROJECT_ROOT, "logs", "mmlu", "baseline")
    if args.phase1_root is None:
        args.phase1_root = os.path.join(PROJECT_ROOT, "logs", "mmlu", "phase1")
    if args.scorer_ckpt is None:
        preferred_ckpt = os.path.join(PROJECT_ROOT, "logs", "mmlu", "scorer", "best.pt")
        if os.path.isfile(preferred_ckpt):
            args.scorer_ckpt = preferred_ckpt
        else:
            # Best-effort: pick any existing scorer*/best.pt (e.g., older runs) without hardcoding names.
            candidates: List[str] = []
            try:
                mmlu_logs = os.path.join(PROJECT_ROOT, "logs", "mmlu")
                for entry in os.listdir(mmlu_logs):
                    if not str(entry).startswith("scorer"):
                        continue
                    ckpt = os.path.join(mmlu_logs, str(entry), "best.pt")
                    if os.path.isfile(ckpt):
                        candidates.append(ckpt)
            except Exception:
                candidates = []

            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                args.scorer_ckpt = candidates[0]
            else:
                args.scorer_ckpt = preferred_ckpt

    # Backward compatible: if the default evaluatedataset root doesn't exist, pick any existing evaluatedataset*.
    try:
        eval_root_rel = str(args.evaluatedataset_root)
        preferred_eval_rel = os.path.join("logs", "mmlu", "evaluatedataset")
        if os.path.normpath(eval_root_rel) == os.path.normpath(preferred_eval_rel):
            preferred_eval_abs = os.path.join(PROJECT_ROOT, preferred_eval_rel)
            if not os.path.isdir(preferred_eval_abs):
                picked_abs = _pick_mmlu_logs_dir("evaluatedataset", prefix="evaluatedataset")
                picked_name = os.path.basename(picked_abs)
                args.evaluatedataset_root = os.path.join("logs", "mmlu", picked_name)
    except Exception:
        pass
    
    output_dir = os.path.join(PROJECT_ROOT, "logs", "mmlu", "adaptive_k", args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[AdaptiveK] Output: {output_dir}")
    print(f"[AdaptiveK] K range: [{args.k_min}, {args.k_max}], target={args.target_k}")
    print(f"[AdaptiveK] Difficulty mode: {args.difficulty_mode}")
    
    # If neither train nor eval is set, run both by default.
    if not args.do_train and not args.do_eval:
        args.do_train = True
        args.do_eval = True
    
    ckpt_path = args.ckpt
    
    # Train
    if args.do_train:
        from CMAD.graph.graph_final import GraphFinal
        from experiments.mmlu.data_prep import infer_role_order_from_nodes_jsonl
        from experiments.mmlu.exp_utils import get_kwargs
        from experiments.mmlu.train_adaptivek import train_adaptive_k

        # Build Graph
        agent_names = list(args.agent_names)
        n_agents = int(args.agent_nums)
        if len(agent_names) == 1 and n_agents > 1:
            agent_names = agent_names * n_agents

        kwargs = get_kwargs(args.mode, n_agents)
        role_order = infer_role_order_from_nodes_jsonl(
            data_root=str(args.data_root),
            agent_num=int(n_agents),
            mode_filter=str(args.mode),
        )
        kwargs["node_kwargs"] = [{"role": r} for r in role_order]

        graph = GraphFinal(
            domain="mmlu",
            llm_name=args.llm_name,
            agent_names=agent_names,
            decision_method=args.decision_method,
            optimized_spatial=True,
            optimized_temporal=False,
            **kwargs,
        )
        if hasattr(graph, "clear_blocked_spatial_edges"):
            graph.clear_blocked_spatial_edges()
        
        output_dir, metrics = train_adaptive_k(
            graph=graph,
            baseline_root=args.baseline_root,
            phase1_root=args.phase1_root,
            scorer_ckpt=args.scorer_ckpt,
            output_dir=output_dir,
            mode_filter=args.mode,
            agent_num=args.agent_nums,
            num_questions=args.num_questions,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            teacher_temperature=args.teacher_temperature,
            teacher_u_alpha=args.teacher_u_alpha,
            difficulty_mode=args.difficulty_mode,
            k_mapping_mode=args.k_mapping_mode,
            k_min=args.k_min,
            k_max=args.k_max,
            target_k=args.target_k,
            model_score_mode=args.model_score_mode,
            edge_mlp_hidden=args.edge_mlp_hidden,
            sim_mode=args.sim_mode,
            logit_temp=args.logit_temp,
            seed=args.seed,
        )
        
        print(f"[AdaptiveK] Training complete. Checkpoint: {output_dir}")
        
        ckpt_path = os.path.join(output_dir, "adaptive_k_model.pt")
    
    # Evaluate
    if args.do_eval:
        if ckpt_path is None:
            # Try default path
            ckpt_path = os.path.join(output_dir, "adaptive_k_model.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"[Error] Checkpoint not found: {ckpt_path}")
            print("Please specify --ckpt or train first")
            sys.exit(1)
        
        # Set evaluation log directory
        evaluatedataset_dir = None
        if not args.no_evaluatedataset_logging:
            evaluatedataset_dir = os.path.join(str(args.evaluatedataset_root), args.run_name)
        
        result = run_eval_adaptive_k(
            ckpt_path=ckpt_path,
            split=args.eval_split,
            limit=args.limit_questions,
            batch_size=args.eval_batch_size,
            tau=args.tau,
            spatial_select_mode=str(args.spatial_select_mode),
            dagsample_tau=(None if float(args.dagsample_tau) <= 0 else float(args.dagsample_tau)),
            device=args.device,
            log_root=evaluatedataset_dir,
        )
        
        # Save results
        result_path = os.path.join(os.path.dirname(ckpt_path), "eval_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[AdaptiveK] Saved result to {result_path}")


if __name__ == "__main__":
    main()
