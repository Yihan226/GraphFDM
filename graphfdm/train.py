import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf

from common import (
    append_output,
    ensure_dir,
    load_config,
    read_json,
    remove_path,
    replace_pair,
    resolve_path,
    save_config,
    write_json,
)
from evaluate import evaluate_policy
from policy import (
    initialize_policy,
    sample_candidates,
    save_round_dataset,
    train_round,
)
from teacher import build_teacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def validate_config(cfg):
    if int(cfg.online.num_rounds) < 1:
        raise ValueError("online.num_rounds must be positive.")
    if int(cfg.online.k) < 2:
        raise ValueError("online.k must be at least two.")
    if str(cfg.online.split) != "train":
        raise ValueError("Candidates must be collected only from the train split.")
    if float(cfg.sampling.guide_scale) != 1.0:
        raise ValueError("Current-policy collection requires guide_scale=1.")
    for name in ["marginal_kl", "conditional_kl"]:
        if float(getattr(cfg.trust_region, name)) <= 0:
            raise ValueError(f"trust_region.{name} must be positive.")
    for section, name in [
        (cfg.collection, "batch_size"),
        (cfg.graph_train, "batch_size"),
        (cfg.controller_train, "batch_size"),
        (cfg.eval, "batch_size"),
    ]:
        if int(getattr(section, name)) < 1:
            raise ValueError(f"{name} must be positive.")


def teacher_report(weighted, skipped, diagnostics, teacher):
    target_count = int(weighted[["target_row_id", "seed"]].drop_duplicates().shape[0])
    eligible_targets = target_count - len(skipped)
    weights = weighted["group_weight"].to_numpy(np.float64)
    nodes = weighted["num_nodes"].to_numpy(np.float64)
    return {
        "candidate_avg_nmae": float(
            pd.to_numeric(weighted["final_sample_avg_nmae"], errors="coerce").mean()
        ),
        "candidate_validity": float(weighted["validity"].astype(bool).mean()),
        "eligible_target_fraction": eligible_targets / target_count,
        "median_teacher_ess": float(diagnostics["effective_sample_size"].median()),
        "weighted_mean_num_nodes": float(np.sum(nodes * weights) / eligible_targets),
        "teacher_size_entropy": float(diagnostics["teacher_size_entropy"].mean()),
        "target_kl_n": teacher["target_kl_n"],
        "target_kl_s": teacher["target_kl_s"],
        "realized_kl_total": teacher["kl_total"],
        "realized_kl_n": teacher["kl_n"],
        "realized_kl_s": teacher["kl_s"],
        "eta_n": teacher["eta_n"],
        "eta_s": teacher["eta_s"],
        "within_n_normalized_ess": teacher["within_n_normalized_ess"],
        "zero_valid_strata_fraction": teacher["zero_valid_strata_fraction"],
    }


def append_stage(path, summary, cfg, stage, round_id, role, teacher=None):
    row = dict(summary)
    row.update(
        {
            "stage": stage,
            "method": str(cfg.method),
            "round": round_id,
            "split": summary.get("split", stage),
            "checkpoint_role": role,
        }
    )
    if teacher:
        row.update(teacher)
    append_output(path, row)


def append_stage_once(path, summary, cfg, stage, round_id, role, teacher=None):
    if os.path.exists(path):
        existing = pd.read_csv(path, dtype=str)
        selected = (
            (existing["stage"] == str(stage))
            & (existing["round"] == str(round_id))
            & (existing["checkpoint_role"] == str(role))
        )
        if bool(selected.any()):
            return
    append_stage(path, summary, cfg, stage, round_id, role, teacher)


def atomic_torch_save(payload, path):
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_frame_save(frame, path):
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False, float_format="%.8g")
    os.replace(temporary, path)


def initialize_wandb(cfg, output_dir):
    metadata_path = os.path.join(output_dir, "wandb.json")
    if os.path.isfile(metadata_path):
        metadata = read_json(metadata_path)
    else:
        metadata = {
            "id": wandb.util.generate_id(),
            "project": str(cfg.wandb.project),
            "name": f"{cfg.wandb.name}_{os.path.basename(output_dir)}",
        }
        write_json(metadata_path, metadata)
    return wandb.init(
        project=metadata["project"],
        name=metadata["name"],
        id=metadata["id"],
        resume="allow",
        mode=str(cfg.wandb.mode),
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def log_scalars(run, prefix, values, payload):
    for key, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            payload[f"{prefix}/{key}"] = float(value)


def log_round(run, summary):
    payload = {"round": int(summary["round"])}
    log_scalars(run, "candidates", summary["candidate_summary"], payload)
    log_scalars(run, "teacher", summary["teacher_summary"], payload)
    log_scalars(run, "train_graph", summary["graph_training"], payload)
    log_scalars(run, "train_controller", summary["controller_training"], payload)
    log_scalars(run, "val", summary["validation"], payload)
    run.log(payload)


def log_test(run, summary):
    payload = {}
    log_scalars(run, "test10000", summary, payload)
    run.log(payload)


def save_candidate_payload(path, samples, candidates, names, summary):
    atomic_torch_save(
        {
            "samples": samples,
            "frame": candidates,
            "objective_names": list(names),
            "summary": summary,
        },
        path,
    )


def load_candidate_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload["samples"],
        payload["frame"].copy(),
        list(payload["objective_names"]),
        payload["summary"],
    )


def completed_rounds(output_dir, total_rounds, round_offset=0):
    rounds = []
    for round_id in range(int(round_offset), int(round_offset) + int(total_rounds)):
        path = os.path.join(output_dir, f"round_{round_id:03d}", "summary.json")
        if not os.path.exists(path):
            break
        rounds.append(read_json(path))
    return rounds


def best_from_rounds(rounds):
    if not rounds:
        return float("inf"), None
    best = min(rounds, key=lambda item: float(item["validation"]["avg_nmae"]))
    return float(best["validation"]["avg_nmae"]), int(best["round"])


def commit_pending(
    pending_path,
    next_graph,
    next_size,
    current_graph,
    current_size,
    best_graph,
    best_size,
    root_output,
    cfg,
    best_val,
):
    pending = read_json(pending_path)
    validation = pending["validation"]
    round_id = int(pending["round"])
    current_val = float(validation["avg_nmae"])
    replace_pair(next_graph, next_size, current_graph, current_size)
    if current_val < best_val:
        replace_pair(next_graph, next_size, best_graph, best_size)
    append_stage_once(
        root_output,
        validation,
        cfg,
        "val",
        round_id,
        "current",
        pending["output_teacher"],
    )
    summary_path = os.path.join(os.path.dirname(pending_path), "summary.json")
    write_json(summary_path, pending["round_summary"])
    remove_path(os.path.dirname(next_graph))
    remove_path(pending_path)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    validate_config(cfg)
    output_dir = resolve_path(cfg.output_dir)
    saved_config = os.path.join(output_dir, "config.yaml")
    if args.resume:
        if not os.path.isfile(saved_config):
            raise FileNotFoundError(f"Resume config does not exist: {saved_config}")
        cfg = load_config(saved_config)
        validate_config(cfg)
    else:
        if os.path.exists(output_dir) and os.listdir(output_dir):
            raise ValueError(f"Output directory is not empty: {output_dir}")
        ensure_dir(output_dir)
        save_config(saved_config, cfg)
    final_summary_path = os.path.join(output_dir, "summary.json")
    if args.resume and os.path.isfile(final_summary_path):
        remove_path(os.path.join(output_dir, "_current"))
        remove_path(os.path.join(output_dir, "_next"))
        print(f"Run already completed: {output_dir}", flush=True)
        return
    wandb_run = initialize_wandb(cfg, output_dir)
    current_dir = ensure_dir(os.path.join(output_dir, "_current"))
    current_graph = os.path.join(current_dir, "graph.ckpt")
    current_size = os.path.join(current_dir, "size.ckpt")
    if not args.resume:
        initialize_policy(cfg, current_graph, current_size)
    elif not os.path.isfile(current_graph) or not os.path.isfile(current_size):
        raise FileNotFoundError("Resume requires the committed _current graph and size checkpoints.")
    checkpoint_dir = ensure_dir(os.path.join(output_dir, "checkpoints"))
    best_graph = os.path.join(checkpoint_dir, "best_graph.ckpt")
    best_size = os.path.join(checkpoint_dir, "best_size.ckpt")
    final_graph = os.path.join(checkpoint_dir, "final_graph.ckpt")
    final_size = os.path.join(checkpoint_dir, "final_size.ckpt")
    root_output = os.path.join(output_dir, "output.csv")
    round_offset = int(getattr(cfg.online, "round_offset", 0))
    rounds = completed_rounds(output_dir, cfg.online.num_rounds, round_offset)
    best_val, best_round = best_from_rounds(rounds)
    start = time.time()
    for round_id in range(
        round_offset + len(rounds),
        round_offset + int(cfg.online.num_rounds),
    ):
        round_dir = ensure_dir(os.path.join(output_dir, f"round_{round_id:03d}"))
        candidate_path = os.path.join(round_dir, "candidates.pt")
        dataset_path = os.path.join(round_dir, "dataset.pt")
        prepared_path = os.path.join(round_dir, "prepared.json")
        if os.path.isfile(prepared_path):
            if not os.path.isfile(dataset_path):
                raise FileNotFoundError(f"Prepared dataset is missing: {dataset_path}")
            prepared = read_json(prepared_path)
            candidate_summary = prepared["candidate_summary"]
            teacher_summary = prepared["teacher_summary"]
            compact_teacher = prepared["output_teacher"]
            remove_path(candidate_path)
        else:
            if os.path.isfile(candidate_path):
                samples, candidates, names, candidate_summary = load_candidate_payload(candidate_path)
                print(f"Reusing candidates for round {round_id}: {candidate_path}", flush=True)
            else:
                samples, candidates, names, candidate_summary = sample_candidates(
                    cfg,
                    current_graph,
                    current_size,
                    round_id,
                )
                save_candidate_payload(candidate_path, samples, candidates, names, candidate_summary)
            weighted, skipped, diagnostics, teacher_summary = build_teacher(candidates, names, cfg)
            atomic_frame_save(weighted, os.path.join(round_dir, "candidates.csv"))
            size_payload = torch.load(current_size, map_location="cpu", weights_only=False)
            max_nodes = int(size_payload["model_config"]["max_nodes"])
            temporary_dataset = dataset_path + ".tmp"
            save_round_dataset(weighted, samples, names, max_nodes, temporary_dataset)
            os.replace(temporary_dataset, dataset_path)
            compact_teacher = teacher_report(weighted, skipped, diagnostics, teacher_summary)
            prepared = {
                "candidate_summary": candidate_summary,
                "teacher_summary": teacher_summary,
                "output_teacher": compact_teacher,
            }
            write_json(prepared_path, prepared)
            remove_path(candidate_path)
            del samples, candidates, weighted, size_payload
        next_dir = ensure_dir(os.path.join(output_dir, "_next"))
        next_graph = os.path.join(next_dir, "graph.ckpt")
        next_size = os.path.join(next_dir, "size.ckpt")
        pending_path = os.path.join(round_dir, "pending.json")
        if not os.path.isfile(pending_path):
            remove_path(next_dir)
            ensure_dir(next_dir)
            trained_graph, trained_size, graph_metrics, controller_metrics = train_round(
                cfg,
                dataset_path,
                current_graph,
                current_size,
                round_id,
                next_dir,
            )
            validation = evaluate_policy(
                cfg,
                trained_graph,
                trained_size,
                "val",
                cfg.eval.val_total_samples,
                cfg.eval.val_seed,
            )
            compact_teacher["controller_ce"] = controller_metrics["controller_ce"]
            round_summary = {
                "round": round_id,
                "candidate_summary": candidate_summary,
                "teacher_summary": teacher_summary,
                "graph_training": graph_metrics,
                "controller_training": controller_metrics,
                "validation": validation,
            }
            write_json(
                pending_path,
                {
                    "round": round_id,
                    "validation": validation,
                    "output_teacher": compact_teacher,
                    "round_summary": round_summary,
                },
            )
        elif not os.path.isfile(next_graph) or not os.path.isfile(next_size):
            raise FileNotFoundError("Pending round is missing its _next checkpoint pair.")
        commit_pending(
            pending_path,
            next_graph,
            next_size,
            current_graph,
            current_size,
            best_graph,
            best_size,
            root_output,
            cfg,
            best_val,
        )
        rounds = completed_rounds(output_dir, cfg.online.num_rounds, round_offset)
        best_val, best_round = best_from_rounds(rounds)
        log_round(wandb_run, rounds[-1])
    replace_pair(current_graph, current_size, final_graph, final_size)
    test_dir = os.path.join(output_dir, "test_10000")
    test_summary_path = os.path.join(test_dir, "summary.json")
    if os.path.isfile(test_summary_path):
        test_summary = read_json(test_summary_path)
    else:
        test_summary = evaluate_policy(
            cfg,
            best_graph,
            best_size,
            "test",
            cfg.eval.test_total_samples,
            cfg.eval.test_seed,
            test_dir,
        )
    append_stage_once(root_output, test_summary, cfg, "test10000", best_round, "best")
    log_test(wandb_run, test_summary)
    summary = {
        "method": str(cfg.method),
        "completed_rounds": int(cfg.online.num_rounds),
        "round_offset": round_offset,
        "best_round": best_round,
        "best_val_avg_nmae": best_val,
        "best_graph_checkpoint": best_graph,
        "best_size_checkpoint": best_size,
        "final_graph_checkpoint": final_graph,
        "final_size_checkpoint": final_size,
        "elapsed_seconds": float(time.time() - start),
        "rounds": rounds,
        "test10000": test_summary,
    }
    write_json(os.path.join(output_dir, "summary.json"), summary)
    remove_path(current_dir)
    wandb_run.finish()
    print(f"Completed GraphFDM: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
