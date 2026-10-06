import os

import numpy as np
import pandas as pd
import torch

from common import (
    build_context,
    clear_cuda,
    collect_targets,
    controller_base_probability,
    decode_samples,
    distribution_entropy,
    distribution_kl,
    ensure_dir,
    evaluate_final_surface,
    inverse_cdf_sample,
    load_graph_model,
    runtime_config,
    set_seed,
    summarize_frame,
    write_json,
)
from size_controller import load_controller


def sample_total(model, controller, p0, y, total_samples, seed, batch_size):
    sample_ids = np.arange(int(total_samples), dtype=np.int64)
    target_offsets = sample_ids % len(y)
    sample_indices = sample_ids // len(y)
    samples = []
    metadata = []
    controller.eval()
    for batch_start in range(0, len(target_offsets), int(batch_size)):
        offsets = target_offsets[batch_start : batch_start + int(batch_size)]
        condition = y[offsets].to(model.device)
        with torch.no_grad():
            probability = controller.size_distribution(condition, p0)
        probability_np = probability.detach().cpu().numpy()
        num_nodes = inverse_cdf_sample(
            probability_np,
            int(seed) * 1000003 + batch_start + 17011,
        )
        set_seed(int(seed) * 1000003 + batch_start)
        with torch.no_grad():
            batch_samples = model.sample_batch(
                batch_id=batch_start,
                batch_size=len(offsets),
                y=condition,
                keep_chain=0,
                number_chain_steps=model.number_chain_steps,
                num_nodes=torch.tensor(num_nodes, dtype=torch.long, device=model.device),
            )
        samples.extend(batch_samples)
        for local_index, target_offset in enumerate(offsets.tolist()):
            global_index = batch_start + local_index
            n_value = int(num_nodes[local_index])
            metadata.append(
                {
                    "seed": int(seed),
                    "target_row_id": int(target_offset),
                    "sample_index": int(sample_indices[global_index]),
                    "num_nodes": n_value,
                    "size_probability": float(probability_np[local_index, n_value]),
                }
            )
    return samples, metadata, target_offsets


def distribution_metrics(frame, sampling_metrics):
    smiles = [value for value in frame["final_smiles"] if isinstance(value, str) and value]
    metrics = sampling_metrics._compute_distribution_metrics(smiles)
    unique = len(set(smiles))
    metrics.update(
        {
            "valid_unique": unique,
            "valid_count": len(smiles),
            "unique_fraction": unique / len(smiles) if smiles else float("nan"),
            "repetition_fraction": 1.0 - unique / len(smiles) if smiles else float("nan"),
        }
    )
    return metrics


def evaluate_policy(
    cfg,
    graph_checkpoint,
    size_checkpoint,
    split_name,
    total_samples,
    seed,
    output_dir=None,
):
    runtime = runtime_config(cfg, seed, cfg.eval.batch_size)
    datamodule, dataset_infos, sampling_metrics, model_kwargs = build_context(runtime)
    y, y_raw, row_ids, sources, split, names = collect_targets(
        datamodule,
        runtime,
        split_name,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and int(cfg.general.gpus) > 0 else "cpu")
    model = load_graph_model(graph_checkpoint, runtime, model_kwargs, device)
    controller, _ = load_controller(size_checkpoint, device)
    p0 = controller_base_probability(model, controller.max_nodes, device)
    with torch.no_grad():
        target_probability = controller.size_distribution(y.to(device), p0).detach().cpu().numpy()
    samples, metadata, target_offsets = sample_total(
        model,
        controller,
        p0,
        y,
        total_samples,
        seed,
        cfg.eval.batch_size,
    )
    decoded = decode_samples(dataset_infos, samples)
    targets = y_raw[target_offsets].numpy()
    rows = []
    for index, meta in enumerate(metadata):
        row = dict(meta)
        row.update(
            {
                "model_name": str(cfg.method),
                "checkpoint_path": graph_checkpoint,
                "size_checkpoint_path": size_checkpoint,
                "split": str(split_name),
                "target_source_smiles": sources[int(target_offsets[index])],
            }
        )
        for objective_index, name in enumerate(names):
            row[f"target_{name}"] = float(targets[index, objective_index])
        row.update(decoded[index])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = evaluate_final_surface(
        frame,
        targets,
        sampling_metrics,
        cfg.eval.oracle_batch_size,
    )
    summary = summarize_frame(frame, names)
    if bool(getattr(cfg.eval, "compute_distribution_metrics", True)):
        summary.update(distribution_metrics(frame, sampling_metrics))
    summary.update(
        {
            "method": str(cfg.method),
            "split": str(split_name),
            "count": int(len(frame)),
            "target_count": int(len(split)),
            "mean_num_nodes": float(frame["num_nodes"].mean()),
            "size_entropy": distribution_entropy(target_probability),
            "size_kl_to_p0": distribution_kl(
                target_probability,
                p0.detach().cpu().numpy()[None, :],
            ),
        }
    )
    if output_dir:
        ensure_dir(output_dir)
        frame.to_csv(os.path.join(output_dir, "all_samples.csv"), index=False, float_format="%.8g")
        pd.DataFrame([summary]).to_csv(
            os.path.join(output_dir, "output.csv"),
            index=False,
            float_format="%.4f",
        )
        write_json(os.path.join(output_dir, "summary.json"), summary)
    del controller, model, datamodule, dataset_infos, sampling_metrics, model_kwargs
    clear_cuda()
    return summary
