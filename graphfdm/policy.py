import os

import numpy as np
import pandas as pd
import torch

from common import (
    build_context,
    build_loader,
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
    replace_pair,
    resolve_path,
    runtime_config,
    save_graph_model,
    set_seed,
    summarize_frame,
)
from losses import dense_batch, loss_per_graph, weighted_group_mean
from size_controller import (
    initialize_controller,
    load_controller,
    save_controller,
)


def initialize_policy(cfg, graph_output, size_output):
    runtime = runtime_config(cfg, cfg.online.seed, cfg.collection.batch_size)
    datamodule, dataset_infos, sampling_metrics, model_kwargs = build_context(runtime)
    device = torch.device("cuda" if torch.cuda.is_available() and int(cfg.general.gpus) > 0 else "cpu")
    source_graph = resolve_path(cfg.init_graph_checkpoint)
    model = load_graph_model(source_graph, runtime, model_kwargs, device)
    p0 = model.node_dist.prob.detach().to(device=device, dtype=torch.float32)
    p0 /= p0.sum()
    source_size = getattr(cfg, "init_size_checkpoint", None)
    if source_size:
        source_size = resolve_path(source_size)
        controller, _ = load_controller(source_size, device)
        if controller.condition_dim != int(model.ydim):
            raise ValueError("Initial graph and size checkpoints have incompatible condition dimensions.")
        controller_base_probability(model, controller.max_nodes, device)
        ensure_dir(os.path.dirname(graph_output))
        replace_pair(source_graph, source_size, graph_output, size_output)
        del controller, model, datamodule, dataset_infos, sampling_metrics, model_kwargs
        clear_cuda()
        return
    controller = initialize_controller(
        int(model.ydim),
        int(p0.numel() - 1),
        int(cfg.controller.hidden_dim),
        int(cfg.controller.node_embedding_dim),
    ).to(device)
    probe = torch.zeros((3, int(model.ydim)), dtype=torch.float32, device=device)
    with torch.no_grad():
        probability = controller.size_distribution(probe, p0)
    if not torch.allclose(probability, p0[None, :].expand_as(probability), atol=1.0e-7):
        raise ValueError("Fresh size controller does not reproduce p0.")
    temporary_size = size_output + ".init"
    ensure_dir(os.path.dirname(graph_output))
    save_controller(controller, temporary_size, -1)
    replace_pair(source_graph, temporary_size, graph_output, size_output)
    os.remove(temporary_size)
    del controller, model, datamodule, dataset_infos, sampling_metrics, model_kwargs
    clear_cuda()


def sample_candidates(cfg, graph_checkpoint, size_checkpoint, round_id):
    seed = int(cfg.online.seed) + int(round_id) * 100003
    runtime = runtime_config(cfg, seed, cfg.collection.batch_size)
    datamodule, dataset_infos, sampling_metrics, model_kwargs = build_context(runtime)
    y, y_raw, row_ids, sources, split, names = collect_targets(
        datamodule,
        runtime,
        cfg.online.split,
    )
    if len(split) == 0:
        raise ValueError("The selected online split contains no target conditions.")
    device = torch.device("cuda" if torch.cuda.is_available() and int(cfg.general.gpus) > 0 else "cpu")
    model = load_graph_model(graph_checkpoint, runtime, model_kwargs, device)
    controller, _ = load_controller(size_checkpoint, device)
    p0 = controller_base_probability(model, controller.max_nodes, device)
    target_offsets = np.repeat(np.arange(len(y), dtype=np.int64), int(cfg.online.k))
    sample_indices = np.tile(np.arange(int(cfg.online.k), dtype=np.int64), len(y))
    samples = []
    metadata = []
    raw_targets = []
    normalized_targets = []
    controller.eval()
    for batch_start in range(0, len(target_offsets), int(cfg.collection.batch_size)):
        offsets = target_offsets[batch_start : batch_start + int(cfg.collection.batch_size)]
        condition = y[offsets].to(device)
        with torch.no_grad():
            controller_probability = controller.size_distribution(condition, p0)
            proposal_probability = (
                (1.0 - float(cfg.collection.epsilon)) * controller_probability
                + float(cfg.collection.epsilon) * p0[None, :]
            )
        controller_np = controller_probability.detach().cpu().numpy()
        proposal_np = proposal_probability.detach().cpu().numpy()
        num_nodes = inverse_cdf_sample(
            proposal_np,
            seed * 1000003 + batch_start + 17011,
        )
        set_seed(seed * 1000003 + batch_start)
        with torch.no_grad():
            batch_samples = model.sample_batch(
                batch_id=batch_start,
                batch_size=len(offsets),
                y=condition,
                keep_chain=0,
                number_chain_steps=model.number_chain_steps,
                num_nodes=torch.tensor(num_nodes, dtype=torch.long, device=device),
            )
        samples.extend(batch_samples)
        raw_targets.extend(y_raw[offsets].numpy().tolist())
        normalized_targets.extend(y[offsets].numpy().tolist())
        entropy = -(
            controller_np * np.log(np.maximum(controller_np, 1.0e-30))
        ).sum(axis=1)
        for local_index, target_offset in enumerate(offsets.tolist()):
            global_index = batch_start + local_index
            n_value = int(num_nodes[local_index])
            metadata.append(
                {
                    "round_id": int(round_id),
                    "seed": seed,
                    "target_row_id": int(row_ids[target_offset]),
                    "sample_index": int(sample_indices[global_index]),
                    "graph_record_id": int(global_index),
                    "num_nodes": n_value,
                    "controller_probability": float(controller_np[local_index, n_value]),
                    "proposal_probability": float(proposal_np[local_index, n_value]),
                    "controller_entropy": float(entropy[local_index]),
                }
            )
    decoded = decode_samples(dataset_infos, samples)
    raw_targets = np.asarray(raw_targets, dtype=np.float32)
    normalized_targets = np.asarray(normalized_targets, dtype=np.float32)
    rows = []
    for index, meta in enumerate(metadata):
        row = dict(meta)
        row["target_source_smiles"] = sources[int(meta["target_row_id"])]
        for objective_index, name in enumerate(names):
            row[f"target_{name}"] = float(raw_targets[index, objective_index])
            row[f"target_norm_{name}"] = float(normalized_targets[index, objective_index])
        row.update(decoded[index])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = evaluate_final_surface(
        frame,
        raw_targets,
        sampling_metrics,
        cfg.collection.oracle_batch_size,
    )
    summary = summarize_frame(frame, names)
    target_rows = frame.drop_duplicates(["target_row_id", "seed"])
    with torch.no_grad():
        target_probability = controller.size_distribution(y.to(device), p0).detach().cpu().numpy()
    summary.update(
        {
            "target_count": int(len(split)),
            "candidate_count": int(len(frame)),
            "mean_num_nodes": float(frame["num_nodes"].mean()),
            "size_entropy": float(target_rows["controller_entropy"].mean()),
            "size_kl_to_p0": distribution_kl(
                target_probability,
                p0.detach().cpu().numpy()[None, :],
            ),
            "p0_entropy": distribution_entropy(p0.detach().cpu().numpy()),
        }
    )
    del controller, model, datamodule, dataset_infos, sampling_metrics, model_kwargs
    clear_cuda()
    return samples, frame, names, summary


def save_round_dataset(frame, samples, objective_names, max_nodes, path):
    selected = frame.index[frame["group_weight"] > 0].tolist()
    count = len(selected)
    x = torch.zeros((count, int(max_nodes)), dtype=torch.int16)
    e = torch.zeros((count, int(max_nodes), int(max_nodes)), dtype=torch.uint8)
    pos = torch.zeros_like(e)
    y = torch.zeros((count, len(objective_names)), dtype=torch.float32)
    num_nodes = torch.zeros(count, dtype=torch.int16)
    weight = torch.zeros(count, dtype=torch.float32)
    group_id = torch.zeros(count, dtype=torch.int32)
    for output_index, row_index in enumerate(selected):
        row = frame.loc[row_index]
        nodes, edges, positions = samples[int(row["graph_record_id"])]
        n_value = int(nodes.shape[0])
        if n_value > int(max_nodes):
            raise ValueError("Candidate exceeds configured maximum motif count.")
        x[output_index, :n_value] = nodes.detach().cpu().to(torch.int16)
        e[output_index, :n_value, :n_value] = edges.detach().cpu().to(torch.uint8)
        pos[output_index, :n_value, :n_value] = positions.detach().cpu().to(torch.uint8)
        y[output_index] = torch.tensor(
            [float(row[f"target_norm_{name}"]) for name in objective_names],
            dtype=torch.float32,
        )
        num_nodes[output_index] = n_value
        weight[output_index] = float(row["group_weight"])
        group_id[output_index] = int(row["sdc_group_id"])
    payload = {
        "x": x,
        "e": e,
        "pos": pos,
        "y": y,
        "num_nodes": num_nodes,
        "weight": weight,
        "group_id": group_id,
        "metadata": {
            "objective_names": list(objective_names),
            "reduction": "graph_normalized_group_weighted",
            "count": count,
            "max_nodes": int(max_nodes),
        },
    }
    torch.save(payload, path)


def graph_epoch(model, loader, optimizer, cfg, device):
    model.train()
    totals = {"graph_loss": 0.0, "X_ce": 0.0, "E_ce": 0.0, "pos_ce": 0.0}
    steps = 0
    for batch in loader:
        dense, node_mask, y = dense_batch(model, batch, device)
        noisy = model.apply_noise(dense.X, dense.E, dense.pos, y, node_mask)
        prediction = model.forward(noisy)
        components = loss_per_graph(model, prediction, dense)
        weights = batch["weight"].to(device=device, dtype=components["total"].dtype)
        groups = batch["group_id"].to(device=device, dtype=torch.long)
        loss = weighted_group_mean(components["total"], weights, groups)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.graph_train.clip_grad))
        optimizer.step()
        totals["graph_loss"] += float(loss.detach().cpu())
        totals["X_ce"] += float(weighted_group_mean(components["x"], weights, groups).detach().cpu())
        totals["E_ce"] += float(weighted_group_mean(components["e"], weights, groups).detach().cpu())
        totals["pos_ce"] += float(weighted_group_mean(components["pos"], weights, groups).detach().cpu())
        steps += 1
    return {key: value / steps for key, value in totals.items()}


def controller_epoch(controller, p0, loader, optimizer, cfg, device):
    controller.train()
    total = 0.0
    steps = 0
    for batch in loader:
        condition = batch["y"].to(device=device, dtype=torch.float32)
        num_nodes = batch["num_nodes"].to(device=device, dtype=torch.long)
        weights = batch["weight"].to(device=device, dtype=torch.float32)
        groups = batch["group_id"].to(device=device, dtype=torch.long)
        probability = controller.size_distribution(condition, p0)
        selected = probability.gather(1, num_nodes[:, None]).squeeze(1)
        loss = weighted_group_mean(-torch.log(selected.clamp_min(1.0e-30)), weights, groups)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), float(cfg.controller_train.clip_grad))
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return {"controller_ce": total / steps}


def train_round(cfg, dataset_path, graph_checkpoint, size_checkpoint, round_id, output_dir):
    runtime = runtime_config(cfg, cfg.graph_train.seed, cfg.graph_train.batch_size)
    set_seed(int(cfg.graph_train.seed) + int(round_id) * 100003)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    datamodule, dataset_infos, sampling_metrics, model_kwargs = build_context(runtime)
    device = torch.device("cuda" if torch.cuda.is_available() and int(cfg.general.gpus) > 0 else "cpu")
    model = load_graph_model(graph_checkpoint, runtime, model_kwargs, device, evaluate=False)
    controller, _ = load_controller(size_checkpoint, device)
    p0 = controller_base_probability(model, controller.max_nodes, device)
    graph_loader = build_loader(
        payload,
        cfg.graph_train.batch_size,
        int(cfg.graph_train.seed) + int(round_id) * 100003,
    )
    controller_loader = build_loader(
        payload,
        cfg.controller_train.batch_size,
        int(cfg.controller_train.seed) + int(round_id) * 100003,
    )
    graph_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.graph_train.lr),
        weight_decay=float(cfg.graph_train.weight_decay),
        amsgrad=True,
    )
    controller_optimizer = torch.optim.AdamW(
        controller.parameters(),
        lr=float(cfg.controller_train.lr),
        weight_decay=float(cfg.controller_train.weight_decay),
        amsgrad=True,
    )
    graph_metrics = {}
    for epoch in range(int(cfg.graph_train.epochs_per_round)):
        graph_metrics = graph_epoch(model, graph_loader, graph_optimizer, cfg, device)
        print(
            f"round={round_id} graph_epoch={epoch + 1} loss={graph_metrics['graph_loss']:.6f}",
            flush=True,
        )
    controller_metrics = {}
    for epoch in range(int(cfg.controller_train.epochs_per_round)):
        controller_metrics = controller_epoch(
            controller,
            p0,
            controller_loader,
            controller_optimizer,
            cfg,
            device,
        )
        print(
            f"round={round_id} controller_epoch={epoch + 1} ce={controller_metrics['controller_ce']:.6f}",
            flush=True,
        )
    graph_output = os.path.join(output_dir, "graph.ckpt")
    size_output = os.path.join(output_dir, "size.ckpt")
    ensure_dir(output_dir)
    save_graph_model(model, graph_checkpoint, graph_output)
    save_controller(controller, size_output, round_id)
    del controller, model, datamodule, dataset_infos, sampling_metrics, model_kwargs, payload
    clear_cuda()
    return graph_output, size_output, graph_metrics, controller_metrics
