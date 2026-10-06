import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, open_dict
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path, overrides=None):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def resolve_path(path):
    path = str(path)
    return path if os.path.isabs(path) else str(PROJECT_ROOT / path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(clean_json(payload), handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(path, cfg):
    OmegaConf.save(config=cfg, f=path)


def append_output(path, row):
    ensure_dir(os.path.dirname(path))
    new_row = pd.DataFrame([clean_json(row)])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        columns = list(existing.columns) + [name for name in new_row.columns if name not in existing]
        output = pd.concat(
            [existing.reindex(columns=columns), new_row.reindex(columns=columns)],
            ignore_index=True,
        )
    else:
        output = new_row
    output.to_csv(path, index=False, float_format="%.6g")


def set_seed(seed):
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_config(cfg, seed, batch_size):
    runtime = OmegaConf.load(resolve_path(cfg.base_config))
    runtime = OmegaConf.create(OmegaConf.to_container(runtime, resolve=False))
    OmegaConf.set_struct(runtime, False)
    with open_dict(runtime):
        runtime.general.name = str(cfg.run_name)
        runtime.general.wandb = "disabled"
        runtime.general.gpus = int(cfg.general.gpus)
        runtime.general.resume = None
        runtime.general.test_only = None
        runtime.general.pt_checkpoint = None
        runtime.general.chains_to_save = 0
        runtime.general.final_model_chains_to_save = 0
        runtime.general.samples_to_save = 0
        runtime.general.final_model_samples_to_save = 0
        runtime.train.batch_size = int(batch_size)
        runtime.train.num_workers = 0
        runtime.train.seed = int(seed)
        runtime.model.guide_scale = float(cfg.sampling.guide_scale)
        runtime.general.compute_distribution_metrics = bool(
            getattr(cfg.eval, "compute_distribution_metrics", True)
        )
        if cfg.sampling.diffusion_steps is not None:
            runtime.model.diffusion_steps = int(cfg.sampling.diffusion_steps)
    return runtime


def build_context(runtime_cfg):
    from analysis.visualization import MolecularVisualization
    from datasets import conditional_dataset as dataset_mod
    from metrics.molecular_metrics_sampling import SamplingMolecularMetrics
    from metrics.molecular_metrics_train import TrainMolecularMetricsDiscrete

    datamodule = dataset_mod.DataModule(runtime_cfg)
    datamodule.prepare_data()
    dataset_infos = dataset_mod.DataInfos(datamodule=datamodule, cfg=runtime_cfg)
    train_smiles, reference_smiles = datamodule.get_train_smiles()
    if not bool(getattr(runtime_cfg.general, "compute_distribution_metrics", True)):
        reference_smiles = []
    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles, reference_smiles)
    visualization_tools = MolecularVisualization(dataset_infos)
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
    }
    return datamodule, dataset_infos, sampling_metrics, model_kwargs


def load_graph_model(path, runtime_cfg, model_kwargs, device, evaluate=True):
    from diffusion_model import Graph_DiT
    from runtime.common import sync_model_runtime_config

    model = Graph_DiT.load_from_checkpoint(
        resolve_path(path),
        weights_only=False,
        strict=False,
        **model_kwargs,
    )
    model = sync_model_runtime_config(model, runtime_cfg)
    model.to(device)
    if evaluate:
        model.eval()
    return model


def save_graph_model(model, template_path, output_path):
    payload = torch.load(resolve_path(template_path), map_location="cpu", weights_only=False)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["state_dict"] = state_dict
        if isinstance(payload.get("hyper_parameters"), dict):
            payload["hyper_parameters"]["cfg"] = model.cfg
    else:
        payload = {"state_dict": state_dict}
    ensure_dir(os.path.dirname(output_path))
    torch.save(payload, output_path)


def replace_pair(graph_source, size_source, graph_destination, size_destination):
    ensure_dir(os.path.dirname(graph_destination))
    graph_tmp = graph_destination + ".tmp"
    size_tmp = size_destination + ".tmp"
    shutil.copy2(graph_source, graph_tmp)
    shutil.copy2(size_source, size_tmp)
    os.replace(graph_tmp, graph_destination)
    os.replace(size_tmp, size_destination)


def remove_path(path):
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def split_frame(datamodule, split):
    train, val, test = datamodule.get_data_split()
    key = str(split).strip().lower()
    if key == "train":
        return train.reset_index(drop=True)
    if key in {"val", "valid", "validation"}:
        return val.reset_index(drop=True)
    if key == "test":
        return test.reset_index(drop=True)
    raise ValueError(f"Unknown split: {split}")


def collect_targets(datamodule, runtime_cfg, split):
    from datasets.conditional_dataset import normalize_objective_vector
    from shared import get_objective_names

    frame = split_frame(datamodule, split)
    names = list(get_objective_names(runtime_cfg))
    raw = []
    normalized = []
    sources = []
    for _, row in frame.iterrows():
        raw.append([float(row[name]) for name in names])
        normalized.append(
            normalize_objective_vector(row, names, runtime_cfg, datamodule.normalization_stats)
        )
        sources.append(str(row["smiles"]))
    return (
        torch.tensor(normalized, dtype=torch.float32),
        torch.tensor(raw, dtype=torch.float32),
        np.arange(len(frame), dtype=np.int64),
        sources,
        frame,
        names,
    )


def decode_samples(dataset_infos, samples):
    from motif_utils import decode_motif_graphs, enrich_decoded_graphs_with_validity

    decoded = decode_motif_graphs(dataset_infos.tokenizer, samples, cache={})
    decoded = enrich_decoded_graphs_with_validity(
        decoded,
        ensure_connected=getattr(dataset_infos, "ensure_connected", True),
    )
    rows = []
    for item in decoded:
        rows.append(
            {
                "reward_smiles": item.get("reward_smiles"),
                "final_smiles": item.get("smiles"),
                "validity_nc": bool(item.get("validity_nc", False)),
                "validity_raw": bool(item.get("validity_raw", False)),
                "validity": item.get("smiles") is not None,
                "num_components": int(item.get("num_components", 0) or 0),
            }
        )
    return rows


def predict_batched(sampling_metrics, name, smiles, batch_size):
    outputs = []
    for start in range(0, len(smiles), int(batch_size)):
        batch = smiles[start : start + int(batch_size)]
        outputs.append(np.asarray(sampling_metrics._predict_objective(name, batch), dtype=np.float32))
        sampling_metrics.eval_cache.clear()
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.float32)


def evaluate_final_surface(frame, targets, sampling_metrics, oracle_batch_size):
    from metrics.metric_utils import regression_abs_error

    targets = np.asarray(targets, dtype=np.float32)
    names = list(sampling_metrics.objective_names)
    smiles = frame["final_smiles"].tolist()
    parseable = np.asarray(
        [bool(value) and Chem.MolFromSmiles(str(value)) is not None for value in smiles],
        dtype=bool,
    )
    indices = np.where(parseable)[0]
    valid_smiles = [smiles[index] for index in indices]
    error_matrix = np.full((len(frame), len(names)), np.nan, dtype=np.float32)
    normalized_matrix = np.full_like(error_matrix, np.nan)
    for objective_index, name in enumerate(names):
        predictions = np.full(len(frame), np.nan, dtype=np.float32)
        predictions[indices] = predict_batched(
            sampling_metrics,
            name,
            valid_smiles,
            oracle_batch_size,
        )
        errors = regression_abs_error(name, targets[:, objective_index], predictions)
        std = float(sampling_metrics.normalization_stats[name]["std"])
        if std <= 0:
            raise ValueError(f"Normalization std for {name} must be positive.")
        normalized = errors / std
        frame[f"final_pred_{name}"] = predictions
        frame[f"final_abs_error_{name}"] = errors
        frame[f"final_normalized_abs_error_{name}"] = normalized
        error_matrix[:, objective_index] = errors
        normalized_matrix[:, objective_index] = normalized
    complete = np.isfinite(normalized_matrix).all(axis=1)
    sample_avg_mae = np.full(len(frame), np.nan, dtype=np.float32)
    sample_avg_nmae = np.full(len(frame), np.nan, dtype=np.float32)
    sample_avg_mae[complete] = error_matrix[complete].mean(axis=1)
    sample_avg_nmae[complete] = normalized_matrix[complete].mean(axis=1)
    frame["final_property_evaluable"] = complete
    frame["final_sample_avg_mae"] = sample_avg_mae
    frame["final_sample_avg_nmae"] = sample_avg_nmae
    return frame


def bool_mean(frame, column):
    return float(frame[column].astype(bool).mean()) if len(frame) else float("nan")


def summarize_frame(frame, names):
    summary = {
        "count": int(len(frame)),
        "validity": bool_mean(frame, "validity"),
        "validity_nc": bool_mean(frame, "validity_nc"),
        "validity_raw": bool_mean(frame, "validity_raw"),
        "final_property_coverage": bool_mean(frame, "final_property_evaluable"),
        "avg_mae": float(pd.to_numeric(frame["final_sample_avg_mae"], errors="coerce").mean()),
        "avg_nmae": float(pd.to_numeric(frame["final_sample_avg_nmae"], errors="coerce").mean()),
    }
    for name in names:
        summary[f"{name}/mae"] = float(
            pd.to_numeric(frame[f"final_abs_error_{name}"], errors="coerce").mean()
        )
        summary[f"{name}/nmae"] = float(
            pd.to_numeric(frame[f"final_normalized_abs_error_{name}"], errors="coerce").mean()
        )
    return summary


def inverse_cdf_sample(probabilities, seed):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = np.random.RandomState(int(seed) % (2**32)).uniform(size=len(probabilities))
    sampled = np.asarray(
        [np.searchsorted(cumulative[index], value, side="right") for index, value in enumerate(uniforms)],
        dtype=np.int64,
    )
    if (sampled <= 0).any():
        raise ValueError("Sampled motif count must be positive.")
    return sampled


def distribution_entropy(probability):
    probability = np.asarray(probability, dtype=np.float64)
    return float(-(probability * np.log(np.maximum(probability, 1.0e-30))).sum(axis=-1).mean())


def distribution_kl(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    values = first * (
        np.log(np.maximum(first, 1.0e-30)) - np.log(np.maximum(second, 1.0e-30))
    )
    return float(values.sum(axis=-1).mean())


def controller_base_probability(graph_model, max_nodes, device):
    probability = graph_model.node_dist.prob.detach().to(device=device, dtype=torch.float32)
    if probability.numel() > int(max_nodes) + 1:
        raise ValueError("Size controller does not cover model motif-count support.")
    if probability.numel() < int(max_nodes) + 1:
        probability = torch.nn.functional.pad(
            probability,
            (0, int(max_nodes) + 1 - probability.numel()),
        )
    return probability / probability.sum()


class TensorDataset(Dataset):
    def __init__(self, payload):
        self.payload = payload
        self.length = int(payload["x"].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.payload.items() if torch.is_tensor(value)}


class GroupBatchSampler(Sampler):
    def __init__(self, group_ids, batch_size, shuffle, seed):
        groups = {}
        for index, value in enumerate(group_ids.tolist()):
            groups.setdefault(int(value), []).append(index)
        self.groups = [groups[key] for key in sorted(groups)]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.groups and max(len(group) for group in self.groups) > self.batch_size:
            raise ValueError("Batch size must contain a complete target group.")

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        batch = []
        for group_index in order:
            group = self.groups[group_index]
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self):
        count = 0
        size = 0
        for group in self.groups:
            if size and size + len(group) > self.batch_size:
                count += 1
                size = 0
            size += len(group)
        return count + int(size > 0)


def build_loader(payload, batch_size, seed, shuffle=True):
    dataset = TensorDataset(payload)
    sampler = GroupBatchSampler(payload["group_id"], batch_size, shuffle, seed)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=False)


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
