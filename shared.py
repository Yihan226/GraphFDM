import json
import os

import torch
from omegaconf import OmegaConf


SASCORE_OBJECTIVE = "SA"


def get_objectives_config(cfg):
    objectives_cfg = getattr(cfg, "objectives", None)
    if objectives_cfg is None:
        objectives_cfg = OmegaConf.create({})
    return objectives_cfg


def get_objective_tasks(cfg):
    objectives_cfg = get_objectives_config(cfg)
    tasks = getattr(objectives_cfg, "tasks", [])
    if tasks is None:
        return []
    return [str(name).strip() for name in tasks if str(name).strip()]


def include_sascore_objective(cfg):
    objectives_cfg = get_objectives_config(cfg)
    return bool(getattr(objectives_cfg, "include_sascore", True))


def get_objective_names(cfg):
    names = []
    if include_sascore_objective(cfg):
        names.append(SASCORE_OBJECTIVE)
    names.extend(get_objective_tasks(cfg))
    return names


def get_objective_stem(cfg):
    objectives_cfg = get_objectives_config(cfg)
    source = getattr(objectives_cfg, "source", None)
    if source is not None and str(source).strip():
        return str(source).strip()
    tasks = get_objective_tasks(cfg)
    if len(tasks) == 0:
        return str(getattr(cfg.dataset, "pretrain_data_source", "pretrain")).strip()
    return "_".join(tasks)


def get_objective_weights(cfg):
    names = get_objective_names(cfg)
    objectives_cfg = get_objectives_config(cfg)
    raw_weights = getattr(objectives_cfg, "weights", None)
    if raw_weights is None:
        weights = [1.0] * len(names)
    else:
        weights = [float(value) for value in raw_weights]
    if len(weights) != len(names):
        raise ValueError(
            "objectives.weights must have the same length as [SA] + objectives.tasks "
            f"(got {len(weights)} weights for {len(names)} objectives)."
        )
    total = sum(weights)
    if total <= 0:
        raise ValueError("objectives.weights must sum to a positive value.")
    return [value / total for value in weights]


def get_value_normalization_mode(cfg):
    objectives_cfg = get_objectives_config(cfg)
    value = getattr(objectives_cfg, "value_normalization", None)
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


def get_normalization_clip(cfg):
    objectives_cfg = get_objectives_config(cfg)
    clip_value = getattr(objectives_cfg, "normalization_clip", None)
    if clip_value is None:
        return None
    return float(clip_value)


def get_normalization_stats_path(cfg):
    objectives_cfg = get_objectives_config(cfg)
    path = getattr(objectives_cfg, "normalization_stats_path", None)
    if path is None:
        return None
    path = str(path).strip()
    return path or None


def load_normalization_stats(path):
    if path is None:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return {str(name): dict(values) for name, values in stats.items()}


def save_normalization_stats(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def get_normalization_stats(cfg):
    objectives_cfg = get_objectives_config(cfg)
    stats = getattr(objectives_cfg, "normalization_stats", None)
    if stats is not None:
        if hasattr(stats, "items"):
            out = {str(name): dict(values) for name, values in stats.items()}
        else:
            out = dict(stats)
    else:
        out = load_normalization_stats(get_normalization_stats_path(cfg))
    if SASCORE_OBJECTIVE not in out and "sascore" in out:
        out[SASCORE_OBJECTIVE] = dict(out["sascore"])
    return out


def normalize_objective_value(raw_value, objective_name, cfg=None, stats=None, eps=1e-8):
    raw_value = float(raw_value)
    if cfg is None:
        mode = None
        clip_value = None
    else:
        mode = get_value_normalization_mode(cfg)
        clip_value = get_normalization_clip(cfg)
    if mode is None or mode == "none":
        return raw_value
    if mode not in {"per_column_zscore", "per_target_zscore"}:
        raise ValueError(f"Unsupported objectives.value_normalization: {mode!r}")
    if stats is None:
        stats = get_normalization_stats(cfg)
    if objective_name not in stats:
        raise ValueError(f"Missing normalization stats for objective {objective_name!r}.")
    objective_stats = stats[objective_name]
    mean = float(objective_stats["mean"])
    std = float(objective_stats["std"])
    y_norm = (raw_value - mean) / (std + eps)
    if clip_value is not None:
        y_norm = max(-clip_value, min(clip_value, y_norm))
    return float(y_norm)


def flatten_optional_y(y):
    if y.dim() == 1:
        return y.unsqueeze(1)
    if y.dim() > 2:
        return y.reshape(y.shape[0], -1)
    return y


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def safe_tensor_mean(x, device):
    if x.numel() == 0:
        return torch.tensor(0.0, device=device)
    return x.mean()
