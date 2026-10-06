import copy
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from runtime.train_methods import (
    METHOD_PT_CONDITIONAL,
    METHOD_PT_UNCONDITIONAL,
    ensure_method_config,
)
from shared import (
    get_normalization_stats,
    get_normalization_stats_path,
    get_objective_names,
    get_objective_tasks,
    get_objective_weights,
)


def update_config_with_new_keys(cfg, saved_cfg):
    cfg_container = OmegaConf.to_container(cfg, resolve=False) if not isinstance(cfg, dict) else copy.deepcopy(cfg)
    saved_container = OmegaConf.to_container(saved_cfg, resolve=False) if not isinstance(saved_cfg, dict) else copy.deepcopy(saved_cfg)

    def merge(dst, src):
        for key, value in src.items():
            if key not in dst:
                dst[key] = value
                continue
            if isinstance(dst[key], dict) and isinstance(value, dict):
                merge(dst[key], value)

    merge(cfg_container, saved_container)
    return OmegaConf.create(cfg_container)


def overlay_config_values(base_cfg, override_cfg):
    base_container = OmegaConf.to_container(base_cfg, resolve=False) if not isinstance(base_cfg, dict) else copy.deepcopy(base_cfg)
    override_container = (
        OmegaConf.to_container(override_cfg, resolve=False) if not isinstance(override_cfg, dict) else copy.deepcopy(override_cfg)
    )

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict):
                existing = dst.get(key)
                if not isinstance(existing, dict):
                    dst[key] = copy.deepcopy(value)
                else:
                    merge(existing, value)
            else:
                dst[key] = value

    merge(base_container, override_container)
    return OmegaConf.create(base_container)


def validate_config(cfg):
    ensure_method_config(cfg)
    train_method = cfg.general.train_method
    data_family = str(getattr(cfg.dataset, "data", "")).strip().lower()
    pt_checkpoint = getattr(cfg.general, "pt_checkpoint", None)
    if data_family not in {"polymer", "molecule"}:
        raise ValueError("cfg.dataset.data must be 'polymer' or 'molecule'.")
    if cfg.general.resume is not None and pt_checkpoint is not None:
        raise ValueError("general.resume and general.pt_checkpoint cannot both be set.")
    if pt_checkpoint is not None and train_method != METHOD_PT_CONDITIONAL:
        raise ValueError("general.pt_checkpoint is only supported for pt_conditional.")
    if train_method != METHOD_PT_UNCONDITIONAL:
        if len(get_objective_tasks(cfg)) == 0:
            raise ValueError("multiobjective training requires objectives.tasks.")
        objective_names = get_objective_names(cfg)
        get_objective_weights(cfg)
        stats_path = get_normalization_stats_path(cfg)
        if not stats_path:
            raise ValueError("multiobjective training requires objectives.normalization_stats_path.")
        if os.path.exists(stats_path):
            normalization_stats = get_normalization_stats(cfg)
            for objective_name in objective_names:
                if objective_name not in normalization_stats:
                    raise ValueError(f"Missing normalization stats for objective {objective_name!r}.")
                objective_stats = normalization_stats[objective_name]
                if "mean" not in objective_stats or "std" not in objective_stats:
                    raise ValueError(f"Normalization stats for objective {objective_name!r} must contain mean and std.")
                if float(objective_stats["std"]) <= 0:
                    raise ValueError(f"Normalization std for objective {objective_name!r} must be positive.")
    if getattr(cfg.general, "wandb", "disabled") != "disabled" and "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY is required when wandb logging is enabled.")


def seed_everything_if_needed(cfg):
    seed = getattr(cfg.train, "seed", None)
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def resolve_run_artifacts(run_name, create=False):
    root = os.getcwd()
    paths = {
        "run_root": root,
        "graphs_root": os.path.join(root, "graphs", run_name),
        "checkpoint_root": os.path.join(root, "checkpoints", run_name),
        "output_csv": os.path.join(root, "output.csv"),
        "final_smiles": os.path.join(root, "final_smiles.txt"),
    }
    if create:
        os.makedirs(paths["graphs_root"], exist_ok=True)
        os.makedirs(paths["checkpoint_root"], exist_ok=True)
    return paths


def get_resume(model_cls, cfg, model_kwargs):
    saved_cfg = cfg.copy()
    name = cfg.general.name + "_resume"
    resume = cfg.general.test_only
    model = model_cls.load_from_checkpoint(resume, weights_only=False, **model_kwargs)
    new_cfg = model.cfg
    new_cfg = overlay_config_values(new_cfg, saved_cfg)
    new_cfg.general.test_only = resume
    new_cfg.general.name = name
    new_cfg = update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def get_resume_adaptive(model_cls, cfg, model_kwargs):
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split("outputs")[0]
    resume_path = os.path.join(root_dir, cfg.general.resume)
    if cfg.model.type != "discrete":
        raise NotImplementedError("Unknown model")
    model = model_cls.load_from_checkpoint(resume_path, weights_only=False, **model_kwargs)
    new_cfg = model.cfg
    new_cfg = overlay_config_values(new_cfg, cfg)
    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + "_resume"
    new_cfg = update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_wandb(cfg, project_name, run_name, define_train_metrics=False):
    mode = getattr(cfg.general, "wandb", "disabled")
    if mode == "disabled":
        return []
    wandb.init(
        project=project_name,
        name=run_name,
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.define_metric("trainer/global_step")
    if define_train_metrics:
        wandb.define_metric("train/ce_loss", step_metric="trainer/global_step")
        wandb.define_metric("train/kl_loss", step_metric="trainer/global_step")
    return WandbLogger(experiment=wandb.run, log_model=False)


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()


def sync_model_runtime_config(model, cfg):
    model.cfg = cfg
    if hasattr(model, "name"):
        model.name = cfg.general.name
    if hasattr(model, "test_only"):
        model.test_only = cfg.general.test_only
    if hasattr(model, "guide_scale"):
        model.guide_scale = cfg.model.guide_scale
    if hasattr(model, "train_method"):
        model.train_method = getattr(cfg.general, "train_method", model.train_method)
    if hasattr(model, "log_every_steps") and hasattr(cfg.general, "log_every_steps"):
        model.log_every_steps = cfg.general.log_every_steps
    if hasattr(model, "number_chain_steps") and hasattr(cfg.general, "number_chain_steps"):
        model.number_chain_steps = cfg.general.number_chain_steps
    if hasattr(model, "save_chain_strategy"):
        model.save_chain_strategy = str(getattr(cfg.general, "save_chain_strategy", model.save_chain_strategy)).strip().lower()
    return model


def build_model(model_cls, cfg, model_kwargs, train_method):
    ensure_method_config(cfg)
    pt_ckpt = getattr(cfg.general, "pt_checkpoint", None)
    if cfg.general.resume is not None:
        current_path = os.path.dirname(os.path.realpath(__file__))
        root_dir = current_path.split("outputs")[0]
        resume_path = os.path.join(root_dir, cfg.general.resume)
        print(f"Resuming from checkpoint: {resume_path}")
        model = model_cls.load_from_checkpoint(resume_path, cfg=cfg, weights_only=False, **model_kwargs)
    elif train_method == METHOD_PT_CONDITIONAL and pt_ckpt:
        print(f"PT conditional warm start from checkpoint: {pt_ckpt}")
        model = model_cls(cfg=cfg, **model_kwargs)
        try:
            checkpoint = torch.load(pt_ckpt, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Failed to load checkpoint {pt_ckpt!r}. "
                "This usually means the checkpoint was saved before the dataset refactor and still "
                "references a removed module. Make sure the compatibility shim for old checkpoints is present."
            ) from exc
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        model = model_cls(cfg=cfg, **model_kwargs)
    return model


def build_trainer(cfg, logger, enable_checkpointing, callbacks):
    gradient_clip_val = cfg.train.clip_grad
    use_gpu = torch.cuda.is_available() and cfg.general.gpus > 0
    trainer_kwargs = {
        "gradient_clip_val": gradient_clip_val,
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": cfg.general.gpus if use_gpu else 1,
        "max_epochs": cfg.train.n_epochs,
        "enable_checkpointing": enable_checkpointing,
        "check_val_every_n_epoch": cfg.train.check_val_every_n_epoch,
        "val_check_interval": cfg.train.val_check_interval,
        "strategy": "ddp" if cfg.general.gpus > 1 else "auto",
        "enable_progress_bar": cfg.general.enable_progress_bar,
        "callbacks": callbacks,
        "reload_dataloaders_every_n_epochs": 0,
        "logger": logger,
    }
    for key in ["fast_dev_run", "limit_train_batches", "limit_val_batches", "limit_test_batches"]:
        value = getattr(cfg.general, key, None)
        if value is not None:
            trainer_kwargs[key] = value
    return Trainer(**trainer_kwargs)


def _infer_checkpoint_mode(monitor_name, explicit_mode=None):
    if explicit_mode is not None:
        mode = str(explicit_mode).strip().lower()
        if mode not in {"min", "max"}:
            raise ValueError("general.checkpoint_mode must be 'min' or 'max'.")
        return mode

    metric_name = str(monitor_name).strip().lower()
    metric_leaf = metric_name.split("/")[-1]
    if metric_leaf in {"acc", "r2", "validity", "validity_nc", "validity_raw"}:
        return "max"
    return "min"


def _resolve_checkpoint_monitor(cfg):
    monitor = getattr(cfg.general, "checkpoint_monitor", None)
    if monitor is None:
        return None, None

    monitor = str(monitor).strip()
    if not monitor:
        return None, None

    train_method = cfg.general.train_method
    if train_method == METHOD_PT_CONDITIONAL:
        monitor_lower = monitor.lower()
        if monitor_lower in {"mae", "r2"}:
            if monitor_lower == "mae":
                monitor = "val/avg_mae"
            else:
                monitor = "val/avg_r2"
        elif "/" in monitor and not monitor.startswith(("val/", "test/")):
            monitor = f"val/{monitor}"
        elif monitor_lower in {"avg_mae", "avg_r2"}:
            monitor = f"val/{monitor_lower}"

    mode = _infer_checkpoint_mode(monitor, getattr(cfg.general, "checkpoint_mode", None))
    return monitor, mode


def _resolve_checkpoint_every_n_epochs(cfg, monitor_name):
    every_n_epochs = getattr(cfg.general, "checkpoint_every_n_epochs", None)
    if every_n_epochs is not None:
        return int(every_n_epochs)

    if monitor_name is not None:
        check_val_every_n_epoch = getattr(cfg.train, "check_val_every_n_epoch", None)
        if check_val_every_n_epoch is not None:
            return max(1, int(check_val_every_n_epoch))
        return 1

    return 100


def build_checkpoint_callback(cfg):
    train_method = cfg.general.train_method
    if train_method in {METHOD_PT_UNCONDITIONAL, METHOD_PT_CONDITIONAL}:
        monitor, mode = _resolve_checkpoint_monitor(cfg)
        checkpoint_kwargs = {
            "dirpath": os.path.join("checkpoints", cfg.general.name),
            "filename": str(getattr(cfg.general, "checkpoint_filename", "{epoch}")),
            "save_last": bool(getattr(cfg.general, "checkpoint_save_last", True)),
            "save_top_k": int(getattr(cfg.general, "checkpoint_save_top_k", 1)),
            "every_n_epochs": _resolve_checkpoint_every_n_epochs(cfg, monitor),
            "monitor": monitor,
        }
        if monitor is not None:
            checkpoint_kwargs["mode"] = mode
        return ModelCheckpoint(
            **checkpoint_kwargs,
        )
    else:
        raise ValueError(f"please set up monitor for saving checkpoint: {train_method}")
