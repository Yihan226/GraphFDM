# These imports are tricky because they use c++, do not move them
import os
import warnings

import hydra
import torch
from omegaconf import DictConfig

from analysis.visualization import MolecularVisualization
from diffusion_model import Graph_DiT
from metrics.molecular_metrics_sampling import SamplingMolecularMetrics
from metrics.molecular_metrics_train import TrainMolecularMetricsDiscrete
from runtime.common import (
    build_checkpoint_callback,
    build_model,
    build_trainer,
    finish_wandb,
    get_resume,
    get_resume_adaptive,
    seed_everything_if_needed,
    setup_wandb,
    sync_model_runtime_config,
    validate_config,
)
from runtime.train_methods import METHOD_PT_UNCONDITIONAL, ensure_method_config
from runtime.smoke import run_pt_smoke_checks

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")


def initialize_resume_best_threshold(trainer, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    states = [
        state
        for state in payload.get("callbacks", {}).values()
        if isinstance(state, dict) and state.get("best_model_score") is not None
    ]
    if len(states) != 1:
        raise ValueError("Resume checkpoint must contain exactly one monitored best score.")
    callback = trainer.checkpoint_callback
    score = states[0]["best_model_score"].detach().cpu()
    callback.best_model_score = score
    callback.current_score = score
    callback.best_model_path = checkpoint_path
    callback.kth_best_model_path = checkpoint_path
    callback.kth_value = score
    callback.best_k_models = {checkpoint_path: score}


@hydra.main(version_base="1.1", config_path="configs", config_name="pt_molecule")
def main(cfg: DictConfig):
    validate_config(cfg)
    ensure_method_config(cfg)
    seed_everything_if_needed(cfg)

    if cfg.general.train_method == METHOD_PT_UNCONDITIONAL:
        from datasets import unconditional_dataset as dataset_mod
    else:
        from datasets import conditional_dataset as dataset_mod

    datamodule = dataset_mod.DataModule(cfg)
    datamodule.prepare_data()
    dataset_infos = dataset_mod.DataInfos(datamodule=datamodule, cfg=cfg)
    sampling_enabled = bool(getattr(cfg.general, "validation_sampling", True)) or bool(
        getattr(cfg.general, "test_after_fit", True)
    )
    if sampling_enabled:
        train_smiles, reference_smiles = datamodule.get_train_smiles()
    else:
        train_smiles, reference_smiles = [], []
    dataset_infos.compute_input_output_dims(datamodule=datamodule)
    run_pt_smoke_checks(cfg, dataset_infos)

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles, reference_smiles)
    visualization_tools = MolecularVisualization(dataset_infos)
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
    }
    resume_model = None
    fit_checkpoint = None

    if cfg.general.test_only:
        cfg, resume_model = get_resume(Graph_DiT, cfg, model_kwargs)
        resume_model = sync_model_runtime_config(resume_model, cfg)
        os.chdir(cfg.general.test_only.rsplit(f"{os.sep}checkpoints{os.sep}", 1)[0])
    elif cfg.general.resume is not None:
        cfg, _ = get_resume_adaptive(Graph_DiT, cfg, model_kwargs)
        fit_checkpoint = cfg.general.resume
        if bool(getattr(cfg.general, "resume_in_place", True)):
            os.chdir(cfg.general.resume.rsplit(f"{os.sep}checkpoints{os.sep}", 1)[0])

    train_method = cfg.general.train_method
    wandb_project = str(getattr(cfg.general, "wandb_project", "gdit-pt-multiobjective"))
    logger = setup_wandb(cfg, wandb_project, cfg.general.name, define_train_metrics=False)
    trainer = build_trainer(cfg, logger, True, [build_checkpoint_callback(cfg)])
    require_resume_improvement = bool(
        getattr(cfg.general, "checkpoint_require_resume_improvement", False)
    )
    if fit_checkpoint and require_resume_improvement:
        initialize_resume_best_threshold(trainer, fit_checkpoint)

    try:
        if not cfg.general.test_only:
            model = build_model(Graph_DiT, cfg, model_kwargs, train_method)
            trainer.fit(
                model,
                datamodule=datamodule,
                ckpt_path=fit_checkpoint,
                weights_only=False if fit_checkpoint else None,
            )
            if bool(getattr(cfg.general, "test_after_fit", True)):
                test_checkpoint = "best" if bool(getattr(cfg.general, "test_best_after_fit", False)) else None
                test_only_if_new_best = bool(getattr(cfg.general, "test_only_if_new_best", False))
                new_best = not fit_checkpoint or os.path.realpath(
                    trainer.checkpoint_callback.best_model_path
                ) != os.path.realpath(fit_checkpoint)
                if not test_only_if_new_best or new_best:
                    trainer.test(
                        model,
                        datamodule=datamodule,
                        ckpt_path=test_checkpoint,
                        weights_only=False if test_checkpoint else None,
                    )
                else:
                    print("No validation improvement over the resume checkpoint; skipping test5000.")
        else:
            model = resume_model if resume_model is not None else Graph_DiT(cfg=cfg, **model_kwargs)
            trainer.test(model, datamodule=datamodule, ckpt_path=None)
    finally:
        finish_wandb()


if __name__ == "__main__":
    main()
