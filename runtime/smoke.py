"""Lightweight configuration checks shared by PT and SFT."""


def run_pt_smoke_checks(cfg, dataset_infos):
    if int(cfg.model.diffusion_steps) < 1:
        raise ValueError("model.diffusion_steps must be positive.")
    if int(dataset_infos.max_n_nodes) < 1:
        raise ValueError("The tokenizer/data pair contains no nonempty graphs.")
