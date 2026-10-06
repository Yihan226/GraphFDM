"""Training modes required by the GraphFDM reproduction package."""

from omegaconf import OmegaConf


METHOD_PT_UNCONDITIONAL = "pt_unconditional"
METHOD_PT_CONDITIONAL = "pt_conditional"
SUPPORTED_METHODS = {METHOD_PT_UNCONDITIONAL, METHOD_PT_CONDITIONAL}


def normalize_train_method(train_method):
    value = METHOD_PT_UNCONDITIONAL if train_method is None else str(train_method).strip().lower()
    return value or METHOD_PT_UNCONDITIONAL


def is_rl_method(train_method):
    del train_method
    return False


def ensure_method_config(cfg):
    if OmegaConf.is_config(cfg):
        OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg.general, "train_method") or cfg.general.train_method is None:
        cfg.general.train_method = METHOD_PT_UNCONDITIONAL
    cfg.general.train_method = normalize_train_method(cfg.general.train_method)
    if cfg.general.train_method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(
            f"Unsupported train_method {cfg.general.train_method!r}; expected one of {supported}."
        )
    return cfg


def get_method_name(cfg):
    return normalize_train_method(getattr(cfg.general, "train_method", METHOD_PT_UNCONDITIONAL))
