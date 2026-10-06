import numpy as np
from task_registry import is_classification_task


LOG10_MAE_TARGETS = frozenset({"O2", "N2", "CO2"})


def uses_log10_mae(task_name) -> bool:
    return str(task_name).strip() in LOG10_MAE_TARGETS


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _prepare_mae_inputs(task_name, y_true, y_pred):
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must share the same shape, got {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
        )
    if not uses_log10_mae(task_name):
        return y_true, y_pred

    keep = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.0) & (y_pred > 0.0)
    y_true_metric = np.full(y_true.shape, np.nan, dtype=np.float32)
    y_pred_metric = np.full(y_pred.shape, np.nan, dtype=np.float32)
    y_true_metric[keep] = np.log10(y_true[keep]).astype(np.float32)
    y_pred_metric[keep] = np.log10(y_pred[keep]).astype(np.float32)
    return y_true_metric, y_pred_metric


def classification_abs_error(y_true, y_pred) -> np.ndarray:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must share the same shape, got {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
        )
    abs_error = np.abs(y_pred - y_true).astype(np.float32)
    invalid = ~np.isfinite(y_true) | ~np.isfinite(y_pred)
    abs_error[invalid] = np.nan
    return abs_error


def classification_accuracy(y_true, y_pred) -> float:
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must share the same shape, got {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
        )
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    if not keep.any():
        return float("nan")
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    return float(np.mean((y_pred >= 0.5) == (y_true >= 0.5)))


def regression_abs_error(task_name, y_true, y_pred) -> np.ndarray:
    y_true_metric, y_pred_metric = _prepare_mae_inputs(task_name, y_true, y_pred)
    abs_error = np.abs(y_pred_metric - y_true_metric).astype(np.float32)
    invalid = ~np.isfinite(y_true_metric) | ~np.isfinite(y_pred_metric)
    abs_error[invalid] = np.nan
    return abs_error


def regression_delta(task_name, y_true, y_pred) -> np.ndarray:
    y_true_metric, y_pred_metric = _prepare_mae_inputs(task_name, y_true, y_pred)
    delta = (y_pred_metric - y_true_metric).astype(np.float32)
    invalid = ~np.isfinite(y_true_metric) | ~np.isfinite(y_pred_metric)
    delta[invalid] = np.nan
    return delta


def regression_abs_error_by_task(task_names, y_true, y_pred) -> np.ndarray:
    task_names = np.asarray(task_names, dtype=object).reshape(-1)
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if task_names.shape[0] != y_true.shape[0] or y_true.shape != y_pred.shape:
        raise ValueError(
            "task_names, y_true, and y_pred must be aligned "
            f"(got {task_names.shape[0]}, {y_true.shape[0]}, {y_pred.shape[0]})"
        )

    abs_error = np.full(y_true.shape, np.nan, dtype=np.float32)
    for task_name in sorted(set(task_names.tolist())):
        task_mask = task_names == task_name
        abs_error[task_mask] = regression_abs_error(task_name, y_true[task_mask], y_pred[task_mask])
    return abs_error


def regression_mae(task_name, y_true, y_pred) -> float:
    abs_error = regression_abs_error(task_name, y_true, y_pred)
    if abs_error.size == 0 or np.isnan(abs_error).all():
        return float("nan")
    return float(np.nanmean(abs_error))


def regression_r2(task_name, y_true, y_pred) -> float:
    y_true_metric, y_pred_metric = _prepare_mae_inputs(task_name, y_true, y_pred)
    keep = np.isfinite(y_true_metric) & np.isfinite(y_pred_metric)
    if not keep.any():
        return float("nan")
    y_true_metric = y_true_metric[keep]
    y_pred_metric = y_pred_metric[keep]
    denom = float(np.sum((y_true_metric - float(np.mean(y_true_metric))) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_pred_metric - y_true_metric) ** 2) / denom)


def task_abs_error(task_name, y_true, y_pred) -> np.ndarray:
    if is_classification_task(task_name):
        return classification_abs_error(y_true, y_pred)
    return regression_abs_error(task_name, y_true, y_pred)


def task_abs_error_by_task(task_names, y_true, y_pred) -> np.ndarray:
    task_names = np.asarray(task_names, dtype=object).reshape(-1)
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if task_names.shape[0] != y_true.shape[0] or y_true.shape != y_pred.shape:
        raise ValueError(
            "task_names, y_true, and y_pred must be aligned "
            f"(got {task_names.shape[0]}, {y_true.shape[0]}, {y_pred.shape[0]})"
        )

    abs_error = np.full(y_true.shape, np.nan, dtype=np.float32)
    for task_name in sorted(set(task_names.tolist())):
        task_mask = task_names == task_name
        abs_error[task_mask] = task_abs_error(task_name, y_true[task_mask], y_pred[task_mask])
    return abs_error


def task_mae(task_name, y_true, y_pred) -> float:
    abs_error = task_abs_error(task_name, y_true, y_pred)
    if abs_error.size == 0 or np.isnan(abs_error).all():
        return float("nan")
    return float(np.nanmean(abs_error))
