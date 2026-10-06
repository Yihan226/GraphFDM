import csv
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from fcd_torch import FCD as FCDMetric
from mini_moses.metrics.metrics import FragMetric, compute_intermediate_statistics, internal_diversity
from rdkit import Chem

from metrics.metric_utils import classification_accuracy, regression_mae, regression_r2, task_mae
from metrics.property_metric import TaskModel, calculateSAS
from motif_utils import decode_motif_graphs, enrich_decoded_graphs_with_validity
from runtime.cache import clear_caches, get_or_create
from runtime.common import resolve_run_artifacts
from runtime.train_methods import is_rl_method
from shared import SASCORE_OBJECTIVE
from task_registry import is_classification_task


def _format_csv_scalar(value):
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        if np.isinf(value):
            return str(float(value))
        return f"{float(value):.4f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _is_sascore_objective(name):
    return str(name).strip().casefold() in {SASCORE_OBJECTIVE.casefold(), "sascore", "sa", "sas"}


def _safe_is_classification_task(name):
    if _is_sascore_objective(name):
        return False
    try:
        return is_classification_task(name)
    except KeyError:
        return False


def result_to_csv(path, dict_data):
    file_exists = os.path.exists(path)
    log_name = dict_data.pop("log_name", None)
    if log_name is None:
        raise ValueError("The provided dictionary must contain a 'log_name' key.")
    field_names = ["log_name"] + list(dict_data.keys())
    dict_data = {key: _format_csv_scalar(value) for key, value in dict_data.items()}
    dict_data["log_name"] = log_name
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict_data)


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, train_smiles, reference_smiles, n_jobs=1, device="cpu", batch_size=512):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.train_smiles = train_smiles
        self.n_jobs = n_jobs
        self.device_name = device
        self.batch_size = batch_size
        self.decode_cache = {}
        self.mol_cache = {}
        self.eval_cache = {}
        train_method = getattr(dataset_infos.cfg.general, "train_method", "pt_unconditional")
        self.objective_names = list(getattr(dataset_infos, "objective_names", []))
        self.normalization_stats = dict(getattr(dataset_infos, "normalization_stats", {}))
        self.objective_weights = np.asarray(
            getattr(dataset_infos, "objective_weights", [1.0] * max(1, len(self.objective_names))),
            dtype=np.float32,
        )
        if len(self.objective_names) > 0:
            self.objective_weights = self.objective_weights / max(float(self.objective_weights.sum()), 1e-12)
        self.is_unconditional_orig = not is_rl_method(train_method) and len(self.objective_names) == 0

        if reference_smiles is not None and len(reference_smiles) > 0:
            print(f"--- Computing intermediate statistics for training for #{len(reference_smiles)} smiles ---")
            start_time = time.time()
            self.stat_ref = compute_intermediate_statistics(
                reference_smiles,
                n_jobs=n_jobs,
                device=device,
                batch_size=batch_size,
            )
            print(f"--- End computing intermediate statistics: using {time.time() - start_time:.2f}s ---")
        else:
            self.stat_ref = None

        self.task_evaluator = {}
        if not self.is_unconditional_orig:
            for objective_name in self.objective_names:
                if not objective_name or _is_sascore_objective(objective_name):
                    continue
                model_path = os.path.join(dataset_infos.evaluator_dir, f"{objective_name}.joblib")
                self.task_evaluator[objective_name] = TaskModel(model_path, objective_name, dataset_infos.data_root)

    def _cached_mol(self, smiles):
        return get_or_create(self.mol_cache, smiles, lambda: Chem.MolFromSmiles(smiles))

    def _cached_eval(self, name, smiles_list, compute_fn):
        if len(smiles_list) == 0:
            return np.asarray([], dtype=np.float32)
        key = (name, tuple(smiles_list))
        return get_or_create(self.eval_cache, key, lambda: np.asarray(compute_fn(smiles_list), dtype=np.float32))

    def _compute_distribution_metrics(self, valid):
        metrics = {}
        if len(valid) == 0:
            return metrics
        close_pool = False
        if self.n_jobs != 1:
            pool = Pool(self.n_jobs)
            close_pool = True
        else:
            pool = 1
        valid_mols = [self._cached_mol(smiles) for smiles in valid]
        metrics["interval_diversity"] = internal_diversity(valid_mols, pool, device=self.device_name)
        if self.stat_ref is not None:
            kwargs = {"n_jobs": pool, "device": self.device_name, "batch_size": self.batch_size}
            kwargs_fcd = {"n_jobs": self.n_jobs, "device": self.device_name, "batch_size": self.batch_size}
            metrics["sim/Frag"] = FragMetric(**kwargs)(gen=valid_mols, pref=self.stat_ref["Frag"])
            metrics["dist/FCD"] = FCDMetric(**kwargs_fcd)(gen=valid, pref=self.stat_ref["FCD"])
        if close_pool:
            pool.close()
            pool.join()
        return metrics

    def _compute_avg_sascore(self, smiles_list):
        valid_smiles = []
        for smiles in smiles_list:
            if not smiles:
                continue
            if self._cached_mol(smiles) is None:
                continue
            valid_smiles.append(smiles)
        if len(valid_smiles) == 0:
            return float("nan")
        scores = np.asarray(calculateSAS(valid_smiles), dtype=np.float32)
        if scores.size == 0:
            return float("nan")
        return float(np.nanmean(scores))

    def _predict_objective(self, objective_name, smiles_list):
        if len(smiles_list) == 0:
            return np.asarray([], dtype=np.float32)
        if _is_sascore_objective(objective_name):
            return self._cached_eval(SASCORE_OBJECTIVE, smiles_list, calculateSAS)
        if objective_name not in self.task_evaluator:
            return np.full(len(smiles_list), np.nan, dtype=np.float32)
        return self._cached_eval(objective_name, smiles_list, self.task_evaluator[objective_name])

    def _normalize_targets(self, targets_np, n_rows):
        targets_np = np.asarray(targets_np, dtype=np.float32)
        if targets_np.ndim == 1 and len(self.objective_names) == 1:
            targets_np = targets_np[:, None]
        if len(self.objective_names) == 0:
            return targets_np.reshape(n_rows, -1) if targets_np.size > 0 else np.zeros((n_rows, 0), dtype=np.float32)
        if targets_np.ndim != 2 or targets_np.shape != (n_rows, len(self.objective_names)):
            raise ValueError(
                f"Sampling metrics expected targets with shape [{n_rows}, {len(self.objective_names)}], "
                f"got {tuple(targets_np.shape)}."
            )
        return targets_np

    def score_relaxed_smiles(self, all_smiles, targets_np, test=False):
        targets_np = self._normalize_targets(targets_np, len(all_smiles))
        valid_index = np.asarray([smiles is not None for smiles in all_smiles], dtype=bool)
        valid_smiles = [all_smiles[i] for i in range(len(all_smiles)) if valid_index[i]]
        valid_global_idx = np.where(valid_index)[0]
        targets_log = {
            "objective_names": self.objective_names,
            "input_values": targets_np,
            "output_values": np.full((len(all_smiles), len(self.objective_names)), np.nan, dtype=np.float32),
        }

        if len(self.objective_names) == 0:
            return {
                "metrics": {},
                "targets_log": targets_log,
                "valid_relaxed_mask": valid_index,
            }

        metrics = {}
        mae_values = []
        nmae_values = []
        r2_values = []
        for obj_idx, objective_name in enumerate(self.objective_names):
            preds_valid = np.asarray(self._predict_objective(objective_name, valid_smiles), dtype=np.float32)
            if preds_valid.shape[0] != len(valid_smiles):
                raise ValueError(f"Evaluator for {objective_name!r} returned {preds_valid.shape[0]} rows for {len(valid_smiles)} smiles.")
            targets_log["output_values"][valid_global_idx, obj_idx] = preds_valid
            y_true = targets_np[valid_global_idx, obj_idx].astype(np.float32)
            y_pred = preds_valid.astype(np.float32)
            if len(y_true) == 0:
                metrics[f"{objective_name}/mae"] = float("nan")
                continue
            if _safe_is_classification_task(objective_name):
                cur_mae = task_mae(objective_name, y_true, y_pred)
                metrics[f"{objective_name}/mae"] = cur_mae
                metrics[f"{objective_name}/acc"] = classification_accuracy(y_true, y_pred)
            else:
                cur_mae = regression_mae(objective_name, y_true, y_pred)
                cur_r2 = regression_r2(objective_name, y_true, y_pred)
                metrics[f"{objective_name}/mae"] = cur_mae
                metrics[f"{objective_name}/r2"] = cur_r2
                objective_stats = self.normalization_stats.get(objective_name, {})
                objective_std = float(objective_stats.get("std", 0.0))
                if objective_std > 0:
                    cur_nmae = float(np.nanmean(np.abs(y_pred - y_true) / objective_std))
                    metrics[f"{objective_name}/nmae"] = cur_nmae
                    if np.isfinite(cur_nmae):
                        nmae_values.append(cur_nmae)
                if np.isfinite(cur_r2):
                    r2_values.append(cur_r2)
            if np.isfinite(cur_mae):
                mae_values.append(cur_mae)

        metrics["avg_mae"] = float(np.mean(mae_values)) if len(mae_values) > 0 else float("nan")
        if len(nmae_values) > 0:
            metrics["avg_nmae"] = float(np.mean(nmae_values))
        if test or len(r2_values) > 0:
            metrics["avg_r2"] = float(np.mean(r2_values)) if len(r2_values) > 0 else float("nan")
        return {
            "metrics": metrics,
            "targets_log": targets_log,
            "valid_relaxed_mask": valid_index,
        }

    def forward(self, molecules, targets, name, current_epoch, val_counter, test=False):
        if isinstance(targets, list):
            targets_np = torch.cat(targets, dim=0).detach().cpu().numpy()
        else:
            targets_np = targets.detach().cpu().numpy()

        artifacts = resolve_run_artifacts(name, create=not test)
        self.eval_cache.clear()
        decoded_base = decode_motif_graphs(self.dataset_infos.tokenizer, molecules, cache=self.decode_cache)
        decoded = enrich_decoded_graphs_with_validity(decoded_base, ensure_connected=getattr(self.dataset_infos, "ensure_connected", True))
        all_smiles = [item["smiles"] for item in decoded]
        valid_smiles = [smiles for smiles in all_smiles if smiles is not None]
        unique_smiles = list(dict.fromkeys(valid_smiles))
        validity_nc = float(sum(1 for item in decoded if item["validity_nc"]) / max(len(decoded), 1))
        validity_raw = float(sum(1 for item in decoded if item.get("validity_raw", False)) / max(len(decoded), 1))

        metrics = {
            "validity": float(len(valid_smiles) / max(len(all_smiles), 1)),
            "validity_nc": validity_nc,
            "validity_raw": validity_raw,
        }
        metrics.update(self._compute_distribution_metrics(valid_smiles))
        if test:
            metrics[f"avg_{SASCORE_OBJECTIVE}"] = self._compute_avg_sascore(valid_smiles)
        prop_result = self.score_relaxed_smiles(all_smiles, targets_np, test=test)
        metrics.update(prop_result["metrics"])
        targets_log = prop_result["targets_log"]

        if test:
            with open(artifacts["final_smiles"], "w") as fp:
                input_headers = [f"input_{name}" for name in self.objective_names]
                output_headers = [f"output_{name}" for name in self.objective_names]
                fp.write(",".join(["smiles"] + input_headers + output_headers) + "\n")
                input_values = targets_log["input_values"]
                output_values = targets_log["output_values"]
                for i, smiles in enumerate(all_smiles):
                    row = [smiles]
                    row.extend(_format_csv_scalar(value) for value in input_values[i])
                    row.extend(_format_csv_scalar(value) for value in output_values[i])
                    fp.write(",".join(str(value) for value in row) + "\n")
        else:
            os.makedirs(artifacts["graphs_root"], exist_ok=True)
            text_path = os.path.join(artifacts["graphs_root"], f"valid_unique_molecules_e{current_epoch}_b{val_counter}.txt")
            with open(text_path, "w") as textfile:
                for smiles in unique_smiles:
                    textfile.write(smiles + "\n")

        all_logs = dict(metrics)
        all_logs["log_name"] = "test" if test else f"epoch{current_epoch}_batch{val_counter}"
        result_to_csv(artifacts["output_csv"], all_logs.copy())
        self.last_metrics = metrics
        self.last_log_name = all_logs["log_name"]
        self.last_targets_log = targets_log
        self.last_all_smiles = all_smiles
        self.last_targets = targets_np
        return all_smiles

    def reset(self):
        clear_caches(self.decode_cache, self.mol_cache, self.eval_cache)
