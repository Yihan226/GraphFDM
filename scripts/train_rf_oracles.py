import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


PROPERTY_NAMES = ["mu", "alpha", "homo", "lumo", "cv"]


def main():
    experiment_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(experiment_root))

    from metrics.property_metric import TaskModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=experiment_root / "data")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--evaluation_dataset", type=Path)
    parser.add_argument("--normalization_stats", type=Path)
    parser.add_argument("--evaluator_root", type=Path)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    dataset_root = args.data_root / "molecule"
    dataset_path = args.dataset or dataset_root / "raw" / "rqm9_v1_clean.csv.gz"
    evaluator_root = args.evaluator_root or args.data_root / "evaluator"
    evaluator_root.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    frame = pd.read_csv(dataset_path)
    fingerprints = np.empty((len(frame), 2048), dtype=np.uint8)
    print(f"Building ECFP4 fingerprints for {len(frame)} molecules", flush=True)
    for index, smiles in enumerate(frame["smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Clean rQM9 SMILES failed to parse at row {index}")
        fingerprints[index] = TaskModel.fingerprints_from_mol(mol)[0]

    evaluation = pd.read_csv(args.evaluation_dataset) if args.evaluation_dataset else None
    evaluation_fingerprints = None
    if evaluation is not None:
        evaluation_fingerprints = np.empty((len(evaluation), 2048), dtype=np.uint8)
        print(f"Building held-out ECFP4 fingerprints for {len(evaluation)} molecules", flush=True)
        for index, smiles in enumerate(evaluation["smiles"]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Held-out rQM9 SMILES failed to parse at row {index}")
            evaluation_fingerprints[index] = TaskModel.fingerprints_from_mol(mol)[0]
    normalization = json.loads(args.normalization_stats.read_text()) if args.normalization_stats else None

    summary = {
        "dataset": str(dataset_path.resolve()),
        "dataset_rows": len(frame),
        "fingerprint": "ECFP4-2048",
        "models": {},
    }
    models = {}
    for name in PROPERTY_NAMES:
        target = frame[name].to_numpy(dtype=np.float64)
        model_path = evaluator_root / f"{name}.joblib"
        print(f"Training {name} RandomForestRegressor", flush=True)
        model = RandomForestRegressor(
            random_state=0,
            n_jobs=args.n_jobs,
            n_estimators=int(args.n_estimators),
        )
        model.fit(fingerprints, target)
        prediction = model.predict(fingerprints)
        training_mae = float(mean_absolute_error(target, prediction))
        dump(model, model_path)
        model_summary = {
            "path": str(model_path.resolve()),
            "training_rows": len(frame),
            "training_mae": training_mae,
        }
        if evaluation is not None:
            evaluation_target = evaluation[name].to_numpy(dtype=np.float64)
            evaluation_prediction = model.predict(evaluation_fingerprints)
            evaluation_mae = float(mean_absolute_error(evaluation_target, evaluation_prediction))
            model_summary["heldout_rows"] = len(evaluation)
            model_summary["heldout_mae"] = evaluation_mae
            if normalization is not None:
                model_summary["heldout_normalized_mae"] = evaluation_mae / float(normalization[name]["std"])
        summary["models"][name] = model_summary
        models[name] = model
        print(f"{name} training MAE: {training_mae}", flush=True)

    if evaluation is not None and normalization is not None:
        summary["heldout_avg_normalized_mae"] = float(
            np.mean([summary["models"][name]["heldout_normalized_mae"] for name in PROPERTY_NAMES])
        )

    benchmark_frame = evaluation.iloc[:200] if evaluation is not None else frame.iloc[:200]
    benchmark_calls = []
    for _ in range(3):
        start = time.perf_counter()
        benchmark_fingerprints = np.empty((len(benchmark_frame), 2048), dtype=np.uint8)
        for index, smiles in enumerate(benchmark_frame["smiles"]):
            benchmark_fingerprints[index] = TaskModel.fingerprints_from_mol(Chem.MolFromSmiles(smiles))[0]
        for model in models.values():
            model.predict(benchmark_fingerprints)
        benchmark_calls.append(time.perf_counter() - start)
    median_call = float(np.median(benchmark_calls))
    summary["runtime"] = {
        "batch_size": len(benchmark_frame),
        "batch_call_seconds": benchmark_calls,
        "median_batch_call_seconds": median_call,
        "median_seconds_per_molecule": median_call / len(benchmark_frame),
        "median_molecules_per_second": len(benchmark_frame) / median_call,
        "model_size_bytes": sum((evaluator_root / f"{name}.joblib").stat().st_size for name in PROPERTY_NAMES),
    }

    (evaluator_root / "oracle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
