import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROPERTY_NAMES = ["mu", "alpha", "homo", "lumo", "cv"]
PT_SIZE = 100000
CONDITIONAL_SIZE = 10000
TRAIN_SIZE = 6000
VAL_SIZE = 2000


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        type=Path,
        default=root / "data" / "molecule" / "raw" / "rqm9_v1_clean.csv.gz",
    )
    parser.add_argument("--data_root", type=Path, default=root / "data")
    parser.add_argument("--manifest_dir", type=Path, default=root / "manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pt_size", type=int, default=PT_SIZE)
    parser.add_argument("--conditional_size", type=int, default=CONDITIONAL_SIZE)
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--val_size", type=int, default=VAL_SIZE)
    args = parser.parse_args()

    raw_dir = args.data_root / "molecule" / "raw"
    manifest_dir = args.manifest_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(args.clean)
    required = ["source_index", "smiles", *PROPERTY_NAMES]
    missing = [name for name in required if name not in clean.columns]
    if missing:
        raise ValueError(f"Clean rQM9 is missing columns: {missing}")
    pt_size = int(args.pt_size)
    conditional_size = int(args.conditional_size)
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    test_size = conditional_size - train_size - val_size
    if min(pt_size, conditional_size, train_size, val_size, test_size) < 1:
        raise ValueError("All requested dataset and split sizes must be positive.")
    if conditional_size > pt_size:
        raise ValueError("conditional_size cannot exceed pt_size.")
    if len(clean) < pt_size:
        raise ValueError(f"Expected at least {pt_size} clean rQM9 records, found {len(clean)}")
    if clean["smiles"].isna().any() or clean[PROPERTY_NAMES].isna().any().any():
        raise ValueError("Clean rQM9 contains missing SMILES or property values")
    if clean["smiles"].duplicated().any():
        raise ValueError("Clean rQM9 contains duplicate SMILES")

    order = np.random.RandomState(args.seed).permutation(len(clean))
    pt_positions = order[:pt_size]
    conditional_positions = pt_positions[:conditional_size]
    pt = clean.iloc[np.sort(pt_positions)].reset_index(drop=True)
    conditional = clean.iloc[np.sort(conditional_positions)].reset_index(drop=True)

    split_order = np.random.RandomState(args.seed).permutation(conditional_size)
    train_positions = split_order[:train_size]
    val_positions = split_order[train_size : train_size + val_size]
    test_positions = split_order[train_size + val_size :]
    train = conditional.iloc[train_positions]
    val = conditional.iloc[val_positions]
    test = conditional.iloc[test_positions]

    pt_smiles = set(pt["smiles"])
    conditional_smiles = set(conditional["smiles"])
    split_sets = [set(frame["smiles"]) for frame in [train, val, test]]
    if len(pt_smiles) != pt_size or len(conditional_smiles) != conditional_size:
        raise ValueError("PT or conditional dataset size changed after deduplication")
    if not conditional_smiles.issubset(pt_smiles):
        raise ValueError("The conditional dataset is not a subset of PT")
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Conditional train/validation/test splits overlap")
    if set().union(*split_sets) != conditional_smiles:
        raise ValueError("Conditional splits do not cover the 10K dataset")

    pt_path = raw_dir / "pretrain.csv.gz"
    conditional_path = raw_dir / "mu_alpha_homo_lumo_cv.csv.gz"
    stats_path = raw_dir / "mu_alpha_homo_lumo_cv.stats.json"
    pt[["smiles"]].to_csv(pt_path, index=False)
    conditional[["smiles", *PROPERTY_NAMES]].to_csv(conditional_path, index=False)

    stats = {
        name: {
            "count": train_size,
            "mean": float(train[name].mean()),
            "std": float(train[name].std(ddof=0)),
        }
        for name in PROPERTY_NAMES
    }
    write_json(stats_path, stats)

    pt_manifest_path = manifest_dir / "pt_100k_seed42.json"
    conditional_manifest_path = manifest_dir / "conditional_10k_seed42.json"
    split_manifest_path = manifest_dir / "sft_622_seed42.json"
    write_json(
        pt_manifest_path,
        {
            "dataset": "rQM9_v1",
            "seed": args.seed,
            "selection": "first_100000_of_seeded_permutation",
            "size": pt_size,
            "source_indices": pt["source_index"].astype(int).tolist(),
            "selection_order_source_indices": clean.iloc[pt_positions]["source_index"].astype(int).tolist(),
        },
    )
    write_json(
        conditional_manifest_path,
        {
            "dataset": "rQM9_v1",
            "seed": args.seed,
            "selection": "first_10000_of_pt_permutation",
            "size": conditional_size,
            "source_indices": conditional["source_index"].astype(int).tolist(),
        },
    )
    write_json(
        split_manifest_path,
        {
            "seed": args.seed,
            "train_fraction": 0.6,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "train": train["smiles"].tolist(),
            "val": val["smiles"].tolist(),
            "test": test["smiles"].tolist(),
        },
    )

    summary = {
        "dataset": "rQM9_v1",
        "seed": args.seed,
        "source_records": len(clean),
        "pt_records": pt_size,
        "conditional_records": conditional_size,
        "sft_sdc_train_records": train_size,
        "validation_records": val_size,
        "test_records": test_size,
        "conditional_is_subset_of_pt": True,
        "rf_training_records": pt_size,
        "pt_path": str(pt_path.resolve()),
        "conditional_path": str(conditional_path.resolve()),
        "normalization_source": f"conditional_train_{train_size}",
        "normalization_stats_path": str(stats_path.resolve()),
        "pt_manifest_path": str(pt_manifest_path.resolve()),
        "conditional_manifest_path": str(conditional_manifest_path.resolve()),
        "split_manifest_path": str(split_manifest_path.resolve()),
    }
    write_json(manifest_dir / "preparation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
