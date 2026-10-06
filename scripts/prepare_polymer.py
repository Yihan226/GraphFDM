"""Validate user-supplied polymer assets and create the fixed split metadata."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROPERTIES = ("SA", "O2", "N2", "CO2")


def resolve_csv(raw_dir, stem):
    for candidate in (stem, stem.replace("_", "-"), stem.replace("-", "_")):
        for suffix in (".csv.gz", ".csv"):
            path = raw_dir / f"{candidate}{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"Missing {stem}.csv[.gz] under {raw_dir}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=root / "data")
    parser.add_argument("--tokenizer_dir", type=Path, default=root / "assets" / "tokenizers" / "polymer")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    polymer_root = args.data_root.resolve() / "polymer"
    raw_dir = polymer_root / "raw"
    pretrain_path = resolve_csv(raw_dir, "pretrain")
    conditional_path = resolve_csv(raw_dir, "O2_N2_CO2")
    pretrain = pd.read_csv(pretrain_path)
    conditional = pd.read_csv(conditional_path)
    if "smiles" not in pretrain:
        raise ValueError("Polymer pretraining data must contain a smiles column.")
    missing = [name for name in ("smiles", *PROPERTIES) if name not in conditional]
    if missing:
        raise ValueError(f"Polymer conditional data are missing columns: {missing}")
    if len(conditional) != 553:
        raise ValueError(f"Expected 553 conditional polymer records, found {len(conditional)}.")

    order = np.random.RandomState(args.seed).permutation(len(conditional))
    train = conditional.iloc[order[:337]]
    val = conditional.iloc[order[337:444]]
    test = conditional.iloc[order[444:]]
    manifest = {
        "seed": int(args.seed),
        "train_fraction": 337 / 553,
        "val_fraction": 107 / 553,
        "test_fraction": 109 / 553,
        "train": train["smiles"].astype(str).tolist(),
        "val": val["smiles"].astype(str).tolist(),
        "test": test["smiles"].astype(str).tolist(),
    }
    manifest_path = polymer_root / "split_337_107_109_seed42.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    stats = {
        name: {
            "count": len(train),
            "mean": float(train[name].astype(float).mean()),
            "std": float(train[name].astype(float).std(ddof=0)),
        }
        for name in PROPERTIES
    }
    (raw_dir / "O2_N2_CO2.stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for suffix in ("node", "edge", "motif", "ring"):
        path = args.tokenizer_dir / f"pretrain-token.{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing polymer tokenizer asset: {path}")
    evaluator_dir = args.data_root.resolve() / "evaluator"
    for name in ("O2", "N2", "CO2"):
        path = evaluator_dir / f"{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Missing polymer property evaluator: {path}")
    print(
        f"Validated {len(pretrain)} pretraining polymers and the 337/107/109 conditional split."
    )


if __name__ == "__main__":
    main()
