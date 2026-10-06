import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.property_metric import calculateSAS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_stats", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    frame = pd.read_csv(args.input_csv)
    with open(args.split_manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    frame.insert(1, "SA", calculateSAS(frame["smiles"].tolist()))
    train_smiles = set(manifest["train"])
    train_frame = frame[frame["smiles"].isin(train_smiles)]
    if len(train_frame) != len(train_smiles):
        raise ValueError("The split manifest does not match the conditional dataset.")

    properties = ["SA", "mu", "alpha", "homo", "lumo", "cv"]
    stats = {}
    for name in properties:
        values = train_frame[name].astype(float)
        stats[name] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    frame.to_csv(args.output_csv, index=False, compression="gzip")
    with open(args.output_stats, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
