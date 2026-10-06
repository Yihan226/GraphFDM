"""End-to-end preparation of the rQM9 molecule benchmark."""

import argparse
import subprocess
import sys
from pathlib import Path


def run(*arguments):
    print("+", " ".join(map(str, arguments)), flush=True)
    subprocess.run([str(value) for value in arguments], check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "minimal"), default="full")
    parser.add_argument("--data_root", type=Path, default=root / "data")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    source_dir = data_root / "molecule" / "source"
    manifests = data_root / "molecule" / "manifests"
    if not args.skip_download:
        run(sys.executable, root / "scripts" / "download_rqm9.py", "--output_dir", source_dir)
    run(
        sys.executable,
        root / "scripts" / "prepare_rqm9.py",
        "--source_dir",
        source_dir,
        "--data_root",
        data_root,
        "--manifest_dir",
        manifests,
    )

    if args.profile == "minimal":
        sizes = (512, 100, 60, 20)
    else:
        sizes = (100000, 10000, 6000, 2000)
    run(
        sys.executable,
        root / "scripts" / "prepare_100k_10k.py",
        "--clean",
        data_root / "molecule" / "raw" / "rqm9_v1_clean.csv.gz",
        "--data_root",
        data_root,
        "--manifest_dir",
        manifests,
        "--pt_size",
        sizes[0],
        "--conditional_size",
        sizes[1],
        "--train_size",
        sizes[2],
        "--val_size",
        sizes[3],
    )

    raw = data_root / "molecule" / "raw"
    conditional = raw / "mu_alpha_homo_lumo_cv.csv.gz"
    stats = raw / "mu_alpha_homo_lumo_cv.stats.json"
    run(
        sys.executable,
        root / "scripts" / "build_six_property_conditional.py",
        "--input_csv",
        conditional,
        "--split_manifest",
        manifests / "sft_622_seed42.json",
        "--output_csv",
        conditional,
        "--output_stats",
        stats,
    )
    oracle_source = conditional if args.profile == "minimal" else raw / "rqm9_v1_clean.csv.gz"
    run(
        sys.executable,
        root / "scripts" / "train_rf_oracles.py",
        "--data_root",
        data_root,
        "--dataset",
        oracle_source,
        "--normalization_stats",
        stats,
        "--n_estimators",
        20 if args.profile == "minimal" else 100,
    )
    print(f"Molecule data are ready under {data_root / 'molecule'}")


if __name__ == "__main__":
    main()
