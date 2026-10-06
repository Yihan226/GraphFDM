# Molecule data

Run `python scripts/prepare_molecule.py --profile full`. The script downloads `qm9-sdf-data.zip` from the public PropMolFlow Zenodo record, verifies MD5 `b8146a429edaac13ab82d12570181d25`, extracts rQM9 v1, applies the documented cleaning rules, constructs the deterministic 100k/10k and 6000/2000/2000 splits, adds SA, and trains the RF property evaluators.

Use `--profile minimal` only for the reduced execution check.
