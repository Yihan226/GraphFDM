import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


PROPERTY_NAMES = ["mu", "alpha", "homo", "lumo", "cv"]
EXPECTED_ARCHIVE_MD5 = "b8146a429edaac13ab82d12570181d25"


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_clean_frame(source_dir):
    sdf_path = source_dir / "rQM9_v1.sdf"
    property_path = source_dir / "gdb9.sdf.csv"
    problematic_path = source_dir / "problematic_indices_v1.npy"

    properties = pd.read_csv(property_path)
    required = ["mol_id", *PROPERTY_NAMES]
    missing = [name for name in required if name not in properties.columns]
    if missing:
        raise ValueError(f"Missing rQM9 columns: {missing}")
    properties = properties[required].copy()
    for name in PROPERTY_NAMES:
        properties[name] = pd.to_numeric(properties[name], errors="raise")

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    if len(supplier) != len(properties):
        raise ValueError(f"SDF/CSV row mismatch: {len(supplier)} vs {len(properties)}")

    problematic = set(np.load(problematic_path).astype(int).tolist())
    if any(index < 0 or index >= len(supplier) for index in problematic):
        raise ValueError("problematic_indices_v1.npy contains out-of-range indices")

    records = []
    counts = {
        "source_records": len(supplier),
        "official_problematic": 0,
        "sanitize_failed": 0,
        "formal_charge_filtered": 0,
        "stereochemistry_removed": 0,
        "disconnected": 0,
        "empty_smiles": 0,
        "nonfinite_property": 0,
    }
    for index, mol in enumerate(supplier):
        if index in problematic:
            counts["official_problematic"] += 1
            continue
        if mol is None:
            counts["sanitize_failed"] += 1
            continue
        source_id = str(properties.iloc[index]["mol_id"])
        if mol.GetProp("_Name") != source_id:
            raise ValueError(f"SDF/CSV identity mismatch at row {index}: {mol.GetProp('_Name')} vs {source_id}")
        if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
            counts["formal_charge_filtered"] += 1
            continue
        has_stereochemistry = any(
            atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
        ) or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds())
        counts["stereochemistry_removed"] += int(has_stereochemistry)
        Chem.RemoveStereochemistry(mol)
        mol = Chem.RemoveHs(mol)
        if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
            raise ValueError(f"Explicit hydrogen remains after normalization at row {index}")
        if len(Chem.GetMolFrags(mol)) != 1:
            counts["disconnected"] += 1
            continue
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        if not smiles:
            counts["empty_smiles"] += 1
            continue
        values = properties.iloc[index][PROPERTY_NAMES].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            counts["nonfinite_property"] += 1
            continue
        records.append(
            {
                "source_index": index,
                "mol_id": source_id,
                "smiles": smiles,
                **{name: float(value) for name, value in zip(PROPERTY_NAMES, values)},
            }
        )

    frame = pd.DataFrame(records).sort_values("source_index").reset_index(drop=True)
    duplicate_mask = frame.duplicated("smiles", keep=False)
    counts["canonical_duplicate_groups"] = int(frame.loc[duplicate_mask, "smiles"].nunique())
    counts["canonical_duplicate_rows_removed"] = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].reset_index(drop=True)
    counts["clean_unique_records"] = len(frame)
    return frame, counts


def write_outputs(frame, counts, data_root, manifest_dir):
    raw_dir = data_root / "molecule" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    clean_path = raw_dir / "rqm9_v1_clean.csv.gz"
    frame.to_csv(clean_path, index=False)
    summary = {
        **counts,
        "smiles_representation": "canonical_nonisomeric_no_explicit_hydrogen_no_formal_charge",
        "property_names": PROPERTY_NAMES,
        "clean_path": str(clean_path.resolve()),
    }
    (manifest_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=root / "data" / "molecule" / "source")
    parser.add_argument("--data_root", type=Path, default=root / "data")
    parser.add_argument("--manifest_dir", type=Path, default=root / "manifests")
    args = parser.parse_args()

    archive_path = args.source_dir / "qm9-sdf-data.zip"
    if archive_path.exists() and md5sum(archive_path) != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"Official archive checksum mismatch: {archive_path}")

    RDLogger.DisableLog("rdApp.*")
    frame, counts = build_clean_frame(args.source_dir)
    summary = write_outputs(frame, counts, args.data_root, args.manifest_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
