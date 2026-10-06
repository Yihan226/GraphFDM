"""Download the rQM9 v1 archive used by the molecule benchmark."""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = "https://zenodo.org/records/17726328/files/qm9-sdf-data.zip?download=1"
MD5 = "b8146a429edaac13ab82d12570181d25"
REQUIRED = ("rQM9_v1.sdf", "gdb9.sdf.csv", "problematic_indices_v1.npy")


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=root / "data" / "molecule" / "source")
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if all((output_dir / name).is_file() for name in REQUIRED):
        print(f"rQM9 source files already exist in {output_dir}")
        return

    archive = output_dir / "qm9-sdf-data.zip"
    if not archive.exists():
        print(f"Downloading {args.url}", flush=True)
        urllib.request.urlretrieve(args.url, archive)
    observed = md5sum(archive)
    if observed != MD5:
        raise ValueError(f"Archive checksum mismatch: expected {MD5}, observed {observed}")

    extracted = output_dir / "_extracted"
    extracted.mkdir(exist_ok=True)
    safe_extract(archive, extracted)
    for name in REQUIRED:
        matches = list(extracted.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {name} in the official archive, found {len(matches)}")
        shutil.copy2(matches[0], output_dir / name)
    shutil.rmtree(extracted)
    print(f"Prepared rQM9 source files in {output_dir}")


if __name__ == "__main__":
    main()
