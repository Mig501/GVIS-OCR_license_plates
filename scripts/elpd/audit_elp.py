
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from zipfile import ZipFile
import hashlib
import re

ZIP_PATH = Path(
    "/home/miguel.sanchez/datasets/ocr_matriculas/"
    "raw/ELPD/european-license-plates-dataset.zip"
)

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    print("=== AUDITORIA ELPD ===")

    split_counts = Counter()
    plate_splits = defaultdict(set)
    plate_files = defaultdict(list)
    image_hashes = defaultdict(list)

    with ZipFile(ZIP_PATH) as zf:
        for filename in zf.namelist():
            path = PurePosixPath(filename)

            if path.suffix.lower() not in EXTENSIONS:
                continue

            parts = path.parts
            split = (
                parts[1]
                if len(parts) >= 3 and parts[0] == "dataset_final"
                else "unknown"
            )

            # ID candidato: no se utiliza todavía como ground truth.
            plate_id = re.sub(
                r"_\d+$", "", path.stem.upper()
            )

            split_counts[split] += 1
            plate_splits[plate_id].add(split)
            plate_files[plate_id].append(filename)

            digest = hashlib.sha256(
                zf.read(filename)
            ).hexdigest()

            image_hashes[digest].append(filename)

    print("\n=== IMAGENES POR PARTICION ===")
    for split, count in sorted(split_counts.items()):
        print(f"{split}: {count}")

    print("\n=== IDs REPETIDOS ENTRE PARTICIONES ===")
    repeated = {
        plate: files
        for plate, files in plate_files.items()
        if len(plate_splits[plate]) > 1
    }

    print("Total:", len(repeated))

    for plate, files in list(sorted(repeated.items()))[:20]:
        print(f"\n{plate}:")
        for filename in files:
            print(" ", filename)

    print("\n=== IMAGENES EXACTAMENTE DUPLICADAS ===")

    duplicates = {
        digest: files
        for digest, files in image_hashes.items()
        if len(files) > 1
    }

    print("Grupos duplicados:", len(duplicates))

    for files in list(duplicates.values())[:10]:
        for filename in files:
            print(" ", filename)
        print()

    print("=== AUDITORIA FINALIZADA ===")


if __name__ == "__main__":
    main()
