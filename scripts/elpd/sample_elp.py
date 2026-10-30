
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from shutil import copyfileobj
import random
import re


ZIP_PATH = Path(
    "/home/miguel.sanchez/datasets/ocr_matriculas/"
    "raw/ELPD/european-license-plates-dataset.zip"
)

OUTPUT_DIR = Path("results/inspection/ELPD")

SPLITS = ("train", "val", "test")

# IDs que aparecen en distintas particiones
SHARED_IDS = {"381ATK83", "75N1960G", "B121PHD"}


def plate_id(filename):
    stem = PurePosixPath(filename).stem.upper()
    return re.sub(r"_\d+$", "", stem)


def main():
    print("=== MUESTRA VISUAL ELPD ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    with ZipFile(ZIP_PATH) as zf:
        files = {
            split: []
            for split in SPLITS
        }

        for name in zf.namelist():
            path = PurePosixPath(name)

            if (
                len(path.parts) == 3
                and path.parts[0] == "dataset_final"
                and path.parts[1] in files
                and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ):
                files[path.parts[1]].append(name)

        selected = set()

        # 4 imágenes aleatorias por partición
        for split, names in files.items():
            selected.update(
                rng.sample(
                    sorted(names),
                    min(4, len(names))
                )
            )

        # Incluir ejemplos de IDs compartidos
        for names in files.values():
            for name in names:
                if plate_id(name) in SHARED_IDS:
                    selected.add(name)

        print("Imágenes seleccionadas:", len(selected))

        for name in sorted(selected):
            path = PurePosixPath(name)
            split = path.parts[1]

            filename = f"{split}__{path.name}"
            destination = OUTPUT_DIR / filename

            with zf.open(name) as src:
                with open(destination, "wb") as dst:
                    copyfileobj(src, dst)

            print(f"{name} -> {filename}")

    print("\nDestino:", OUTPUT_DIR.resolve())
    print("=== MUESTRA GENERADA ===")


if __name__ == "__main__":
    main()
