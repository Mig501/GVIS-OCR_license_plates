
import csv
import io
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

try:
    from PIL import Image
except ImportError:
    Image = None


DATA_ROOT = Path(
    "/home/miguel.sanchez/datasets/ocr_matriculas"
)

ZIP_PATH = (
    DATA_ROOT
    / "raw/ELPD/european-license-plates-dataset.zip"
)

OUTPUT_DIR = DATA_ROOT / "processed/ELPD_LPO"
IMAGES_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

FIELDS = [
    "image_path",
    "source_path",
    "text_original",
    "text_normalized",
    "dataset",
    "track_id",
    "vehicle_id",
    "scenario",
    "view_id",
    "label_status",
    "crop_method",
    "width",
    "height",
]


def extract_label(filename):
    stem = PurePosixPath(filename).stem

    # Eliminar sufijos como _1, _2, etc.
    match = re.fullmatch(r"(.+?)_(\d+)", stem)

    if match:
        original = match.group(1)
        view_id = match.group(2)
    else:
        original = stem
        view_id = ""

    normalized = re.sub(
        r"[^A-Z0-9]", "", original.upper()
    )

    return original, normalized, view_id


def get_dimensions(data):
    if Image is None:
        return "", ""

    with Image.open(io.BytesIO(data)) as image:
        return image.size


def main():
    print("=== PREPARACION ELPD_LPO ===")

    if not ZIP_PATH.is_file():
        raise FileNotFoundError(ZIP_PATH)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Primero analizar los nombres sin modificar archivos.
    with ZipFile(ZIP_PATH) as archive:
        entries = []

        for info in archive.infolist():
            if info.is_dir():
                continue

            path = PurePosixPath(info.filename)

            if path.suffix.lower() not in EXTENSIONS:
                continue

            if (
                len(path.parts) != 3
                or path.parts[0] != "dataset_final"
                or path.parts[1] not in {"train", "val", "test"}
            ):
                raise ValueError(
                    f"Estructura inesperada: {info.filename}"
                )

            split = path.parts[1]
            original, normalized, view_id = (
                extract_label(info.filename)
            )

            entries.append(
                (info, split, original, normalized, view_id)
            )

        # Detectar IDs compartidos entre particiones.
        plate_splits = defaultdict(set)

        for _, split, _, normalized, _ in entries:
            plate_splits[normalized].add(split)

        shared_ids = {
            plate
            for plate, splits in plate_splits.items()
            if len(splits) > 1
        }

        rows = []

        for info, split, original, normalized, view_id in entries:
            source_name = PurePosixPath(info.filename)

            output_name = f"{split}__{source_name.name}"
            output_path = IMAGES_DIR / output_name

            # Leer la imagen original sin transformarla.
            image_bytes = archive.read(info)

            if output_path.exists():
                if output_path.read_bytes() != image_bytes:
                    raise FileExistsError(
                        f"Archivo diferente ya existente: {output_path}"
                    )
            else:
                output_path.write_bytes(image_bytes)

            width, height = get_dimensions(image_bytes)

            status = "filename_unverified"

            if normalized in shared_ids:
                status = "filename_unverified_cross_split"

            rows.append({
                "image_path": f"images/{output_name}",
                "source_path": (
                    f"raw/ELPD/{ZIP_PATH.name}::"
                    f"{info.filename}"
                ),
                "text_original": original,
                "text_normalized": normalized,
                "dataset": "ELPD",
                "track_id": "",
                "vehicle_id": "",
                "scenario": split,
                "view_id": view_id,
                "label_status": status,
                "crop_method": "pre_cropped_original",
                "width": width,
                "height": height,
            })

    # Escribir el manifest de forma segura.
    temp_manifest = MANIFEST_PATH.with_suffix(".csv.tmp")

    with temp_manifest.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    temp_manifest.replace(MANIFEST_PATH)

    print("\n=== RESULTADOS ===")
    print("Imágenes procesadas:", len(rows))
    print("IDs compartidos entre splits:", len(shared_ids))
    print("Directorio:", IMAGES_DIR)
    print("Manifest:", MANIFEST_PATH)

    if Image is None:
        print(
            "Aviso: Pillow no está instalado; "
            "width y height quedan vacíos."
        )

    print("\n=== PROCESO FINALIZADO ===")


if __name__ == "__main__":
    main()
