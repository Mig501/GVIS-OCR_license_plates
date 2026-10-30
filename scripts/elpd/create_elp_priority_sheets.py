"""Genera hojas de contacto de revisión de alertas altas de ELPD.

Lee `results/audit/ELPD/suspicious_images.csv` y las imágenes de
`datasets/ocr_matriculas/processed/ELPD_LPO/images`.
No modifica las imágenes originales ni el manifest.

Uso desde ~/ocr_matriculas (mediante SLURM):
    python scripts/elp/create_elp_priority_sheets.py
"""

import csv
import math
import os
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


DATA_ROOT = Path(
    os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas")
)
DATASET_DIR = DATA_ROOT / "processed" / "ELPD_LPO"
REPORT_PATH = Path("results/audit/ELPD/suspicious_images.csv")
OUTPUT_DIR = Path("results/audit/ELPD/priority_sheets")

COLUMNS = 4
ROWS = 4
PER_PAGE = COLUMNS * ROWS
CELL_W = 460
CELL_H = 270
IMAGE_W = 430
IMAGE_H = 175


def font(size):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def safe_image_path(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("images",):
        raise ValueError(f"Ruta de imagen no válida: {relative}")
    return DATASET_DIR / path


def load_high_priority_rows():
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"No existe el informe: {REPORT_PATH}")

    with REPORT_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        expected = {"image_path", "text_normalized", "split", "review_priority", "review_reasons"}
        missing = expected.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Columnas no encontradas: {sorted(missing)}")
        rows = [r for r in reader if r["review_priority"].strip().lower() == "high"]

    # Revisar primero el test, después validación y entrenamiento.
    order = {"test": 0, "val": 1, "train": 2}
    rows.sort(key=lambda r: (order.get(r["split"], 3), r["image_path"]))
    return rows


def draw_card(sheet, draw, row, index, x, y, title_font, reason_font):
    draw.rectangle(
        (x + 3, y + 3, x + CELL_W - 4, y + CELL_H - 4),
        outline=(195, 195, 195), width=2,
    )

    image_path = safe_image_path(row["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    with Image.open(image_path) as source:
        original_size = source.size
        photo = ImageOps.exif_transpose(source).convert("RGB")
        photo = ImageOps.contain(photo, (IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)

    ix = x + (CELL_W - photo.width) // 2
    iy = y + 9 + (IMAGE_H - photo.height) // 2
    sheet.paste(photo, (ix, iy))

    label = row["text_normalized"]
    split = row["split"]
    draw.text((x + 12, y + 186), f"{index:02d} · {label}  [{split}]", fill=(15, 15, 15), font=title_font)

    reasons = row.get("review_reasons", "").replace("|", ", ")
    lines = wrap(reasons, width=47, break_long_words=False, break_on_hyphens=False)
    for offset, line in enumerate(lines[:2]):
        draw.text((x + 12, y + 215 + 18 * offset), line, fill=(165, 37, 37), font=reason_font)
    draw.text((x + CELL_W - 104, y + 248), f"{original_size[0]}×{original_size[1]}", fill=(95, 95, 95), font=reason_font)


def main():
    print("=== HOJAS DE REVISION PRIORITARIA ELPD ===")
    rows = load_high_priority_rows()
    if not rows:
        print("No hay imágenes con review_priority=high.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = font(21)
    reason_font = font(15)
    index_rows = []
    pages = math.ceil(len(rows) / PER_PAGE)

    for page in range(pages):
        batch = rows[page * PER_PAGE:(page + 1) * PER_PAGE]
        sheet = Image.new("RGB", (COLUMNS * CELL_W, ROWS * CELL_H), "white")
        draw = ImageDraw.Draw(sheet)

        for position, row in enumerate(batch):
            index = page * PER_PAGE + position + 1
            x = (position % COLUMNS) * CELL_W
            y = (position // COLUMNS) * CELL_H
            draw_card(sheet, draw, row, index, x, y, title_font, reason_font)
            index_rows.append({
                "index": index,
                "sheet": f"high_priority_{page + 1:02d}.jpg",
                "image_path": row["image_path"],
                "text_normalized": row["text_normalized"],
                "split": row["split"],
                "review_reasons": row["review_reasons"],
            })

        output_path = OUTPUT_DIR / f"high_priority_{page + 1:02d}.jpg"
        sheet.save(output_path, quality=92, subsampling=0)
        print("Generada:", output_path)

    index_path = OUTPUT_DIR / "priority_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)

    print("\n=== RESULTADOS ===")
    print("Imágenes de prioridad alta:", len(rows))
    print("Hojas generadas:", pages)
    print("Índice:", index_path)
    print("Directorio:", OUTPUT_DIR.resolve())
    print("No se han modificado las imágenes ni el manifest.csv")
    print("=== FINALIZADO ===")


if __name__ == "__main__":
    main()
