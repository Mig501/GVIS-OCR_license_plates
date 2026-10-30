
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET_DIR = Path(
    "/home/miguel.sanchez/datasets/ocr_matriculas/"
    "processed/ELPD_LPO"
)

OUTPUT_DIR = Path(
    "results/inspection/ELPD/contact_sheets"
)

COLUMNS = 5
ROWS = 6
PER_SHEET = COLUMNS * ROWS

CELL_W = 300
CELL_H = 160
IMAGE_W = 280
IMAGE_H = 105


def main():
    print("=== AUDITORIA VISUAL ELPD ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (DATASET_DIR / "manifest.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        records = list(csv.DictReader(f))

    groups = defaultdict(list)

    for row in records:
        groups[row["scenario"]].append(row)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            15,
        )
    except OSError:
        font = ImageFont.load_default()

    total_sheets = 0

    for split in ("train", "val", "test"):
        rows = sorted(
            groups[split],
            key=lambda row: row["image_path"]
        )

        num_sheets = math.ceil(len(rows) / PER_SHEET)

        for page in range(num_sheets):
            subset = rows[
                page * PER_SHEET:(page + 1) * PER_SHEET
            ]

            sheet = Image.new(
                "RGB",
                (COLUMNS * CELL_W, ROWS * CELL_H),
                "white",
            )

            draw = ImageDraw.Draw(sheet)

            for i, row in enumerate(subset):
                x = (i % COLUMNS) * CELL_W
                y = (i // COLUMNS) * CELL_H

                image_path = DATASET_DIR / row["image_path"]

                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source)
                    image = ImageOps.contain(
                        image.convert("RGB"),
                        (IMAGE_W, IMAGE_H),
                    )

                sheet.paste(image, (x + 10, y + 10))

                label = row["text_normalized"]

                draw.text(
                    (x + 10, y + 118),
                    f"GT candidato: {label}",
                    fill="black",
                    font=font,
                )

                if "cross_split" in row["label_status"]:
                    draw.text(
                        (x + 10, y + 138),
                        "REPETIDA ENTRE SPLITS",
                        fill="red",
                        font=font,
                    )

                draw.rectangle(
                    (x, y, x + CELL_W - 1, y + CELL_H - 1),
                    outline="lightgray",
                )

            output = OUTPUT_DIR / f"{split}_{page + 1:02d}.jpg"

            sheet.save(output, quality=92)
            total_sheets += 1

            print(f"Generada: {output}")

    print("\n=== RESULTADOS ===")
    print("Imágenes revisables:", len(records))
    print("Hojas de contacto:", total_sheets)
    print("Directorio:", OUTPUT_DIR.resolve())
    print("=== FINALIZADO ===")


if __name__ == "__main__":
    main()
