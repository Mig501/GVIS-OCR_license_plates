#!/usr/bin/env python3
"""Create non-destructive candidate LPO crops for three audited ELPD images.

Run from scripts/elpd/ through SLURM, or locally with:
    python scripts/elpd/crop_elp_pending.py --input-dir PATH --output-dir PATH

The source images and the original ELPD_LPO manifest.csv are never modified.
Bounding boxes use PIL's (left, top, right, bottom) convention, with exclusive right/bottom.
"""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path("/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/images")
DEFAULT_OUTPUT = Path("/home/miguel.sanchez/ocr_matriculas/results/inspection/ELPD/crop_candidates")
EXPECTED_SOURCE_SIZE = (1280, 720)

# Coordinates manually verified against the original uploaded 1280x720 images.
# PIL crops contain the plate and its frame, preserving the source pixel values.
CROPS = (
    ("test__PH20CNP.png", "PH20CNP", (676, 392, 785, 424)),
    ("train__SB90TZI.png", "SB90TZI", (494, 463, 675, 519)),
    ("train__TR05RBG.png", "TR05RBG", (710, 439, 944, 515)),
)


def write_preview(crops: list[tuple[str, str, Image.Image]], dest: Path) -> None:
    """Enlarged contact sheet for human review only (not OCR inputs)."""
    width = 900
    cell_height = 245
    canvas = Image.new("RGB", (width, cell_height * len(crops)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for i, (filename, label, crop) in enumerate(crops):
        y0 = i * cell_height
        draw.text((16, y0 + 12), f"{filename} | {label} | crop {crop.width}x{crop.height}", fill="black", font=font)
        # Upscaling affects preview only, never the saved crop.
        available_width, available_height = width - 32, cell_height - 55
        scale = min(available_width / crop.width, available_height / crop.height)
        size = (round(crop.width * scale), round(crop.height * scale))
        enlarged = crop.resize(size, Image.Resampling.NEAREST)
        x = (width - size[0]) // 2
        y = y0 + 38 + (available_height - size[1]) // 2
        canvas.paste(enlarged, (x, y))
        draw.rectangle((x - 1, y - 1, x + size[0], y + size[1]), outline="#666666", width=1)
        if i != len(crops) - 1:
            draw.line((0, y0 + cell_height - 1, width, y0 + cell_height - 1), fill="#cccccc", width=1)
    canvas.save(dest, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewable LPO crop candidates for 3 ELPD full-scene images")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"ERROR: no existe el directorio de origen: {args.input_dir}")

    # Validate all inputs before writing anything.
    for filename, _label, box in CROPS:
        source = args.input_dir / filename
        if not source.is_file():
            raise SystemExit(f"ERROR: falta la imagen de origen: {source}")
        with Image.open(source) as image:
            if image.size != EXPECTED_SOURCE_SIZE:
                raise SystemExit(f"ERROR: tamano inesperado para {filename}: {image.size}, esperado {EXPECTED_SOURCE_SIZE}")
            left, top, right, bottom = box
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise SystemExit(f"ERROR: recorte fuera de limites para {filename}: {box}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    previews = []
    for filename, label, (left, top, right, bottom) in CROPS:
        source = args.input_dir / filename
        with Image.open(source) as image:
            cropped = image.convert("RGB").crop((left, top, right, bottom))
            target = args.output_dir / filename
            cropped.save(target, format="PNG")
            previews.append((filename, label, cropped))
            rows.append({
                "source_image": str(source),
                "candidate_image": str(target),
                "candidate_text": label,
                "x_min": left,
                "y_min": top,
                "x_max": right,
                "y_max": bottom,
                "source_width": image.width,
                "source_height": image.height,
                "crop_width": cropped.width,
                "crop_height": cropped.height,
                "review_status": "pending_visual_approval",
            })
            print(f"Creado: {target.name} | recorte: {cropped.width}x{cropped.height} | bbox: {(left, top, right, bottom)}")

    index = args.output_dir / "crop_candidates.csv"
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    preview = args.output_dir / "crop_preview.jpg"
    write_preview(previews, preview)
    print("\n=== CANDIDATOS DE RECORTE ELPD ===")
    print(f"Recortes candidatos generados: {len(rows)}")
    print(f"Informe de coordenadas: {index}")
    print(f"Hoja de revision: {preview}")
    print("IMPORTANTE: los recortes no se integran todavia en ELPD_LPO.")
    print("Sin modificaciones en imagenes originales ni en manifest.csv.")


if __name__ == "__main__":
    main()
