#!/usr/bin/env python3
"""Genera hojas para validar visualmente el ground truth de ELPD (split agrupado).

Ubicacion en el proyecto: scripts/elpd/create_elp_gt_review_sheets.py

Lectura:
  results/audit/ELPD/eval_manifests/grouped_split_manifest.csv
  /home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/images/
Escritura (solo informes):
  results/audit/ELPD/gt_review/test_grouped/
    test_gt_01.jpg ... test_gt_04.jpg
    gt_validation.csv
    gt_review_summary.txt

No modifica imagenes, manifest.csv ni particiones. El CSV de revision NO se
sobrescribe si existe para no perder decisiones humanas anteriores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
DEFAULT_DATASET = DATA_ROOT / "processed" / "ELPD_LPO"
DEFAULT_GROUPED = Path.cwd() / "results" / "audit" / "ELPD" / "eval_manifests" / "grouped_split_manifest.csv"
DEFAULT_OUTDIR = Path.cwd() / "results" / "audit" / "ELPD" / "gt_review" / "test_grouped"
SPLITS = {"train", "val", "test"}
VALID_STATUSES = {"pending", "verified", "corrected", "unreadable"}
FIELDS = [
    "image_path", "split_original", "split_grouped", "plate_group_id",
    "candidate_text", "review_status", "verified_text", "notes",
]


def get_font(size: int) -> ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_grouped(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero scripts/elpd/build_elp_eval_manifests.py."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "image_path", "text_normalized", "split_original", "split_grouped",
            "plate_group_id", "label_review_status",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Columnas faltantes en grouped_split_manifest.csv: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("El manifiesto agrupado esta vacio")
    seen_paths: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for idx, row in enumerate(rows, 2):
        rel = (row["image_path"] or "").strip()
        p = Path(rel)
        if not rel or p.is_absolute() or ".." in p.parts or rel in seen_paths:
            raise ValueError(f"Fila {idx}: ruta vacia, peligrosa o repetida: {rel!r}")
        seen_paths.add(rel)
        orig = (row["split_original"] or "").strip().lower()
        prop = (row["split_grouped"] or "").strip().lower()
        if orig not in SPLITS or prop not in SPLITS:
            raise ValueError(f"Fila {idx}: split invalido: {orig!r} / {prop!r}")
        group_id = (row["plate_group_id"] or "").strip()
        if group_id:
            group_splits[group_id].add(prop)
    conflict = [k for k, v in group_splits.items() if len(v) > 1]
    if conflict:
        raise ValueError(f"La propuesta todavia comparte identificadores entre splits: {conflict[:5]}")
    return rows


def csv_template(rows: list[dict[str, str]], dest: Path) -> tuple[int, int, str]:
    """Preserva revisiones si el CSV ya existe; devuelve (verificadas, pendientes, nota)."""
    if dest.exists():
        with dest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
                raise ValueError(f"La plantilla existente tiene columnas inesperadas: {dest}")
            existing = list(reader)
        current = {r["image_path"] for r in existing}
        new = {r["image_path"] for r in rows}
        if len(existing) != len(rows) or current != new:
            raise ValueError(
                f"El conjunto de imagenes ha cambiado; se conserva {dest}. "
                "Revisar manualmente antes de continuar para no perder trabajo."
            )
        statuses = Counter((r.get("review_status") or "pending").strip().lower() for r in existing)
        unknown = set(statuses).difference(VALID_STATUSES)
        if unknown:
            raise ValueError(f"Estados de revision no reconocidos: {sorted(unknown)}")
        return statuses["verified"] + statuses["corrected"], statuses["pending"], "Plantilla existente conservada (sin sobrescribir)."

    done = 0
    todo = 0
    with dest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            text = row["text_normalized"].strip().upper()
            verified = row["label_review_status"].strip().lower() == "verified"
            done += int(verified)
            todo += int(not verified)
            writer.writerow({
                "image_path": row["image_path"],
                "split_original": row["split_original"],
                "split_grouped": row["split_grouped"],
                "plate_group_id": row["plate_group_id"],
                "candidate_text": text,
                "review_status": "verified" if verified else "pending",
                "verified_text": text if verified else "",
                "notes": "Verificacion previa registrada en manifest" if verified else "",
            })
    return done, todo, "Plantilla nueva creada."


def sheet(rows: list[dict[str, str]], paths: list[Path], output: Path, sheet_num: int,
          total_sheets: int, start_idx: int, columns: int, per_sheet: int) -> None:
    card_w, card_h = 373, 228
    header_h, footer_h, gap = 90, 46, 8
    inner_w, inner_h = 341, 128
    count = len(rows)
    rows_canvas = math.ceil(per_sheet / columns)
    width = 20 + columns * (card_w + gap)
    height = header_h + rows_canvas * (card_h + gap) + footer_h
    canvas = Image.new("RGB", (width, height), "#f1f5f9")
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, small_font = get_font(24), get_font(17), get_font(13)
    draw.text((23, 13), "ELPD | Validacion visual de etiquetas | test agrupado", font=title_font, fill="#15344f")
    draw.text((23, 51),
              f"Hoja {sheet_num}/{total_sheets}  |  imagenes {start_idx+1}-{start_idx+count}  |  muestra propuesta, no aprobada",
              font=text_font, fill="#52677b")
    for local, (row, image_path) in enumerate(zip(rows, paths)):
        r, c = divmod(local, columns)
        x, y = 16 + c * (card_w + gap), header_h + r * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=7,
                               fill="white", outline="#c9d5df", width=2)
        label = row["text_normalized"].strip().upper()
        draw.text((x + 11, y + 8), f"{start_idx+local+1:03d}. {label}", font=text_font, fill="#133956")
        with Image.open(image_path) as original:
            img = ImageOps.exif_transpose(original).convert("RGB")
            ow, oh = img.size
            ratio = min(inner_w / ow, inner_h / oh)
            size = (max(1, round(ow * ratio)), max(1, round(oh * ratio)))
            img = img.resize(size, Image.Resampling.LANCZOS)
            x_pic = x + 15 + (inner_w - img.width) // 2
            y_pic = y + 35 + (inner_h - img.height) // 2
            canvas.paste(img, (x_pic, y_pic))
        filename = image_path.name
        draw.text((x + 12, y + 169), filename[:48], font=small_font, fill="#24465e")
        draw.text((x + 12, y + 188),
                  f"Original: {ow}x{oh}  |  split: {row['split_original']} -> {row['split_grouped']}",
                  font=small_font, fill="#516778")
        prior = "VERIFICADA ANTES" if row["label_review_status"].lower().strip() == "verified" else "PENDIENTE GT"
        draw.text((x + 12, y + 206), prior, font=small_font,
                  fill="#1f8051" if prior.startswith("VERIFICADA") else "#b4552e")
    draw.text((23, height - 32),
              "Verifica cada caracter directamente en la imagen. No uses el nombre de archivo como evidencia.",
              font=small_font, fill="#425c70")
    canvas.save(output, quality=94, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preparar revision humana GT para el test agrupado ELPD")
    parser.add_argument("--grouped-manifest", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--expected-count", type=int, default=80, help="0: no comprobar cantidad")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    args = parser.parse_args()
    if args.columns <= 0 or args.rows_per_sheet <= 0:
        parser.error("--columns y --rows-per-sheet deben ser mayores que 0")

    manifest = read_grouped(args.grouped_manifest)
    selected = [r for r in manifest if r["split_grouped"].strip().lower() == args.split]
    selected.sort(key=lambda r: (r["plate_group_id"], r["image_path"]))
    if args.expected_count and len(selected) != args.expected_count:
        raise ValueError(f"Se esperaban {args.expected_count} imagenes en {args.split}; hay {len(selected)}")
    if not selected:
        raise ValueError(f"No hay imagenes propuestas para {args.split}")

    files = []
    for row in selected:
        path = args.dataset_root / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Imagen no encontrada: {path}")
        files.append(path)

    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "gt_validation.csv"
    verified, pending, note = csv_template(selected, csv_path)

    per_sheet = args.columns * args.rows_per_sheet
    total_sheets = math.ceil(len(selected) / per_sheet)
    for si in range(total_sheets):
        start = si * per_sheet
        end = min(len(selected), start + per_sheet)
        output = args.outdir / f"{args.split}_gt_{si+1:02d}.jpg"
        sheet(selected[start:end], files[start:end], output, si+1, total_sheets, start,
              args.columns, per_sheet)
        print(f"Generada: {output}")

    orig_counts = Counter(r["split_original"] for r in selected)
    summary = [
        "=== HOJAS DE VALIDACION GT ELPD ===",
        f"Split propuesto: {args.split}",
        f"Imagenes incluidas: {len(selected)}",
        f"Imagenes provenientes de train: {orig_counts['train']}",
        f"Imagenes provenientes de val: {orig_counts['val']}",
        f"Imagenes provenientes de test: {orig_counts['test']}",
        f"Hojas generadas: {total_sheets}",
        f"Etiquetas verificadas anteriormente (o en CSV conservado): {verified}",
        f"Etiquetas pendientes en CSV: {pending}",
        f"Plantilla: {csv_path}",
        f"SHA256 manifiesto derivado: {sha256(args.grouped_manifest)}",
        f"Aviso: {note}",
        "Opciones review_status: pending / verified / corrected / unreadable",
        "verified_text: completar cuando se verifique o corrija; dejar vacio para unreadable.",
        "No se han modificado imagenes ni manifiestos originales.",
        "=== FINALIZADO ===",
    ]
    report = args.outdir / "gt_review_summary.txt"
    report.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
