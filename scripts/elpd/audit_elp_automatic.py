"""Auditoría automática preliminar de ELPD_LPO (sin modificar el dataset).

Dependencias: Pillow, NumPy.
Ejecutar desde ~/ocr_matriculas mediante SLURM o desde VS Code.
"""

import csv
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


DATA_ROOT = Path(
    os.environ.get(
        "OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"
    )
)
DATASET_DIR = DATA_ROOT / "processed" / "ELPD_LPO"
MANIFEST = DATASET_DIR / "manifest.csv"
OUTPUT_DIR = Path.cwd() / "results" / "audit" / "ELPD"

# Umbrales orientativos de selección para revisión humana, no criterios de exclusión.
MIN_WIDTH = 100
MIN_HEIGHT = 30
MIN_ASPECT = 1.8
MAX_ASPECT = 8.5
LOW_BRIGHTNESS = 55
HIGH_BRIGHTNESS = 215
LOW_CONTRAST = 30
LOW_SHARPNESS = 40
NEAR_DUPLICATE_HAMMING = 4

REPORT_FIELDS = [
    "image_path", "text_normalized", "split", "label_status",
    "width", "height", "aspect_ratio", "brightness", "contrast",
    "sharpness", "cross_split_id", "similar_cross_split",
    "review_priority", "review_reasons",
]


def dhash(gray):
    """Hash perceptual de 64 bits. Coincidencia = indicio, no prueba de duplicado."""
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def analyze(row):
    relative = Path(row["image_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Ruta insegura en manifest: {relative}")

    path = DATASET_DIR / relative
    result = {
        "image_path": row["image_path"],
        "text_normalized": row.get("text_normalized", "").strip().upper(),
        "split": row.get("scenario", ""),
        "label_status": row.get("label_status", ""),
        "width": "", "height": "", "aspect_ratio": "",
        "brightness": "", "contrast": "", "sharpness": "",
        "cross_split_id": "no", "similar_cross_split": "",
        "review_priority": "none", "review_reasons": "",
    }
    reasons = set()
    phash = None

    if not re.fullmatch(r"[A-Z0-9]{4,10}", result["text_normalized"]):
        reasons.add("unusual_label")

    try:
        with Image.open(path) as original:
            original.verify()  # Comprueba que se puede interpretar el archivo

        with Image.open(path) as original:
            original = ImageOps.exif_transpose(original)
            w, h = original.size
            result["width"] = w
            result["height"] = h
            if w <= 0 or h <= 0:
                raise ValueError("Dimensiones inválidas")

            ratio = w / h
            result["aspect_ratio"] = round(ratio, 3)

            if w < MIN_WIDTH or h < MIN_HEIGHT:
                reasons.add("small_image")
            if not MIN_ASPECT <= ratio <= MAX_ASPECT:
                reasons.add("unusual_aspect_ratio")

            if str(w) != row.get("width", "").strip() or str(h) != row.get("height", "").strip():
                reasons.add("manifest_size_mismatch")

            gray = original.convert("L")
            gray.thumbnail((256, 96), Image.Resampling.LANCZOS)
            a = np.asarray(gray, dtype=np.float32)

            brightness = float(a.mean())
            contrast = float(a.std())
            result["brightness"] = round(brightness, 2)
            result["contrast"] = round(contrast, 2)

            if brightness < LOW_BRIGHTNESS:
                reasons.add("dark")
            if brightness > HIGH_BRIGHTNESS:
                reasons.add("overexposed")
            if contrast < LOW_CONTRAST:
                reasons.add("low_contrast")

            if min(a.shape) >= 3:
                laplace = (
                    a[:-2, 1:-1] + a[2:, 1:-1]
                    + a[1:-1, :-2] + a[1:-1, 2:]
                    - 4 * a[1:-1, 1:-1]
                )
                sharpness = float(laplace.var())
                result["sharpness"] = round(sharpness, 2)
                if sharpness < LOW_SHARPNESS:
                    reasons.add("possible_blur")

            # Evitar hashes perceptuales poco discriminativos en imágenes planas.
            if contrast >= LOW_CONTRAST:
                phash = dhash(gray)
    except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as exc:
        reasons.add("missing_or_invalid_image")
        print(f"AVISO: {path}: {exc}")

    return result, reasons, phash


def main():
    print("=== AUDITORIA AUTOMATICA ELPD ===")
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    reports = []
    for row in rows:
        reports.append(analyze(row))

    # Un mismo identificador de matrícula no debe considerarse independiente
    # si aparece en particiones distintas.
    splits_by_id = defaultdict(set)
    for item, _, _ in reports:
        splits_by_id[item["text_normalized"]].add(item["split"])
    cross_split_ids = {
        label for label, splits in splits_by_id.items()
        if label and len(splits) > 1
    }

    for item, reasons, _ in reports:
        if item["text_normalized"] in cross_split_ids:
            item["cross_split_id"] = "yes"
            reasons.add("plate_id_cross_split")

    # Similitud perceptual entre imágenes de diferentes particiones.
    # No implica que las imágenes sean iguales o que las etiquetas sean erróneas.
    possible_pairs = 0
    for i, (item_a, reasons_a, hash_a) in enumerate(reports):
        if hash_a is None:
            continue
        for item_b, reasons_b, hash_b in reports[i + 1:]:
            if hash_b is None or item_a["split"] == item_b["split"]:
                continue
            distance = (hash_a ^ hash_b).bit_count()
            if distance <= NEAR_DUPLICATE_HAMMING:
                possible_pairs += 1
                reasons_a.add("possible_visual_duplicate_cross_split")
                reasons_b.add("possible_visual_duplicate_cross_split")
                if not item_a["similar_cross_split"]:
                    item_a["similar_cross_split"] = item_b["image_path"]
                if not item_b["similar_cross_split"]:
                    item_b["similar_cross_split"] = item_a["image_path"]

    counts = Counter()
    for item, reasons, _ in reports:
        counts.update(reasons)
        if "missing_or_invalid_image" in reasons or "unusual_aspect_ratio" in reasons:
            priority = "high"
        elif reasons & {
            "unusual_label", "manifest_size_mismatch", "plate_id_cross_split",
            "possible_visual_duplicate_cross_split", "small_image",
        }:
            priority = "medium"
        elif reasons:
            priority = "low"
        else:
            priority = "none"
        item["review_priority"] = priority
        item["review_reasons"] = "|".join(sorted(reasons))

    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    records = sorted((item for item, _, _ in reports), key=lambda x: (order[x["review_priority"]], x["image_path"]))
    suspicious = [item for item in records if item["review_priority"] != "none"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in (
        ("audit_results.csv", records),
        ("suspicious_images.csv", suspicious),
    ):
        path = OUTPUT_DIR / filename
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(content)

    summary = [
        "=== RESUMEN DE AUDITORIA AUTOMATICA ELPD ===",
        f"Imágenes del manifest: {len(records)}",
        f"Imágenes señaladas para revisión: {len(suspicious)}",
        f"Identificadores entre particiones: {len(cross_split_ids)}",
        f"Parejas visualmente similares entre particiones: {possible_pairs}",
        "",
        "Incidencias detectadas (una imagen puede tener varias):",
    ]
    summary.extend(f"- {reason}: {count}" for reason, count in sorted(counts.items()))
    summary += [
        "", "IMPORTANTE:",
        "Las etiquetas provienen de nombres de archivo y NO se validan automáticamente.",
        "Los umbrales de calidad son heurísticos: pueden producir falsos positivos.",
        "Una imagen borrosa no debe excluirse sin revisión humana.",
        "No se modifican imágenes ni manifest.csv.",
    ]
    summary_text = "\n".join(summary) + "\n"
    (OUTPUT_DIR / "audit_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print("Informes guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
