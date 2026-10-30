"""Consolida decisiones de revision manual de ELPD sin alterar imagenes/manifest.

Ubicacion prevista: scripts/elpd/register_elp_review.py
Ejecutar desde la raiz del proyecto con SLURM:
    python scripts/elpd/register_elp_review.py

Entradas:
  - datasets/.../processed/ELPD_LPO/manifest.csv
  - results/audit/ELPD/audit_results.csv
  - scripts/elpd/elp_review_decisions.csv

Salidas (solo informes):
  - results/audit/ELPD/review_registry.csv
  - results/audit/ELPD/needs_crop.csv
  - results/audit/ELPD/preparation_summary.txt
"""

import csv
import os
import re
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
PROJECT_ROOT = Path.cwd()
MANIFEST_PATH = DATA_ROOT / "processed/ELPD_LPO/manifest.csv"
AUDIT_PATH = PROJECT_ROOT / "results/audit/ELPD/audit_results.csv"
DECISIONS_PATH = PROJECT_ROOT / "scripts/elpd/elp_review_decisions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/audit/ELPD"

CROP_STATUSES = {"pending_review", "valid_crop", "needs_crop", "excluded"}
LABEL_STATUSES = {"pending_review", "verified", "needs_label_review", "unreadable"}
OUTPUT_FIELDS = [
    "image_path", "text_candidate", "verified_text", "split",
    "crop_status", "label_review_status", "pipeline_status",
    "review_priority", "review_reasons", "cross_split_id",
    "original_label_status", "notes",
]


def read_csv(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"Falta el archivo requerido: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Columnas ausentes en {path.name}: {sorted(missing)}")
        return list(reader)


def validate_relpath(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "images":
        raise ValueError(f"image_path no valida: {value}")
    return value


def index_unique(rows, label):
    result = {}
    for row in rows:
        key = validate_relpath(row["image_path"].strip())
        if key in result:
            raise ValueError(f"image_path duplicado en {label}: {key}")
        result[key] = row
    return result


def save_csv(path, rows, fields):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    print("=== CONSOLIDACION DE REVISION ELPD ===")
    manifest = index_unique(
        read_csv(MANIFEST_PATH, ["image_path", "text_normalized", "scenario", "label_status"]),
        "manifest.csv",
    )
    audit = index_unique(
        read_csv(AUDIT_PATH, ["image_path", "review_priority", "review_reasons", "cross_split_id"]),
        "audit_results.csv",
    )
    decisions = index_unique(
        read_csv(DECISIONS_PATH, ["image_path", "crop_status", "label_review_status", "verified_text", "notes"]),
        "elp_review_decisions.csv",
    )

    if set(manifest) != set(audit):
        raise ValueError(
            "El manifest y la auditoria no contienen las mismas imagenes "
            f"(solo manifest={len(set(manifest)-set(audit))}, "
            f"solo auditoria={len(set(audit)-set(manifest))})."
        )
    unknown = set(decisions) - set(manifest)
    if unknown:
        raise ValueError(f"Decisiones de imagenes inexistentes: {sorted(unknown)[:5]}")

    registry = []
    for image_path in sorted(manifest):
        m = manifest[image_path]
        a = audit[image_path]
        d = decisions.get(image_path, {})
        crop_status = (d.get("crop_status") or "pending_review").strip()
        label_status = (d.get("label_review_status") or "pending_review").strip()
        verified_text = (d.get("verified_text") or "").strip().upper()

        if crop_status not in CROP_STATUSES:
            raise ValueError(f"crop_status invalido en {image_path}: {crop_status}")
        if label_status not in LABEL_STATUSES:
            raise ValueError(f"label_review_status invalido en {image_path}: {label_status}")
        if label_status == "verified" and not re.fullmatch(r"[A-Z0-9]+", verified_text):
            raise ValueError(f"Introduce verified_text para la etiqueta verificada de {image_path}")
        if label_status != "verified" and verified_text:
            raise ValueError(f"verified_text sin label_review_status=verified: {image_path}")

        if crop_status == "excluded":
            pipeline_status = "excluded"
        elif crop_status == "needs_crop":
            pipeline_status = "blocked_needs_crop"
        elif label_status in ("needs_label_review", "unreadable"):
            pipeline_status = "blocked_label"
        elif crop_status == "valid_crop" and label_status == "verified":
            pipeline_status = "ready_for_ocr_evaluation"
        else:
            pipeline_status = "pending_validation"

        registry.append({
            "image_path": image_path,
            "text_candidate": m["text_normalized"],
            "verified_text": verified_text,
            "split": m["scenario"],
            "crop_status": crop_status,
            "label_review_status": label_status,
            "pipeline_status": pipeline_status,
            "review_priority": a["review_priority"],
            "review_reasons": a["review_reasons"],
            "cross_split_id": a["cross_split_id"],
            "original_label_status": m["label_status"],
            "notes": (d.get("notes") or "").strip(),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT_DIR / "review_registry.csv", registry, OUTPUT_FIELDS)
    needs_crop = [r for r in registry if r["crop_status"] == "needs_crop"]
    save_csv(OUTPUT_DIR / "needs_crop.csv", needs_crop, OUTPUT_FIELDS)

    status_counts = Counter(r["pipeline_status"] for r in registry)
    split_counts = Counter(r["split"] for r in registry)
    summary = [
        "=== INFORME DE PREPARACION ELPD ===",
        f"Imagenes en el manifest: {len(registry)}",
        f"Decisiones manuales registradas: {len(decisions)}",
        f"Pendientes de recorte: {len(needs_crop)}",
        "",
        "Estado de preparacion:",
        *[f"- {state}: {count}" for state, count in sorted(status_counts.items())],
        "",
        "Particiones originales:",
        *[f"- {split}: {count}" for split, count in sorted(split_counts.items())],
        "",
        "ATENCION:",
        "- 'pending_validation' NO significa que la etiqueta haya sido verificada.",
        "- Las alertas de calidad no implican exclusion automatica.",
        "- Los IDs compartidos entre particiones se mantienen identificados.",
        "- No se alteran las imagenes originales ni manifest.csv.",
    ]
    text = "\n".join(summary) + "\n"
    (OUTPUT_DIR / "preparation_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print("Informes guardados en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
