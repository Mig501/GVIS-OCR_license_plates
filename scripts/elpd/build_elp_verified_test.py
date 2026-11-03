#!/usr/bin/env python3
"""Construye el manifiesto final de evaluacion ELPD (test agrupado).

Ubicacion: scripts/elpd/build_elp_verified_test.py

Entradas (solo lectura):
  scripts/elpd/gt_validation.csv
  results/audit/ELPD/eval_manifests/grouped_split_manifest.csv
  /home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/images/*

Salidas:
  results/audit/ELPD/final_eval/verified_test_manifest.csv
  results/audit/ELPD/final_eval/gt_corrections.csv
  results/audit/ELPD/final_eval/verified_test_summary.txt

No modifica las imagenes ni el manifest.csv original.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

PROJECT = Path.cwd()
DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
DEFAULT_DATASET = DATA_ROOT / "processed" / "ELPD_LPO"
DEFAULT_GROUPED = PROJECT / "results" / "audit" / "ELPD" / "eval_manifests" / "grouped_split_manifest.csv"
DEFAULT_REVIEW = PROJECT / "scripts" / "elpd" / "gt_validation.csv"
DEFAULT_OUTDIR = PROJECT / "results" / "audit" / "ELPD" / "final_eval"
REVIEW_REQUIRED = (
    "image_path", "split_original", "split_grouped", "plate_group_id",
    "candidate_text", "review_status", "verified_text", "notes",
)
GROUPED_REQUIRED = (
    "image_path", "text_normalized", "split_original", "split_grouped",
    "plate_group_id", "label_status", "label_review_status",
)
EXTRA_FIELDS = (
    "text_candidate_original", "plate_group_id_candidate", "label_status_before_gt",
    "label_review_status_before_gt", "gt_review_status", "gt_review_notes",
    "gt_source", "gt_corrected", "strict_gt_eligible",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, required: tuple[str, ...]) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        absent = set(required) - set(fields)
        if absent:
            raise ValueError(f"Faltan columnas en {path.name}: {sorted(absent)}")
        rows = list(rd)
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_relpath(rel: str) -> None:
    p = Path(rel)
    if not rel or p.is_absolute() or ".." in p.parts or p.parts[:1] != ("images",):
        raise ValueError(f"Ruta relativa de imagen no valida: {rel!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir test ELPD con ground truth auditado")
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--grouped-manifest", type=Path, default=DEFAULT_GROUPED)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    base_fields, grouped = read_csv(args.grouped_manifest, GROUPED_REQUIRED)
    _, reviews = read_csv(args.review, REVIEW_REQUIRED)
    if len(grouped) != 735 or len(reviews) != 80:
        raise ValueError(f"Recuento inesperado: grouped={len(grouped)} (735 esperadas), review={len(reviews)} (80 esperadas)")

    selected = [r for r in grouped if r["split_grouped"].strip().lower() == "test"]
    if len(selected) != 80:
        raise ValueError(f"El manifiesto agrupado contiene {len(selected)} imagenes en test, se esperaban 80")
    index: dict[str, dict[str, str]] = {}
    for row in selected:
        path = row["image_path"].strip()
        validate_relpath(path)
        if path in index:
            raise ValueError(f"Ruta duplicada en el manifiesto agrupado: {path}")
        index[path] = row

    seen: set[str] = set()
    result: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    statuses = Counter()
    originals = Counter()
    for n, review in enumerate(reviews, 2):
        rel = review["image_path"].strip()
        validate_relpath(rel)
        if rel in seen:
            raise ValueError(f"Fila {n}: imagen repetida: {rel}")
        seen.add(rel)
        if rel not in index:
            raise ValueError(f"Fila {n}: imagen no perteneciente al test agrupado: {rel}")
        original = index[rel]
        for key in ("split_original", "split_grouped", "plate_group_id"):
            if review[key].strip() != original[key].strip():
                raise ValueError(f"Fila {n}: {key} no coincide con grouped_split_manifest.csv para {rel}")
        candidate = review["candidate_text"].strip().upper()
        if candidate != original["text_normalized"].strip().upper():
            raise ValueError(f"Fila {n}: etiqueta candidata distinta de la del manifiesto para {rel}")
        status = review["review_status"].strip().lower()
        truth = review["verified_text"].strip().upper()
        if status not in ("verified", "corrected"):
            raise ValueError(f"Fila {n}: estado {status!r}; todas deben estar verificadas o corregidas")
        if not truth or not truth.isascii() or not truth.isalnum():
            raise ValueError(f"Fila {n}: ground truth vacio o con formato incorrecto: {truth!r}")
        if status == "verified" and truth != candidate:
            raise ValueError(f"Fila {n}: estado verified pero cambian los caracteres de {rel}")
        if status == "corrected" and truth == candidate:
            raise ValueError(f"Fila {n}: estado corrected pero no cambia la etiqueta de {rel}")
        if not (args.dataset_root / rel).is_file():
            raise FileNotFoundError(f"Falta imagen procesada: {args.dataset_root / rel}")

        row = dict(original)
        row.update({
            "text_candidate_original": original["text_normalized"],
            "plate_group_id_candidate": original["plate_group_id"],
            "label_status_before_gt": original["label_status"],
            "label_review_status_before_gt": original["label_review_status"],
            "text_normalized": truth,                    # Texto correcto para evaluadores OCR
            "plate_group_id": truth,                     # Agrupacion por GT verificado
            "label_status": "human_verified",
            "label_review_status": "verified",
            "gt_review_status": status,
            "gt_review_notes": review["notes"].strip(),
            "gt_source": "manual_review_test_grouped",
            "gt_corrected": "yes" if status == "corrected" else "no",
            "strict_gt_eligible": "yes",
        })
        result.append(row)
        if status == "corrected":
            changes.append({
                "image_path": rel,
                "candidate_text": candidate,
                "verified_text": truth,
                "notes": review["notes"].strip(),
            })
        originals[row["split_original"]] += 1
        statuses[status] += 1

    if set(index) != seen:
        raise ValueError(f"Faltan {len(set(index) - seen)} imagenes del test agrupado en el archivo de revision")
    if statuses != Counter({"verified": 76, "corrected": 4}):
        raise ValueError(f"Estados inesperados para la revision actual: {dict(statuses)}")
    if originals != Counter({"test": 73, "train": 7}):
        raise ValueError(f"Particiones originales inesperadas: {dict(originals)}")

    result.sort(key=lambda r: (r["plate_group_id"], r["image_path"]))
    output_fields = list(dict.fromkeys([*base_fields, *EXTRA_FIELDS]))
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "verified_test_manifest.csv", output_fields, result)
    write_csv(args.outdir / "gt_corrections.csv", ["image_path", "candidate_text", "verified_text", "notes"], changes)

    unique_texts = len({r["plate_group_id"] for r in result})
    lines = [
        "=== ELPD - TEST AGRUPADO CON GT VERIFICADO ===",
        f"Imagenes evaluables: {len(result)}",
        f"Ground truth verificado sin cambios: {statuses['verified']}",
        f"Ground truth corregido: {statuses['corrected']}",
        "Pendientes / ilegibles: 0",
        f"Identificadores de matricula unicos tras correccion: {unique_texts}",
        f"Procedencia original: train={originals['train']}, test={originals['test']}",
        f"SHA256 revision: {sha256(args.review)}",
        f"SHA256 manifiesto agrupado: {sha256(args.grouped_manifest)}",
        f"Manifiesto final: {args.outdir / 'verified_test_manifest.csv'}",
        f"Correcciones: {args.outdir / 'gt_corrections.csv'}",
        "Imagenes y manifest original no modificados.",
        "=== FINALIZADO ===",
    ]
    (args.outdir / "verified_test_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
