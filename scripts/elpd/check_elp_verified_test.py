#!/usr/bin/env python3
"""Comprueba en Legio2 la integridad del test verificado de ELPD.

Ubicacion recomendada: scripts/elpd/check_elp_verified_test.py

Entradas (solo lectura):
  results/audit/ELPD/final_eval/verified_test_manifest.csv
  /home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/images/
  /home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/manifest.csv

Salidas (informes):
  results/audit/ELPD/final_eval/integrity_check/image_integrity.csv
  results/audit/ELPD/final_eval/integrity_check/integrity_summary.txt

No modifica ninguna imagen ni los manifiestos originales.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
DATASET_DEFAULT = DATA_ROOT / "processed" / "ELPD_LPO"
INPUT_DEFAULT = Path("results/audit/ELPD/final_eval/verified_test_manifest.csv")
OUTPUT_DEFAULT = Path("results/audit/ELPD/final_eval/integrity_check")
EXPECTED_SOURCE_SHA256 = "61ab2328e8a89263a9fd08a6577fb82069aa3a29d8a63cd49fa42b24447d71e2"
REQUIRED = {
    "image_path", "text_normalized", "text_candidate_original", "plate_group_id",
    "split_original", "split_grouped", "width", "height", "gt_review_status",
    "gt_corrected", "strict_gt_eligible", "label_status",
}
OUTPUT_COLUMNS = [
    "image_path", "text_normalized", "split_original", "gt_review_status",
    "width_manifest", "height_manifest", "width_actual", "height_actual",
    "file_bytes", "sha256", "status", "issues",
]
LABEL_PATTERN = re.compile(r"[A-Z0-9]+\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_image_path(relpath: str) -> bool:
    """Limita las rutas del CSV al subdirectorio images/ del dataset."""
    path = PurePosixPath(relpath)
    return (
        bool(relpath)
        and "\\" not in relpath
        and not path.is_absolute()
        and len(path.parts) == 2
        and path.parts[0] == "images"
        and path.parts[1] not in {".", ".."}
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def count_source_manifest(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def check(args: argparse.Namespace) -> int:
    if not args.manifest.is_file():
        raise FileNotFoundError(f"No existe el manifiesto verificado: {args.manifest}")
    source_manifest = args.dataset_root / "manifest.csv"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"No existe el manifiesto original: {source_manifest}")

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
        rows = list(reader)

    results: list[dict[str, str | int]] = []
    paths_seen: set[str] = set()
    hashes: dict[str, list[str]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    issues_count: Counter[str] = Counter()
    for number, row in enumerate(rows, start=2):
        relpath = (row["image_path"] or "").strip()
        label = (row["text_normalized"] or "").strip()
        candidate = (row["text_candidate_original"] or "").strip()
        review = (row["gt_review_status"] or "").strip().lower()
        corrected = (row["gt_corrected"] or "").strip().lower()
        errors: list[str] = []
        actual_width = actual_height = ""
        file_bytes = ""
        checksum = ""

        def issue(code: str) -> None:
            if code not in errors:
                errors.append(code)
                issues_count[code] += 1

        if relpath in paths_seen:
            issue("duplicate_image_path")
        paths_seen.add(relpath)
        if not safe_image_path(relpath):
            issue("unsafe_or_invalid_image_path")
        if not LABEL_PATTERN.fullmatch(label):
            issue("invalid_ground_truth")
        if (row["plate_group_id"] or "").strip() != label:
            issue("plate_group_id_mismatch")
        if row["split_grouped"].strip().lower() != "test":
            issue("not_grouped_test")
        if row["strict_gt_eligible"].strip().lower() != "yes":
            issue("not_strict_gt_eligible")
        if row["label_status"].strip().lower() != "human_verified":
            issue("not_human_verified")
        if review not in {"verified", "corrected"}:
            issue("invalid_gt_review_status")
        else:
            statuses[review] += 1
        if review == "verified" and (label != candidate or corrected != "no"):
            issue("verified_mismatch")
        if review == "corrected" and (label == candidate or corrected != "yes"):
            issue("corrected_mismatch")
        try:
            expected_width = int(row["width"])
            expected_height = int(row["height"])
            if expected_width <= 0 or expected_height <= 0:
                issue("invalid_dimensions_in_manifest")
        except (TypeError, ValueError):
            expected_width = expected_height = -1
            issue("invalid_dimensions_in_manifest")

        if safe_image_path(relpath):
            image_path = args.dataset_root / relpath
            if not image_path.is_file():
                issue("missing_image")
            else:
                try:
                    file_bytes = image_path.stat().st_size
                    if file_bytes == 0:
                        issue("empty_image_file")
                    # verify() comprueba la estructura y load() obliga a leer pixeles.
                    with Image.open(image_path) as im:
                        im.verify()
                    with Image.open(image_path) as im:
                        im.load()
                        actual_width, actual_height = im.size
                    if (actual_width, actual_height) != (expected_width, expected_height):
                        issue("dimensions_mismatch")
                    checksum = sha256_file(image_path)
                    hashes[checksum].append(relpath)
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    issue("corrupt_or_unreadable_image")
                    print(f"AVISO fila {number}, {relpath}: {type(exc).__name__}: {exc}")

        results.append({
            "image_path": relpath,
            "text_normalized": label,
            "split_original": row["split_original"],
            "gt_review_status": review,
            "width_manifest": row["width"],
            "height_manifest": row["height"],
            "width_actual": actual_width,
            "height_actual": actual_height,
            "file_bytes": file_bytes,
            "sha256": checksum,
            "status": "ok" if not errors else "error",
            "issues": ";".join(errors),
        })

    source_count = count_source_manifest(source_manifest)
    source_sha = sha256_file(source_manifest)
    duplicate_hash_groups = [paths for paths in hashes.values() if len(paths) > 1]
    global_errors: list[str] = []
    if len(rows) != 80:
        global_errors.append(f"Numero inesperado de filas: {len(rows)} (esperadas 80)")
    if statuses != Counter({"verified": 76, "corrected": 4}):
        global_errors.append(f"Estados GT inesperados: {dict(statuses)} (esperados verified=76, corrected=4)")
    unique_labels = len({(r["text_normalized"] or "").strip() for r in rows})
    if unique_labels != 73:
        global_errors.append(f"Numero de matriculas unicas: {unique_labels} (esperadas 73)")
    if source_count != 735:
        global_errors.append(f"Manifest original: {source_count} filas (esperadas 735)")
    if source_sha != EXPECTED_SOURCE_SHA256:
        global_errors.append("El SHA256 del manifest original difiere del registrado en el paso 30")
    if duplicate_hash_groups:
        global_errors.append(f"Grupos de imagenes con SHA256 identico: {len(duplicate_hash_groups)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    report_csv = args.outdir / "image_integrity.csv"
    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    ok_count = sum(r["status"] == "ok" for r in results)
    errors = len(results) - ok_count
    passed = errors == 0 and not global_errors
    lines = [
        "=== ELPD - COMPROBACION FINAL DEL TEST VERIFICADO ===",
        f"Imagenes en manifiesto: {len(rows)}",
        f"Imagenes correctas: {ok_count}",
        f"Imagenes con incidencias: {errors}",
        f"Ground truth: verified={statuses['verified']}, corrected={statuses['corrected']}",
        f"Matriculas unicas: {unique_labels}",
        f"Filas en manifest original: {source_count}",
        f"SHA256 manifest original: {source_sha}",
        f"SHA256 test verificado: {sha256_file(args.manifest)}",
        f"Grupos de imagenes identicas (SHA256): {len(duplicate_hash_groups)}",
    ]
    if issues_count:
        lines.extend(["Incidencias por tipo:", *[f"- {kind}: {count}" for kind, count in sorted(issues_count.items())]])
    if global_errors:
        lines.extend(["Comprobaciones globales fallidas:", *[f"- {message}" for message in global_errors]])
    lines.extend([
        f"Informe por imagen: {report_csv}",
        "No se han modificado imagenes ni manifiestos originales.",
        "ESTADO: APTO PARA EVALUACION EXTERNA" if passed else "ESTADO: REVISION NECESARIA",
        "=== FINALIZADO ===",
    ])
    summary = args.outdir / "integrity_summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica integridad de las 80 imagenes ELPD con GT auditado")
    parser.add_argument("--manifest", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    raise SystemExit(check(args))


if __name__ == "__main__":
    main()
