#!/usr/bin/env python3
"""Construye manifiestos DERIVADOS para evaluación externa y splits agrupados.

Proyecto: GVIS-OCR_license_plates
Guardar en: scripts/elpd/build_elp_eval_manifests.py

Entrada (solo lectura):
  /home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO/manifest.csv
Salida predeterminada:
  results/audit/ELPD/eval_manifests/
    external_eval_manifest.csv
    grouped_split_manifest.csv
    grouped_split_changes.csv
    manifest_summary.txt

La etiqueta del archivo NO equivale a ground truth verificado. Solo se consideran
verificadas las filas cuyo label_status indique una verificación humana explícita.
Se respeta 'scenario' original. La propuesta opcional se guarda en 'split_grouped'.
NO modifica imágenes, el manifest de origen ni los splits originales.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
DEFAULT_MANIFEST = DATA_ROOT / "processed" / "ELPD_LPO" / "manifest.csv"
DEFAULT_OUTDIR = Path.cwd() / "results" / "audit" / "ELPD" / "eval_manifests"
SPLITS = ("train", "val", "test")
PRIORITY = ("test", "val", "train")
VERIFIED_STATES = {"visually_verified", "human_verified", "manually_verified", "verified"}
NEEDED_FIELDS = {"image_path", "text_normalized", "scenario", "label_status"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el manifiesto: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = NEEDED_FIELDS.difference(fields)
        if missing:
            raise ValueError(f"Faltan columnas en el manifest: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("El manifest está vacío")

    paths = set()
    for line, row in enumerate(rows, 2):
        image_path = (row.get("image_path") or "").strip()
        scenario = (row.get("scenario") or "").strip().lower()
        if not image_path or image_path in paths:
            raise ValueError(f"Fila {line}: image_path vacío o duplicado: {image_path!r}")
        if Path(image_path).is_absolute() or ".." in Path(image_path).parts:
            raise ValueError(f"Fila {line}: ruta de imagen inválida: {image_path!r}")
        if scenario not in SPLITS:
            raise ValueError(f"Fila {line}: scenario inesperado: {scenario!r}")
        paths.add(image_path)
    return fields, rows


def save_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generar manifiestos derivados de ELPD sin modificar el original")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--expected-count", type=int, default=735,
                        help="Filas esperadas (0 para desactivar la comprobación)")
    args = parser.parse_args()

    base_fields, rows = read_manifest(args.manifest)
    if args.expected_count and len(rows) != args.expected_count:
        raise ValueError(f"Se esperaban {args.expected_count} imágenes; aparecen {len(rows)}")

    # Agrupación conservadora por ID de matrícula, no por vehículo demostrado.
    group_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = (row.get("text_normalized") or "").strip().upper()
        if label:
            group_by_label[label].append(row)

    conflicts = {
        label: members for label, members in group_by_label.items()
        if len({m["scenario"].strip().lower() for m in members}) > 1
    }
    destination = {
        label: next(split for split in PRIORITY
                    if any(m["scenario"].strip().lower() == split for m in members))
        for label, members in group_by_label.items()
    }

    original_counts = Counter()
    grouped_counts = Counter()
    transition_counts = Counter()
    external_rows: list[dict[str, str]] = []
    grouped_rows: list[dict[str, str]] = []
    changed_rows: list[dict[str, str]] = []
    verified_count = 0
    empty_labels = 0

    for row in rows:
        label = (row.get("text_normalized") or "").strip().upper()
        scenario = row["scenario"].strip().lower()
        verified = (row.get("label_status") or "").strip().lower() in VERIFIED_STATES
        original_counts[scenario] += 1
        verified_count += int(verified)
        empty_labels += int(not label)

        # Para etiquetas faltantes se evita agrupar por una cadena vacía.
        group_id = label if label else f"UNLABELED::{row['image_path']}"
        group_size = len(group_by_label[label]) if label else 1
        cross = label in conflicts if label else False
        proposed = destination[label] if label else scenario
        grouped_counts[proposed] += 1
        if proposed != scenario:
            transition_counts[(scenario, proposed)] += 1

        common = {
            **row,
            "plate_group_id": group_id,
            "plate_group_size": str(group_size),
            "cross_split_original": "yes" if cross else "no",
            "label_review_status": "verified" if verified else "pending_validation",
        }
        external_rows.append({
            **common,
            "external_eval_scope": "all_images_candidate_labels",
            "strict_gt_eligible": "yes" if verified else "no",
        })

        grouped = {
            **common,
            "split_original": scenario,
            "split_grouped": proposed,
            "split_change_proposed": "yes" if proposed != scenario else "no",
            "proposal_status": (
                "change_proposed_not_applied" if proposed != scenario
                else "grouped_no_change" if cross
                else "original_retained"
            ),
        }
        grouped_rows.append(grouped)
        if proposed != scenario:
            changed_rows.append(grouped)

    # Garantiza que ninguna etiqueta candidata no vacía se reparte entre splits propuestos.
    proposed_by_group: dict[str, set[str]] = defaultdict(set)
    for r in grouped_rows:
        if r["text_normalized"].strip():
            proposed_by_group[r["plate_group_id"]].add(r["split_grouped"])
    if any(len(splits) != 1 for splits in proposed_by_group.values()):
        raise RuntimeError("La propuesta todavía contiene IDs compartidos entre splits")
    if sum(original_counts.values()) != len(rows) or sum(grouped_counts.values()) != len(rows):
        raise RuntimeError("Error de recuento durante la agrupación")

    args.outdir.mkdir(parents=True, exist_ok=True)
    extra_common = ["plate_group_id", "plate_group_size", "cross_split_original", "label_review_status"]
    ext_fields = base_fields + extra_common + ["external_eval_scope", "strict_gt_eligible"]
    grouped_fields = base_fields + extra_common + [
        "split_original", "split_grouped", "split_change_proposed", "proposal_status"
    ]
    save_csv(args.outdir / "external_eval_manifest.csv", ext_fields, external_rows)
    save_csv(args.outdir / "grouped_split_manifest.csv", grouped_fields, grouped_rows)
    save_csv(args.outdir / "grouped_split_changes.csv", grouped_fields, changed_rows)

    summary = [
        "=== MANIFIESTOS DERIVADOS ELPD ===",
        f"Manifest original: {args.manifest}",
        f"SHA256 manifest original: {file_sha256(args.manifest)}",
        f"Imágenes: {len(rows)}",
        f"IDs de matrícula únicos con etiqueta: {len(group_by_label)}",
        f"IDs compartidos entre splits: {len(conflicts)}",
        f"Imágenes afectadas por cruces: {sum(len(v) for v in conflicts.values())}",
        f"Reasignaciones propuestas (NO aplicadas): {len(changed_rows)}",
        f"Etiquetas visualmente verificadas según manifest: {verified_count}",
        f"Etiquetas todavía sin validación humana: {len(rows)-verified_count}",
        f"Etiquetas vacías: {empty_labels}",
        "Particiones originales:",
        *(f"- {s}: {original_counts[s]}" for s in SPLITS),
        "Particiones propuestas:",
        *(f"- {s}: {grouped_counts[s]}" for s in SPLITS),
        "Movimientos propuestos:",
        *(f"- {a} -> {b}: {n}" for (a, b), n in sorted(transition_counts.items())),
        "",
        "AVISOS:",
        "- La coincidencia de texto NO demuestra identidad de vehículo.",
        "- 'external_eval_manifest' incluye etiquetas candidatas, NO todas verificadas.",
        "- Para métricas finales rigurosas, validar manualmente el ground truth utilizado.",
        "- 'scenario' y el manifest original permanecen sin cambios.",
        f"Resultados: {args.outdir}",
        "=== FINALIZADO ===",
    ]
    (args.outdir / "manifest_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
