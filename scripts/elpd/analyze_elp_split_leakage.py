#!/usr/bin/env python3
"""Audita solapamientos por etiqueta entre train/val/test de ELPD_LPO.

Ubicación en el proyecto: scripts/elpd/analyze_elp_split_leakage.py
Ejecutar desde ~/ocr_matriculas (vía la tarea SLURM de VS Code):
    python scripts/elpd/analyze_elp_split_leakage.py

Lee únicamente el manifest existente y escribe informes bajo results/audit/ELPD/.
NO modifica imágenes, splits, etiquetas ni manifest.csv.

Política propuesta (solo informativa): si una matrícula candidata aparece en
varios splits, conservar todas sus imágenes en test si allí hay alguna; si no,
conservarlas en val si allí hay alguna; si no, train. Prioriza mantener los
identificadores que ya formaban parte del conjunto de evaluación.

ATENCIÓN: la coincidencia de texto entre placas NO demuestra que sean del mismo
vehículo ni que sus etiquetas sean correctas. Se requiere revisión humana.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "processed" / "ELPD_LPO" / "manifest.csv"
DEFAULT_OUTDIR = Path.cwd() / "results" / "audit" / "ELPD" / "split_review"
SPLITS = ("train", "val", "test")
PREFERENCE = ("test", "val", "train")
REQUIRED_COLUMNS = {"image_path", "text_normalized", "scenario"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el manifest: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not REQUIRED_COLUMNS.issubset(fields):
            raise ValueError(f"Faltan columnas: {sorted(REQUIRED_COLUMNS - fields)}")
        rows = list(reader)

    if not rows:
        raise ValueError("El manifest está vacío")

    paths = set()
    for num, row in enumerate(rows, start=2):
        path = (row.get("image_path") or "").strip()
        split = (row.get("scenario") or "").strip().lower()
        if not path or path in paths:
            raise ValueError(f"Fila {num}: ruta de imagen vacía o duplicada: {path!r}")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"Fila {num}: ruta no válida: {path!r}")
        if split not in SPLITS:
            raise ValueError(f"Fila {num}: partición inesperada {split!r}; esperadas {SPLITS}")
        paths.add(path)
    return rows


def save_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar IDs de matrícula compartidos entre particiones ELPD")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    counts_before = Counter((r["scenario"] or "").strip().lower() for r in rows)

    # El manifest guarda texto normalizado. Usamos la misma clave al comparar,
    # sin inferir ninguna identidad real ni verificar automáticamente etiquetas.
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    empty_labels = []
    for r in rows:
        label = (r.get("text_normalized") or "").strip().upper()
        if label:
            groups[label].append(r)
        else:
            empty_labels.append(r["image_path"])

    conflicts: dict[str, list[dict[str, str]]] = {
        label: members for label, members in groups.items()
        if len({(m["scenario"] or "").strip().lower() for m in members}) > 1
    }
    destination = {
        label: next(split for split in PREFERENCE
                    if any((r["scenario"] or "").strip().lower() == split for r in members))
        for label, members in conflicts.items()
    }

    details = []
    for label, members in sorted(conflicts.items()):
        per_split = Counter((r["scenario"] or "").strip().lower() for r in members)
        dst = destination[label]
        details.append({
            "plate_id_candidate": label,
            "n_images": str(len(members)),
            "train_original": str(per_split["train"]),
            "val_original": str(per_split["val"]),
            "test_original": str(per_split["test"]),
            "split_proposed": dst,
            "n_to_move": str(len(members) - per_split[dst]),
            "image_paths": " | ".join(sorted(m["image_path"] for m in members)),
            "review_status": "needs_human_review",
        })

    proposals = []
    changes = []
    counts_after = Counter()
    transitions = Counter()
    for r in rows:
        label = (r.get("text_normalized") or "").strip().upper()
        original = (r["scenario"] or "").strip().lower()
        proposed = destination.get(label, original)
        conflict = label in conflicts
        status = "proposed_move_needs_review" if proposed != original else (
            "same_id_across_splits_needs_review" if conflict else (
                "missing_label_needs_review" if not label else "no_detected_label_conflict"
            )
        )
        item = {
            "image_path": r["image_path"],
            "text_normalized": label,
            "label_status": r.get("label_status", ""),
            "split_original": original,
            "split_proposed": proposed,
            "cross_split_label": "yes" if conflict else "no",
            "change_proposed": "yes" if original != proposed else "no",
            "review_status": status,
        }
        proposals.append(item)
        counts_after[proposed] += 1
        if proposed != original:
            changes.append(item)
            transitions[(original, proposed)] += 1

    if sum(counts_after.values()) != len(rows):
        raise RuntimeError("Error interno: el número total de imágenes ha cambiado")
    # Validación adicional: ninguna etiqueta no vacía tiene destinos diferentes.
    dst_by_label = defaultdict(set)
    for item in proposals:
        if item["text_normalized"]:
            dst_by_label[item["text_normalized"]].add(item["split_proposed"])
    if any(len(splits) != 1 for splits in dst_by_label.values()):
        raise RuntimeError("La propuesta contiene etiquetas todavía compartidas entre particiones")

    outdir = args.outdir
    fields = ["image_path", "text_normalized", "label_status", "split_original",
              "split_proposed", "cross_split_label", "change_proposed", "review_status"]
    save_csv(outdir / "split_conflicts.csv", [
        "plate_id_candidate", "n_images", "train_original", "val_original", "test_original",
        "split_proposed", "n_to_move", "image_paths", "review_status"
    ], details)
    save_csv(outdir / "split_proposal.csv", fields, proposals)
    save_csv(outdir / "split_changes.csv", fields, changes)

    n_affected = sum(len(members) for members in conflicts.values())
    summary = [
        "=== REVISION DE SOLAPAMIENTOS ELPD ===",
        f"Manifest analizado: {args.manifest}",
        f"SHA256 del manifest: {sha256(args.manifest)}",
        f"Imagenes totales: {len(rows)}",
        f"IDs de matricula compartidos entre splits: {len(conflicts)}",
        f"Imagenes afectadas: {n_affected}",
        f"Cambios propuestos (NO aplicados): {len(changes)}",
        f"Etiquetas vacias pendientes de revision: {len(empty_labels)}",
        "",
        "Particiones originales:",
        *(f"- {split}: {counts_before[split]}" for split in SPLITS),
        "Particiones propuestas:",
        *(f"- {split}: {counts_after[split]}" for split in SPLITS),
        "Movimientos propuestos:",
        *(f"- {src} -> {dst}: {n}" for (src, dst), n in sorted(transitions.items())),
        "",
        "Criterio: test > val > train cuando un identificador aparece en varios splits.",
        "La coincidencia de una etiqueta no prueba identidad de vehiculo.",
        "Las etiquetas del dataset basadas en nombres de archivo requieren validacion.",
        "Ningun archivo del dataset ni manifest.csv ha sido modificado.",
        f"Informes guardados en: {outdir}",
        "=== FINALIZADO ===",
    ]
    summary_path = outdir / "split_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
