#!/usr/bin/env python3
"""Integra tres recortes de ELPD validados manualmente en ELPD_LPO.

Ubicacion en VS Code: scripts/elpd/integrate_elp_crops.py
Ejecucion: python scripts/elpd/integrate_elp_crops.py [--dry-run]

Precondiciones:
- Se han generado los candidatos con crop_elp_pending.py.
- scripts/elpd/elp_review_decisions.csv contiene la aprobacion visual de los 3.
- Los originales siguen intactos en processed/ELPD_LPO/images/.

El dataset ZIP original no se toca. Antes de reemplazar cada escena se crea una
copia sin perdidas en processed/ELPD_LPO/backups/original_full_scenes/.
Se guarda el manifest original y un informe de la operacion.
"""

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

DATA_ROOT = Path(os.environ.get("OCR_DATA_ROOT", "/home/miguel.sanchez/datasets/ocr_matriculas"))
PROJECT_ROOT = Path.cwd()
DATASET_DIR = DATA_ROOT / "processed" / "ELPD_LPO"
MANIFEST = DATASET_DIR / "manifest.csv"
CANDIDATES = PROJECT_ROOT / "results/inspection/ELPD/crop_candidates"
DECISIONS = PROJECT_ROOT / "scripts/elpd/elp_review_decisions.csv"
BACKUPS = DATASET_DIR / "backups/original_full_scenes"
REPORT = PROJECT_ROOT / "results/audit/ELPD/crop_integration_log.csv"

# Izquierda, arriba, derecha, abajo (derecha y abajo exclusivos).
# Coordenadas aprobadas visualmente sobre las imagenes originales de 1280x720.
CROPS = {
    "test__PH20CNP.png": ("test", "PH20CNP", (676, 392, 785, 424)),
    "train__SB90TZI.png": ("train", "SB90TZI", (494, 463, 675, 519)),
    "train__TR05RBG.png": ("train", "TR05RBG", (710, 439, 944, 515)),
}
SOURCE_SIZE = (1280, 720)
EXPECTED_DATASET_SIZE = 735
MANIFEST_REQUIRED = {"image_path", "text_normalized", "scenario", "width", "height", "crop_method", "label_status"}


def read_csv(path: Path, required: set[str]):
    if not path.is_file():
        raise FileNotFoundError(f"Falta archivo requerido: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not required.issubset(fields):
            raise ValueError(f"Columnas ausentes en {path}: {sorted(required - set(fields))}")
        rows = list(reader)
    return fields, rows


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, target: Path):
    # Un temporal en el mismo directorio permite os.replace atomico.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate():
    fields, rows = read_csv(MANIFEST, MANIFEST_REQUIRED)
    if len(rows) != EXPECTED_DATASET_SIZE:
        raise ValueError(f"Se esperaban {EXPECTED_DATASET_SIZE} entradas en manifest.csv, hay {len(rows)}")
    by_path = {r["image_path"]: r for r in rows}
    if len(by_path) != len(rows):
        raise ValueError("Hay rutas duplicadas en manifest.csv")

    _, decisions = read_csv(DECISIONS, {"image_path", "crop_status", "label_review_status", "verified_text"})
    d_index = {r["image_path"]: r for r in decisions}
    if len(d_index) != len(decisions):
        raise ValueError("Hay decisiones duplicadas en el CSV")

    prepared = []
    for name, (split, label, box) in CROPS.items():
        relpath = f"images/{name}"
        row = by_path.get(relpath)
        if row is None:
            raise ValueError(f"No esta el registro {relpath} en manifest.csv")
        if row["scenario"] != split or row["text_normalized"] != label:
            raise ValueError(f"Particion/etiqueta inesperada en {relpath}: {row['scenario']} / {row['text_normalized']}")

        decision = d_index.get(relpath)
        if not decision or any((decision.get(k) or "").strip() != v for k, v in (
            ("crop_status", "valid_crop"), ("label_review_status", "verified"), ("verified_text", label)
        )):
            raise ValueError(f"Falta la aprobacion visual en elp_review_decisions.csv para {relpath}")

        source = DATASET_DIR / relpath
        candidate = CANDIDATES / name
        backup = BACKUPS / name
        if not source.is_file() or not candidate.is_file():
            raise FileNotFoundError(f"Falta origen o candidato: {source} | {candidate}")

        with Image.open(candidate) as candidate_img:
            cropped = candidate_img.convert("RGB")
        expected_size = (box[2] - box[0], box[3] - box[1])
        if cropped.size != expected_size:
            raise ValueError(f"Candidato con dimensiones incorrectas: {name}, {cropped.size}")

        with Image.open(source) as current_img:
            current = current_img.convert("RGB")
        already_integrated = current.size == expected_size and ImageChops.difference(current, cropped).getbbox() is None
        if current.size != SOURCE_SIZE and not already_integrated:
            raise ValueError(f"Origen con dimensiones inesperadas: {name}: {current.size}")
        if already_integrated and (row["width"] != str(expected_size[0]) or row["height"] != str(expected_size[1])):
            raise ValueError(f"Imagen ya recortada pero manifest desactualizado: {name}; revision manual necesaria")
        if not already_integrated and (row["width"] != "1280" or row["height"] != "720"):
            raise ValueError(f"Dimensiones del manifest inesperadas para {name}")

        # Si ya hubo una integracion anterior, verificamos contra la copia original.
        original = backup if already_integrated else source
        if not original.is_file():
            raise FileNotFoundError(f"Falta copia de la imagen original: {original}")
        with Image.open(original) as original_img:
            if original_img.size != SOURCE_SIZE:
                raise ValueError(f"El original no es 1280x720: {original}")
            extracted = original_img.convert("RGB").crop(box)
            if ImageChops.difference(extracted, cropped).getbbox() is not None:
                raise ValueError(f"El candidato NO coincide pixel a pixel con el recorte original: {name}")
        if backup.is_file() and not already_integrated and sha256(backup) != sha256(source):
            raise ValueError(f"Existe una copia de seguridad distinta al origen: {backup}")
        prepared.append((name, row, source, candidate, backup, box, already_integrated))
    return fields, rows, prepared


def main():
    parser = argparse.ArgumentParser(description="Integrar 3 recortes verificados en ELPD_LPO")
    parser.add_argument("--dry-run", action="store_true", help="Comprobar todo sin realizar cambios")
    args = parser.parse_args()

    print("=== INTEGRACION DE RECORTES ELPD LPO ===")
    fields, rows, prepared = validate()
    print(f"Filas de manifest verificadas: {len(rows)}")
    print(f"Recortes aprobados comprobados: {len(prepared)}")
    if args.dry_run:
        print("MODO DRY-RUN: no se han modificado imagenes ni manifest.csv")
        return

    # Crear backups completos antes de tocar ninguna imagen del dataset.
    BACKUPS.mkdir(parents=True, exist_ok=True)
    manifest_backup = BACKUPS / "manifest_before_manual_crops.csv"
    if not manifest_backup.is_file():
        atomic_copy(MANIFEST, manifest_backup)
    else:
        old_fields, old_rows = read_csv(manifest_backup, MANIFEST_REQUIRED)
        if len(old_rows) != EXPECTED_DATASET_SIZE or set(old_fields) != set(fields):
            raise ValueError("El backup de manifest.csv es incompatible. Revisar antes de continuar.")
    for name, _row, source, _candidate, backup, _box, done in prepared:
        if not backup.is_file():
            if done:
                raise ValueError(f"Sin backup original para imagen ya integrada: {name}")
            atomic_copy(source, backup)
        print(f"Backup disponible: {backup.name}")

    changed = []
    try:
        for name, row, source, candidate, backup, box, done in prepared:
            if not done:
                atomic_copy(candidate, source)
                changed.append((source, backup))
            row["width"] = str(box[2] - box[0])
            row["height"] = str(box[3] - box[1])
            row["crop_method"] = "manual_bbox_from_full_scene"
            row["label_status"] = "visually_verified"
            print(f"{'Ya integrado' if done else 'Integrado'}: {name} | {row['width']}x{row['height']} | etiqueta {row['text_normalized']}")

        # El manifest se actualiza solo despues de integrar todas las imagenes.
        atomic_csv(MANIFEST, fields, rows)
    except Exception:
        for source, backup in reversed(changed):
            atomic_copy(backup, source)
        print("ERROR: imagenes restauradas desde el backup; comprobar el estado del manifest.")
        raise

    report_rows = []
    for name, row, source, candidate, backup, box, _done in prepared:
        report_rows.append({
            "image_path": row["image_path"],
            "text_verified": row["text_normalized"],
            "split": row["scenario"],
            "bbox": ",".join(map(str, box)),
            "original_sha256": sha256(backup),
            "crop_sha256": sha256(source),
            "crop_width": row["width"],
            "crop_height": row["height"],
            "crop_method": row["crop_method"],
        })
    atomic_csv(REPORT, list(report_rows[0]), report_rows)
    print("\n=== RESULTADO ===")
    print("Recortes integrados/verificados: 3")
    print(f"Imagenes totales: {len(rows)}")
    print(f"Manifest actualizado: {MANIFEST}")
    print(f"Backups preservados: {BACKUPS}")
    print(f"Informe de integracion: {REPORT}")
    print("ZIP de origen intacto. Las otras 732 imagenes no se han modificado.")
    print("=== FINALIZADO ===")


if __name__ == "__main__":
    main()
