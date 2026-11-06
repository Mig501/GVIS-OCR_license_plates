#!/usr/bin/env python3
"""Preflight for zero-shot OCR baselines on the audited ELPD test subset.

Read-only checks, designed to run in the GVIS SLURM/Apptainer environment.
Does not download models or execute inference.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
from pathlib import Path
from collections import Counter

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
MANIFEST = PROJECT / 'results/audit/ELPD/final_eval/verified_test_manifest.csv'
IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
OUT = PROJECT / 'results/baselines/ELPD/preflight.txt'


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    report = ['=== GVIS: PREFLIGHT BASELINES ELPD ===']
    errors: list[str] = []
    report.append(f'Manifest: {MANIFEST}')
    report.append(f'Raiz de imagenes: {IMAGE_ROOT}')

    if not MANIFEST.is_file():
        errors.append('No existe verified_test_manifest.csv. Descargar/regenerar con la tarea de ELPD.')
        rows = []
    else:
        with MANIFEST.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            required = {'image_path', 'text_normalized', 'gt_review_status', 'strict_gt_eligible'}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                errors.append(f'Columnas obligatorias ausentes: {sorted(missing)}')
            rows = list(reader)
        report.append('SHA256 manifest: ' + hashlib.sha256(MANIFEST.read_bytes()).hexdigest())

    report.append(f'Filas: {len(rows)} (esperadas: 80)')
    if len(rows) != 80:
        errors.append('El manifiesto debe contener 80 imágenes del test ELPD auditado.')

    statuses = Counter(r.get('gt_review_status', '') for r in rows)
    report.append(f'GT: verified={statuses["verified"]}, corrected={statuses["corrected"]}, otros={len(rows)-statuses["verified"]-statuses["corrected"]}')
    if statuses != {'verified': 76, 'corrected': 4}:
        errors.append('Los estados de GT no corresponden a las 76 verificadas + 4 corregidas.')

    if len({r.get('image_path', '') for r in rows}) != len(rows):
        errors.append('Hay rutas de imagen repetidas.')

    missing_files: list[str] = []
    invalid_rows: list[str] = []
    for i, r in enumerate(rows, start=2):
        rel = r.get('image_path', '').strip()
        text = r.get('text_normalized', '').strip()
        eligible = r.get('strict_gt_eligible', '').strip().lower()
        if not rel or not text or eligible != 'yes':
            invalid_rows.append(f'fila {i}: image_path/text_normalized/strict_gt_eligible incorrecto')
        # Never trust manifest path traversal; enforce paths under the dataset's image directory.
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts or not rel.startswith('images/'):
            invalid_rows.append(f'fila {i}: ruta relativa no permitida {rel!r}')
            continue
        if not (IMAGE_ROOT / p).is_file():
            missing_files.append(rel)
    report.append(f'Imagenes localizadas: {len(rows)-len(missing_files)} / {len(rows)}')
    report.append(f'Rutas o GT incorrectos: {len(invalid_rows)}')
    if missing_files:
        errors.append(f'Faltan {len(missing_files)} imágenes; ejemplos: {missing_files[:3]}')
    if invalid_rows:
        errors.append(f'Filas inválidas: {invalid_rows[:3]}')

    report.append('\nDependencias de PARSeq:')
    for key, package in [('torch', 'torch'), ('torchvision', 'torchvision'),
                         ('pytorch_lightning', 'pytorch-lightning'), ('timm', 'timm'),
                         ('PIL', 'Pillow')]:
        version = package_version(package)
        report.append(f'- {key}: {version or "NO INSTALADO"}')
        if version is None:
            errors.append(f'Dependencia pendiente para PARSeq: {package}')
    try:
        import torch
        report.append(f'CUDA disponible: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            report.append(f'GPU: {torch.cuda.get_device_name(0)}')
        else:
            errors.append('No hay GPU CUDA visible. Ejecutar usando GVIS: Ejecutar en GPU.')
    except Exception as exc:
        errors.append(f'No se pudo importar torch: {type(exc).__name__}: {exc}')

    report.append('\nEstado: ' + ('PENDIENTE - revisar incidencias' if errors else 'LISTO PARA INTEGRAR PARSEQ'))
    report.extend('ERROR: ' + e for e in errors)
    report.append('Preflight de solo lectura: no descarga modelos ni modifica datasets.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))
    print(f'Informe: {OUT}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
