#!/usr/bin/env python3
"""Audita los manifests HR de LRLPR-26 y los archivos referenciados, sin modificarlos.

Uso habitual (en un nodo SLURM):
    python scripts/lrlpr/check_lrlpr_split_integrity.py

No crea nuevas particiones, no mueve imágenes ni modifica anotaciones.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

DATASET_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
REQUIRED = {
    'image_path', 'annotation_path', 'text_normalized', 'track_id', 'scenario',
    'plate_layout', 'view_id', 'width', 'height', 'corners_available',
    'source_split', 'dataset',
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-root', type=Path, default=DATASET_ROOT / 'processed/LRLPR-26_LPO')
    p.add_argument('--selected-root', type=Path, default=DATASET_ROOT / 'raw/LRLPR-26/selected_hr')
    p.add_argument('--report-root', type=Path, default=PROJECT_ROOT / 'results/inspection/LRLPR-26')
    p.add_argument('--expected-images', type=int, default=100000)
    p.add_argument('--expected-tracks', type=int, default=20000)
    p.add_argument('--expected-train', type=int, default=90000)
    p.add_argument('--expected-val', type=int, default=10000)
    p.add_argument('--expected-plates', type=int, default=16946)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_manifest(path: Path, must_have_split: bool, issue) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        issue(path.name, 'missing_manifest', str(path))
        return [], []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = REQUIRED - set(fields)
        if missing:
            issue(path.name, 'missing_columns', ','.join(sorted(missing)))
        if ('split' in fields) != must_have_split:
            issue(path.name, 'unexpected_split_column', str('split' in fields))
        rows = list(reader)
    return fields, rows


def main() -> int:
    a = parse_args()
    base = a.manifest_root.resolve()
    src_root = a.selected_root.resolve()
    report = a.report_root.resolve()
    report.mkdir(parents=True, exist_ok=True)
    problems: list[tuple[str, str, str]] = []
    issue_count = 0

    def issue(where: str, reason: str, detail: str) -> None:
        nonlocal issue_count
        issue_count += 1
        if len(problems) < 300:
            problems.append((where, reason, str(detail)))

    print('=== LRLPR-26: COMPROBACION DE SPLITS E INTEGRIDAD HR ===', flush=True)
    print(f'Manifiestos: {base}', flush=True)
    original_path = base / 'manifest.csv'
    initial_sha = sha256(original_path) if original_path.is_file() else None
    original_fields, original = read_manifest(original_path, False, issue)
    combined_fields, combined = read_manifest(base / 'split_manifest.csv', True, issue)
    _, train = read_manifest(base / 'train_manifest.csv', True, issue)
    _, val = read_manifest(base / 'val_manifest.csv', True, issue)

    for label, rows, expected in [('original', original, a.expected_images),
                                  ('split', combined, a.expected_images),
                                  ('train', train, a.expected_train),
                                  ('val', val, a.expected_val)]:
        if len(rows) != expected:
            issue(label, 'unexpected_row_count', f'{len(rows)}; esperadas {expected}')
    if combined_fields and original_fields and combined_fields != original_fields + ['split']:
        issue('split_manifest.csv', 'unexpected_columns_or_order', str(combined_fields))

    def unique_by_path(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
        indexed: dict[str, dict[str, str]] = {}
        for i, row in enumerate(rows, 2):
            path = row.get('image_path', '')
            if not path:
                issue(label, 'empty_image_path', f'fila {i}')
            elif path in indexed:
                issue(label, 'duplicate_image_path', path)
            else:
                indexed[path] = row
        return indexed

    orig_map = unique_by_path(original, 'manifest.csv')
    split_map = unique_by_path(combined, 'split_manifest.csv')
    train_map = unique_by_path(train, 'train_manifest.csv')
    val_map = unique_by_path(val, 'val_manifest.csv')
    if set(orig_map) != set(split_map):
        issue('split_manifest.csv', 'path_set_differs_from_original',
              f'faltan={len(set(orig_map)-set(split_map))}, sobran={len(set(split_map)-set(orig_map))}')
    if set(train_map) & set(val_map):
        issue('train/val', 'overlapping_image_paths', f'{len(set(train_map)&set(val_map))}')
    if set(train_map) | set(val_map) != set(split_map):
        issue('train/val', 'not_exact_partition',
              f'union={len(set(train_map)|set(val_map))}, split={len(split_map)}')

    tracks: dict[str, list[dict[str, str]]] = defaultdict(list)
    label_splits: dict[str, set[str]] = defaultdict(set)
    track_splits: dict[str, set[str]] = defaultdict(set)
    strata = Counter()
    img_correct = 0
    annotations: dict[str, dict] = {}
    for n, row in enumerate(combined, 1):
        image_key = row.get('image_path', '')
        split = row.get('split', '')
        if split not in {'train', 'val'}:
            issue(image_key, 'invalid_split', split)
            continue
        orig = orig_map.get(image_key)
        if orig is not None and any(orig.get(k, '') != row.get(k, '') for k in original_fields):
            issue(image_key, 'changed_source_fields', 'Contenido distinto del manifest original')
        subset_row = (train_map if split == 'train' else val_map).get(image_key)
        if subset_row != row:
            issue(image_key, 'different_subset_row', 'No coincide con train/val_manifest')
        if row.get('dataset') != 'LRLPR-26' or row.get('source_split') != 'train':
            issue(image_key, 'wrong_dataset_or_source_split', 'Revisar procedencia')
        track = row.get('track_id', '')
        plate = row.get('text_normalized', '')
        if not track or not plate:
            issue(image_key, 'empty_track_or_label', f'{track}, {plate}')
        tracks[track].append(row)
        label_splits[plate].add(split)
        track_splits[track].add(split)
        strata[(split, row.get('scenario'), row.get('plate_layout'))] += 1

        candidate = (base / image_key).resolve()
        try:
            candidate.relative_to(src_root)
        except ValueError:
            issue(image_key, 'image_outside_selected_root', str(candidate))
            continue
        if not candidate.is_file():
            issue(image_key, 'missing_image', str(candidate))
            continue
        try:
            with Image.open(candidate) as im:
                dims = (int(row['width']), int(row['height']))
                if im.size != dims:
                    issue(image_key, 'dimension_mismatch', f'{im.size} != {dims}')
                expected_fmt = 'PNG' if row.get('scenario') == 'Scenario-A' else 'JPEG'
                if im.format != expected_fmt:
                    issue(image_key, 'format_mismatch', f'{im.format} != {expected_fmt}')
                im.verify()  # Verifica integridad de encabezado/estructura sin transformar imagen
            img_correct += 1
        except (OSError, ValueError, KeyError, SyntaxError) as exc:
            issue(image_key, 'image_open_or_decode_failure', str(exc))

        ann_key = row.get('annotation_path', '')
        ann_path = (base / ann_key).resolve()
        try:
            ann_path.relative_to(src_root)
        except ValueError:
            issue(image_key, 'annotation_outside_selected_root', str(ann_path))
            continue
        if ann_key not in annotations:
            try:
                annotations[ann_key] = json.loads(ann_path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                issue(image_key, 'invalid_annotation', str(exc))
                annotations[ann_key] = {}
        ann = annotations[ann_key]
        if ann.get('plate_text') != row.get('text_original'):
            issue(image_key, 'annotation_plate_text_mismatch', 'Texto distinto del JSON')
        if re.sub(r'[^A-Z0-9]', '', str(ann.get('plate_text', '')).upper()) != plate:
            issue(image_key, 'annotation_normalized_text_mismatch', 'Texto normalizado distinto')
        if ann.get('plate_layout') != row.get('plate_layout'):
            issue(image_key, 'annotation_layout_mismatch', 'Layout distinto')
        corners = ann.get('corners', {})
        has_corners = isinstance(corners, dict) and candidate.name in corners
        if row.get('corners_available') != ('yes' if has_corners else 'no'):
            issue(image_key, 'corner_flag_mismatch', row.get('corners_available', ''))
        if (row.get('scenario') == 'Scenario-A') != has_corners:
            issue(image_key, 'unexpected_corners_for_scenario', row.get('scenario', ''))
        if n % 20000 == 0:
            print(f'Imagenes comprobadas: {n}/{len(combined)}', flush=True)

    if len(tracks) != a.expected_tracks:
        issue('tracks', 'unexpected_track_count', f'{len(tracks)} != {a.expected_tracks}')
    if len(label_splits) != a.expected_plates:
        issue('plates', 'unexpected_label_count', f'{len(label_splits)} != {a.expected_plates}')
    for track, rows in tracks.items():
        if len(rows) != 5 or len({r.get('view_id') for r in rows}) != 5:
            issue(track, 'invalid_track_views', f'{len(rows)} imagenes')
        if len({r.get('text_normalized') for r in rows}) != 1:
            issue(track, 'inconsistent_track_label', 'Múltiples etiquetas')
        if len(track_splits[track]) != 1:
            issue(track, 'track_leakage', ','.join(sorted(track_splits[track])))
    plate_leaks = sum(len(x) > 1 for x in label_splits.values())
    track_leaks = sum(len(x) > 1 for x in track_splits.values())
    if plate_leaks:
        issue('train/val', 'plate_leakage', str(plate_leaks))
    if track_leaks:
        issue('train/val', 'track_leakage_count', str(track_leaks))
    if initial_sha is not None and sha256(original_path) != initial_sha:
        issue('manifest.csv', 'original_sha_changed', 'El original cambió durante la verificación')

    report_csv = report / 'hr_split_integrity_issues.csv'
    with report_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['location', 'issue', 'details'])
        writer.writerows(problems)
    lines = [
        '=== LRLPR-26: COMPROBACION DE SPLITS E INTEGRIDAD HR ===',
        f'Imagenes en manifest: {len(original)}',
        f'Imagenes train: {len(train)}',
        f'Imagenes val: {len(val)}',
        f'Imagenes abiertas/verificadas: {img_correct}',
        f'Tracks: {len(tracks)}',
        f'Matriculas distintas: {len(label_splits)}',
        f'Matriculas compartidas entre train y val: {plate_leaks}',
        f'Tracks compartidos entre train y val: {track_leaks}',
        f'Anotaciones JSON revisadas: {len(annotations)}',
        'Distribucion por escenario/layout (imagenes train / val):',
    ]
    for scenario in ['Scenario-A', 'Scenario-B']:
        for layout in ['Brazilian', 'Mercosur']:
            lines.append(f'- {scenario}/{layout}: '
                         f'{strata[("train", scenario, layout)]} / {strata[("val", scenario, layout)]}')
    lines.extend([
        f'Incidencias: {issue_count}',
        f'SHA256 original intacto: {"SI" if initial_sha is not None and sha256(original_path) == initial_sha else "NO"}',
        'No se han modificado imagenes, anotaciones ni manifests.',
        'ESTADO: APTO PARA ENTRENAMIENTO/VALIDACION' if issue_count == 0 else 'ESTADO: REVISAR INCIDENCIAS',
        '=== FINALIZADO ===',
    ])
    summary = report / 'hr_split_integrity_summary.txt'
    summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Informe: {summary}', flush=True)
    print(f'Incidencias: {report_csv}', flush=True)
    return 0 if issue_count == 0 else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'ERROR inesperado: {exc}', file=sys.stderr)
        sys.exit(1)
