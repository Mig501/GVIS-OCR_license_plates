#!/usr/bin/env python3
"""Audita 100 000 HR de LRLPR-26 sin alterar imágenes ni manifiestos.

Optimización: rutas normalizadas sin consultas repetidas al filesystem,
JSON leído una vez por track y verificación de imágenes concurrente (I/O).

Uso desde SLURM:
    python scripts/lrlpr/check_lrlpr_split_integrity_fast.py

Requiere Pillow. Emite progreso cada 10 000 imágenes verificadas.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATASET_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
REQUIRED = {
    'image_path', 'annotation_path', 'text_original', 'text_normalized',
    'track_id', 'scenario', 'plate_layout', 'view_id', 'width', 'height',
    'corners_available', 'source_split', 'dataset',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-root', type=Path, default=DATASET_ROOT / 'processed/LRLPR-26_LPO')
    parser.add_argument('--selected-root', type=Path, default=DATASET_ROOT / 'raw/LRLPR-26/selected_hr')
    parser.add_argument('--report-root', type=Path, default=PROJECT_ROOT / 'results/inspection/LRLPR-26')
    parser.add_argument('--expected-images', type=int, default=100000)
    parser.add_argument('--expected-tracks', type=int, default=20000)
    parser.add_argument('--expected-train', type=int, default=90000)
    parser.add_argument('--expected-val', type=int, default=10000)
    parser.add_argument('--expected-plates', type=int, default=16946)
    parser.add_argument('--workers', type=int, default=4,
                        help='Lecturas concurrentes de imágenes; recomendado 4 con 4 CPUs SLURM')
    parser.add_argument('--batch-size', type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def read_manifest(path: Path, with_split: bool, issue):
    if not path.is_file():
        issue(path.name, 'missing_manifest', str(path))
        return [], []
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        absent = REQUIRED - set(columns)
        if absent:
            issue(path.name, 'missing_columns', ','.join(sorted(absent)))
        if ('split' in columns) != with_split:
            issue(path.name, 'unexpected_split_column', str('split' in columns))
        return columns, list(reader)


def lexical_absolute(base: Path, relative: str) -> Path:
    """Normaliza una ruta SIN resolve(): evita miles de consultas al servidor Ceph.

    La protección es contra escapes léxicos. Se asume que el conjunto autorizado
    y las rutas de manifiestos generadas por el proyecto no contienen symlinks.
    """
    return Path(os.path.abspath(base / relative))


def check_image(task):
    key, image_path, width, height, scenario = task
    try:
        expected_dim = int(width), int(height)
        expected_fmt = 'PNG' if scenario == 'Scenario-A' else 'JPEG'
        with Image.open(image_path) as im:
            found_dim, found_fmt = im.size, im.format
            im.verify()
        errors = []
        if found_dim != expected_dim:
            errors.append(('dimension_mismatch', f'{found_dim} != {expected_dim}'))
        if found_fmt != expected_fmt:
            errors.append(('format_mismatch', f'{found_fmt} != {expected_fmt}'))
        return key, not errors, errors
    except (OSError, ValueError, SyntaxError, KeyError) as exc:
        return key, False, [('image_open_or_decode_failure', str(exc))]


def main() -> int:
    cfg = parse_args()
    if cfg.workers < 1 or cfg.batch_size < 1:
        raise ValueError('workers y batch-size deben ser positivos')
    started = time.monotonic()
    base = cfg.manifest_root.resolve()
    selected_root = cfg.selected_root.resolve()
    report_root = cfg.report_root.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    problems = []
    issue_count = 0

    def issue(where, reason, detail):
        nonlocal issue_count
        issue_count += 1
        if len(problems) < 300:
            problems.append((str(where), str(reason), str(detail)))

    print('=== LRLPR-26: AUDITORIA HR OPTIMIZADA ===', flush=True)
    print(f'Imagenes en paralelo: {cfg.workers}; bloque: {cfg.batch_size}', flush=True)
    print(f'Raiz manifiestos: {base}', flush=True)
    original_file = base / 'manifest.csv'
    original_sha = sha256(original_file) if original_file.is_file() else None
    original_columns, original = read_manifest(original_file, False, issue)
    combined_columns, combined = read_manifest(base / 'split_manifest.csv', True, issue)
    _, train = read_manifest(base / 'train_manifest.csv', True, issue)
    _, val = read_manifest(base / 'val_manifest.csv', True, issue)

    for name, rows, expected in [('original', original, cfg.expected_images),
                                 ('split', combined, cfg.expected_images),
                                 ('train', train, cfg.expected_train),
                                 ('val', val, cfg.expected_val)]:
        if len(rows) != expected:
            issue(name, 'unexpected_row_count', f'{len(rows)} vs {expected}')
    if original_columns and combined_columns and combined_columns != original_columns + ['split']:
        issue('split_manifest.csv', 'unexpected_columns_or_order', str(combined_columns))

    def indexed(rows, name):
        result = {}
        for n, row in enumerate(rows, 2):
            image_key = row.get('image_path', '')
            if not image_key:
                issue(name, 'empty_image_path', f'fila {n}')
            elif image_key in result:
                issue(name, 'duplicate_image_path', image_key)
            else:
                result[image_key] = row
        return result

    orig_map = indexed(original, 'manifest.csv')
    split_map = indexed(combined, 'split_manifest.csv')
    train_map = indexed(train, 'train_manifest.csv')
    val_map = indexed(val, 'val_manifest.csv')
    if set(orig_map) != set(split_map):
        issue('split_manifest.csv', 'path_set_differs_from_original',
              f'faltan={len(set(orig_map)-set(split_map))}, sobran={len(set(split_map)-set(orig_map))}')
    if set(train_map) & set(val_map):
        issue('train/val', 'overlapping_image_paths', str(len(set(train_map) & set(val_map))))
    if set(train_map) | set(val_map) != set(split_map):
        issue('train/val', 'not_exact_partition', 'Unión distinta del manifiesto completo')

    tracks = defaultdict(list)
    plate_splits = defaultdict(set)
    track_splits = defaultdict(set)
    strata = Counter()
    annotations = {}
    image_tasks = []
    for i, row in enumerate(combined, 1):
        image_key = row.get('image_path', '')
        split = row.get('split', '')
        if split not in ('train', 'val'):
            issue(image_key, 'invalid_split', split)
            continue
        original_row = orig_map.get(image_key)
        if original_row is None:
            issue(image_key, 'missing_from_original', '')
        elif any(original_row.get(k, '') != row.get(k, '') for k in original_columns):
            issue(image_key, 'changed_source_fields', '')
        subset_row = (train_map if split == 'train' else val_map).get(image_key)
        if subset_row != row:
            issue(image_key, 'different_subset_row', '')
        if row.get('dataset') != 'LRLPR-26' or row.get('source_split') != 'train':
            issue(image_key, 'wrong_dataset_or_source_split', '')
        track, plate = row.get('track_id', ''), row.get('text_normalized', '')
        if not track or not plate:
            issue(image_key, 'empty_track_or_label', f'{track}, {plate}')
        tracks[track].append(row)
        plate_splits[plate].add(split)
        track_splits[track].add(split)
        strata[split, row.get('scenario'), row.get('plate_layout')] += 1

        file_path = lexical_absolute(base, image_key)
        try:
            file_path.relative_to(selected_root)
        except ValueError:
            issue(image_key, 'image_outside_selected_root', str(file_path))
        else:
            image_tasks.append((image_key, file_path, row.get('width'), row.get('height'), row.get('scenario')))

        annotation_key = row.get('annotation_path', '')
        if annotation_key not in annotations:
            annotation_path = lexical_absolute(base, annotation_key)
            try:
                annotation_path.relative_to(selected_root)
                annotations[annotation_key] = json.loads(annotation_path.read_text(encoding='utf-8'))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                issue(image_key, 'invalid_annotation', str(exc))
                annotations[annotation_key] = {}
        annotation = annotations[annotation_key]
        if annotation.get('plate_text') != row.get('text_original'):
            issue(image_key, 'annotation_plate_text_mismatch', '')
        if re.sub(r'[^A-Z0-9]', '', str(annotation.get('plate_text', '')).upper()) != plate:
            issue(image_key, 'annotation_normalized_text_mismatch', '')
        if annotation.get('plate_layout') != row.get('plate_layout'):
            issue(image_key, 'annotation_layout_mismatch', '')
        corners = annotation.get('corners', {})
        has_corners = isinstance(corners, dict) and file_path.name in corners
        if row.get('corners_available') != ('yes' if has_corners else 'no'):
            issue(image_key, 'corner_flag_mismatch', row.get('corners_available'))
        if (row.get('scenario') == 'Scenario-A') != has_corners:
            issue(image_key, 'unexpected_corners_for_scenario', '')
        if i % 20000 == 0:
            print(f'Metadatos/JSON revisados: {i}/{len(combined)}', flush=True)

    print(f'Metadatos revisados; comenzando a verificar {len(image_tasks)} imagenes...', flush=True)
    verified = 0
    image_valid = 0
    # Lotes pequeños: memoria acotada y progreso visible en el log.
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        for offset in range(0, len(image_tasks), cfg.batch_size):
            batch = image_tasks[offset:offset + cfg.batch_size]
            for image_key, valid, errors in executor.map(check_image, batch):
                verified += 1
                image_valid += int(valid)
                for code, detail in errors:
                    issue(image_key, code, detail)
            if verified % 10000 == 0 or verified == len(image_tasks):
                print(f'Imagenes verificadas: {verified}/{len(image_tasks)} | '
                      f'{(time.monotonic()-started)/60:.1f} min', flush=True)

    if len(tracks) != cfg.expected_tracks:
        issue('tracks', 'unexpected_track_count', f'{len(tracks)} vs {cfg.expected_tracks}')
    if len(plate_splits) != cfg.expected_plates:
        issue('plates', 'unexpected_label_count', f'{len(plate_splits)} vs {cfg.expected_plates}')
    for track, rows in tracks.items():
        if len(rows) != 5 or len({r.get('view_id') for r in rows}) != 5:
            issue(track, 'invalid_track_views', str(len(rows)))
        if len({r.get('text_normalized') for r in rows}) != 1:
            issue(track, 'inconsistent_track_label', '')
        if len(track_splits[track]) != 1:
            issue(track, 'track_leakage', ','.join(sorted(track_splits[track])))
    plate_leaks = sum(len(parts) > 1 for parts in plate_splits.values())
    track_leaks = sum(len(parts) > 1 for parts in track_splits.values())
    if plate_leaks:
        issue('train/val', 'plate_leakage', plate_leaks)
    if track_leaks:
        issue('train/val', 'track_leakage_count', track_leaks)
    if original_sha is not None and sha256(original_file) != original_sha:
        issue('manifest.csv', 'original_sha_changed', '')

    issues_path = report_root / 'hr_split_integrity_issues.csv'
    with issues_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['location', 'issue', 'details'])
        writer.writerows(problems)
    current_sha = sha256(original_file) if original_file.is_file() else None
    lines = [
        '=== LRLPR-26: COMPROBACION DE SPLITS E INTEGRIDAD HR ===',
        f'Imagenes en manifest: {len(original)}',
        f'Imagenes train: {len(train)}',
        f'Imagenes val: {len(val)}',
        f'Imagenes abiertas/verificadas: {verified}',
        f'Imagenes sin incidencias visuales/tecnicas: {image_valid}',
        f'Tracks: {len(tracks)}',
        f'Matriculas distintas: {len(plate_splits)}',
        f'Matriculas compartidas entre train y val: {plate_leaks}',
        f'Tracks compartidos entre train y val: {track_leaks}',
        f'Anotaciones JSON revisadas: {len(annotations)}',
        'Distribucion por escenario/layout (imagenes train / val):',
    ]
    for scenario in ['Scenario-A', 'Scenario-B']:
        for layout in ['Brazilian', 'Mercosur']:
            lines.append(f'- {scenario}/{layout}: {strata["train", scenario, layout]} / {strata["val", scenario, layout]}')
    lines.extend([
        f'Incidencias: {issue_count}',
        f'SHA256 original intacto: {"SI" if original_sha is not None and current_sha == original_sha else "NO"}',
        f'Tiempo total: {(time.monotonic()-started)/60:.1f} min',
        'No se han modificado imagenes, anotaciones ni manifiestos.',
        'ESTADO: APTO PARA ENTRENAMIENTO/VALIDACION' if issue_count == 0 else 'ESTADO: REVISAR INCIDENCIAS',
        '=== FINALIZADO ===',
    ])
    summary_path = report_root / 'hr_split_integrity_summary.txt'
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Informe: {summary_path}', flush=True)
    print(f'Incidencias: {issues_path}', flush=True)
    return 0 if issue_count == 0 else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
