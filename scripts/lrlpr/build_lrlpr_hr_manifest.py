#!/usr/bin/env python3
"""Construye un manifiesto OCR a partir de las 100 000 imágenes HR de LRLPR-26.

No copia ni transforma imágenes, no crea particiones y no cambia anotaciones.
Las rutas image_path/annotation_path son relativas a la carpeta del manifiesto.

Uso en el proyecto:
    python scripts/lrlpr/build_lrlpr_hr_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
SCENARIOS = ('Scenario-A', 'Scenario-B')
LAYOUTS = ('Brazilian', 'Mercosur')
CORNER_NAMES = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
FIELDNAMES = [
    'image_path', 'source_path', 'text_original', 'text_normalized',
    'dataset', 'track_id', 'vehicle_id', 'scenario', 'view_id',
    'label_status', 'crop_method', 'width', 'height',
    'plate_layout', 'annotation_path', 'corners_available', 'source_split',
]


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    try:
        temp.write_text(content, encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def relative_to_manifest(target: Path, manifest_root: Path) -> str:
    return Path(os.path.relpath(target, start=manifest_root)).as_posix()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-root', type=Path,
        default=DATA_ROOT / 'raw/LRLPR-26/selected_hr',
        help='Raíz extraída con train/Scenario-A y train/Scenario-B')
    parser.add_argument('--manifest-root', type=Path,
        default=DATA_ROOT / 'processed/LRLPR-26_LPO',
        help='Directorio donde se escribirá manifest.csv')
    parser.add_argument('--report-root', type=Path,
        default=PROJECT_ROOT / 'results/inspection/LRLPR-26',
        help='Directorio para informes pequeños')
    parser.add_argument('--expected-tracks', type=int, default=20000)
    parser.add_argument('--expected-images', type=int, default=100000)
    return parser.parse_args()


def build(args: argparse.Namespace) -> None:
    selected = args.selected_root.resolve()
    manifest_root = args.manifest_root.resolve()
    report_root = args.report_root.resolve()
    print('=== LRLPR-26: MANIFIESTO DE IMAGENES HR ===', flush=True)
    print(f'Origen: {selected}')
    print(f'Manifiesto: {manifest_root / "manifest.csv"}')

    if not (selected / 'train').is_dir():
        raise FileNotFoundError(f'No existe la raíz train/: {selected / "train"}')

    rows: list[dict[str, object]] = []
    label_tracks: dict[str, set[str]] = defaultdict(set)
    label_scenarios: dict[str, set[str]] = defaultdict(set)
    label_layouts: dict[str, set[str]] = defaultdict(set)
    scenario_counts: Counter[str] = Counter()
    layout_tracks: Counter[tuple[str, str]] = Counter()
    errors: list[str] = []
    tracks_seen = 0

    for scenario in SCENARIOS:
        for layout_dir in LAYOUTS:
            folder = selected / 'train' / scenario / layout_dir
            if not folder.is_dir():
                errors.append(f'Falta carpeta {folder}')
                continue
            track_dirs = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith('track_'))
            print(f'Procesando {scenario}/{layout_dir}: {len(track_dirs)} tracks', flush=True)
            for track in track_dirs:
                tracks_seen += 1
                track_id = f'{scenario}/{layout_dir}/{track.name}'
                annotation_path = track / 'annotations.json'
                try:
                    annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
                    text_original = annotation['plate_text']
                    plate_layout = annotation['plate_layout']
                    corners = annotation['corners']
                    if not isinstance(text_original, str) or not text_original.strip():
                        raise ValueError('plate_text vacío/no textual')
                    if plate_layout != layout_dir:
                        raise ValueError(f'plate_layout {plate_layout!r} distinto de {layout_dir!r}')
                    if not isinstance(corners, dict):
                        raise ValueError('corners no es un diccionario')
                    text_normalized = re.sub(r'[^A-Z0-9]', '', text_original.upper())
                    if not text_normalized:
                        raise ValueError('etiqueta normalizada vacía')
                    hr_files = sorted(track.glob('hr-*'))
                    if len(hr_files) != 5:
                        raise ValueError(f'{len(hr_files)} imágenes HR (se esperaban 5)')
                    names = {p.name for p in hr_files}
                    # Scenario-A usa PNG y Scenario-B JPG: NO imponer una extensión única.
                    expected_ext = '.png' if scenario == 'Scenario-A' else '.jpg'
                    expected_names = {f'hr-{i:03d}{expected_ext}' for i in range(1, 6)}
                    if names != expected_names:
                        raise ValueError(f'Nombres HR inesperados: {sorted(names)}; esperados: {sorted(expected_names)}')
                    if scenario == 'Scenario-A' and not all(name in corners for name in names):
                        raise ValueError('Faltan coordenadas para alguna imagen HR')
                    if scenario == 'Scenario-B' and corners:
                        raise ValueError('Scenario-B tiene corners inesperados, revisar antes de continuar')

                    track_rows: list[dict[str, object]] = []
                    for image_file in hr_files:
                        with Image.open(image_file) as im:
                            width, height = im.size
                            expected_format = 'PNG' if scenario == 'Scenario-A' else 'JPEG'
                            if im.format != expected_format or width <= 0 or height <= 0:
                                raise ValueError(f'Formato o dimensiones inválidas: {image_file.name}')
                        available = image_file.name in corners
                        if available:
                            c = corners[image_file.name]
                            if not isinstance(c, dict) or any(
                                not isinstance(c.get(k), (list, tuple)) or len(c[k]) != 2
                                or not all(isinstance(z, (int, float)) for z in c[k])
                                for k in CORNER_NAMES
                            ):
                                raise ValueError(f'Esquinas inválidas para {image_file.name}')
                        track_rows.append({
                            'image_path': relative_to_manifest(image_file, manifest_root),
                            'source_path': image_file.relative_to(selected).as_posix(),
                            'text_original': text_original,
                            'text_normalized': text_normalized,
                            'dataset': 'LRLPR-26',
                            'track_id': track_id,
                            'vehicle_id': '',  # No inferimos identidad de vehículo
                            'scenario': scenario,
                            'view_id': image_file.stem.removeprefix('hr-'),
                            'label_status': 'annotation_unverified',
                            'crop_method': 'pre_cropped_original',
                            'width': width,
                            'height': height,
                            'plate_layout': plate_layout,
                            'annotation_path': relative_to_manifest(annotation_path, manifest_root),
                            'corners_available': 'yes' if available else 'no',
                            'source_split': 'train',
                        })
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, UnidentifiedImageError) as exc:
                    errors.append(f'{track_id}: {exc}')
                    continue
                rows.extend(track_rows)
                scenario_counts[scenario] += len(track_rows)
                layout_tracks[(scenario, layout_dir)] += 1
                label_tracks[text_normalized].add(track_id)
                label_scenarios[text_normalized].add(scenario)
                label_layouts[text_normalized].add(layout_dir)

    if tracks_seen != args.expected_tracks:
        errors.append(f'Tracks: {tracks_seen} (esperados {args.expected_tracks})')
    if len(rows) != args.expected_images:
        errors.append(f'Imágenes: {len(rows)} (esperadas {args.expected_images})')
    if errors:
        print(f'ERROR: {len(errors)} incidencias. No se escribe el manifiesto.', file=sys.stderr)
        for problem in errors[:25]:
            print(f'- {problem}', file=sys.stderr)
        raise RuntimeError('Validación fallida. Revisar errores sin alterar datos originales.')

    collisions = [
        {
            'text_normalized': label,
            'num_tracks': len(tracks),
            'track_ids': ';'.join(sorted(tracks)),
            'scenarios': ';'.join(sorted(label_scenarios[label])),
            'plate_layouts': ';'.join(sorted(label_layouts[label])),
        }
        for label, tracks in sorted(label_tracks.items()) if len(tracks) > 1
    ]
    summary_lines = [
        '=== LRLPR-26: MANIFIESTO DE IMAGENES HR ===',
        f'Anotaciones/tracks validados: {tracks_seen}',
        f'Imagenes HR en manifest: {len(rows)}',
        f'Escenario A: {scenario_counts["Scenario-A"]} HR',
        f'Escenario B: {scenario_counts["Scenario-B"]} HR',
        f'IDs de texto distintos: {len(label_tracks)}',
        f'Etiquetas presentes en varios tracks: {len(collisions)}',
        'Imágenes copiadas o modificadas: 0',
        'Particiones nuevas creadas: NO',
        'Las etiquetas se mantienen como annotation_unverified.',
        'Las rutas de las imágenes son relativas a la carpeta del manifiesto.',
    ]
    for (scenario, layout), count in sorted(layout_tracks.items()):
        summary_lines.append(f'- {scenario}/{layout}: {count} tracks')
    summary_lines += [
        f'Manifiesto: {manifest_root / "manifest.csv"}',
        f'Conflictos: {report_root / "hr_label_collisions.csv"}',
        '=== FINALIZADO ===',
    ]
    atomic_csv(manifest_root / 'manifest.csv', FIELDNAMES, rows)
    atomic_csv(report_root / 'hr_label_collisions.csv',
        ['text_normalized', 'num_tracks', 'track_ids', 'scenarios', 'plate_layouts'], collisions)
    atomic_text(report_root / 'hr_manifest_summary.txt', '\n'.join(summary_lines) + '\n')
    print('\n'.join(summary_lines), flush=True)


if __name__ == '__main__':
    try:
        build(get_args())
    except Exception as exc:
        print(f'FALLO: {exc}', file=sys.stderr)
        sys.exit(1)
