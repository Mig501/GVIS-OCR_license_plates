#!/usr/bin/env python3
"""Extrae exclusivamente imágenes HR y anotaciones de entrenamiento de LRLPR-26.

Conserva la estructura de tracks y el ZIP original. Diseñado para SLURM.
No necesita paquetes externos.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
DEFAULT_ZIP = DATA_ROOT / 'raw/LRLPR-26/LRLPR-26-5opEvJTW.zip'
DEFAULT_OUT = DATA_ROOT / 'raw/LRLPR-26/selected_hr'
DEFAULT_REPORT_DIR = PROJECT_ROOT / 'results/inspection/LRLPR-26'
IMAGE_RE = re.compile(r'^hr-(\d+)\.(?:jpg|jpeg|png)$', re.IGNORECASE)
EXPECTED = {'Scenario-A': (50_000, 10_000), 'Scenario-B': (50_000, 10_000)}


def classify(info: zipfile.ZipInfo):
    if info.is_dir():
        return None
    name = info.filename
    path = PurePosixPath(name)
    if (name.startswith('/') or '\\' in name or
            any(part in ('', '.', '..') for part in path.parts)):
        raise ValueError(f'Ruta no segura dentro del ZIP: {name!r}')
    if len(path.parts) < 4 or path.parts[0] != 'train' or path.parts[1] not in EXPECTED:
        return None
    if not path.parent.name.startswith('track_'):
        return None
    match = IMAGE_RE.fullmatch(path.name)
    if match:
        return ('hr', path.parts[1], path.parent.as_posix(), int(match.group(1)))
    if path.name == 'annotations.json':
        return ('json', path.parts[1], path.parent.as_posix(), None)
    return None


def atomic_extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.part_', dir=out.parent)
    try:
        with os.fdopen(fd, 'wb') as dst, zf.open(info, 'r') as src:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        if os.stat(tmp_name).st_size != info.file_size:
            raise IOError(f'Tamaño incorrecto al extraer: {info.filename}')
        os.replace(tmp_name, out)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--reports', type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    if not args.zip.is_file():
        raise FileNotFoundError(f'ZIP no encontrado: {args.zip}')

    print('=== LRLPR-26: EXTRACCION SELECTIVA HR ===', flush=True)
    print(f'ZIP: {args.zip}', flush=True)
    print(f'Destino: {args.output}', flush=True)
    print('Inspeccionando la estructura y validando tracks antes de extraer...', flush=True)

    counts = collections.defaultdict(lambda: collections.Counter())
    tracks = collections.defaultdict(lambda: {'frames': set(), 'json': 0, 'images': []})
    selected = []
    seen_zip_names = set()

    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            parsed = classify(info)
            if parsed is None:
                continue
            kind, scenario, track, frame = parsed
            if info.filename in seen_zip_names:
                raise ValueError(f'Ruta duplicada en el ZIP: {info.filename}')
            seen_zip_names.add(info.filename)
            # No extraemos enlaces simbólicos.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Enlace simbólico no permitido: {info.filename}')
            counts[scenario][kind] += 1
            item = tracks[track]
            if kind == 'hr':
                if frame in item['frames']:
                    raise ValueError(f'Frame HR duplicado en {track}: {frame}')
                item['frames'].add(frame)
                item['images'].append(info.filename)
            else:
                item['json'] += 1
            selected.append(info)

        problems = []
        for scenario, (expected_hr, expected_json) in EXPECTED.items():
            hr_count, json_count = counts[scenario]['hr'], counts[scenario]['json']
            print(f'{scenario}: {hr_count} HR, {json_count} JSON', flush=True)
            if (hr_count, json_count) != (expected_hr, expected_json):
                problems.append(f'{scenario}: esperado ({expected_hr}, {expected_json}), obtenido ({hr_count}, {json_count})')
        if len(tracks) != 20_000:
            problems.append(f'Tracks encontrados: {len(tracks)}; esperados: 20000')
        for track, value in tracks.items():
            if value['frames'] != {1, 2, 3, 4, 5} or value['json'] != 1:
                problems.append(f'{track}: HR={sorted(value["frames"])}, JSON={value["json"]}')
                if len(problems) >= 12:
                    break
        if problems:
            raise ValueError('Validación previa fallida (no se ha extraído nada):\n- ' + '\n- '.join(problems))

        print('Prevalidación correcta: 20.000 tracks, 100.000 imágenes HR, 20.000 JSON.', flush=True)
        written, already = 0, 0
        for n, info in enumerate(selected, 1):
            # Respetar los directorios originales dentro de selected_hr.
            target = args.output.joinpath(*PurePosixPath(info.filename).parts)
            if target.exists():
                if not target.is_file() or target.stat().st_size != info.file_size:
                    raise IOError(f'Archivo existente incompatible; no sobrescrito: {target}')
                already += 1
            else:
                atomic_extract(zf, info, target)
                written += 1
            if n % 20_000 == 0:
                print(f'Procesados: {n:,}/{len(selected):,}', flush=True)

    args.reports.mkdir(parents=True, exist_ok=True)
    index = args.reports / 'hr_track_index.csv'
    with index.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'category', 'track_path', 'annotation_path', 'hr_images'])
        for track in sorted(tracks):
            members = sorted(tracks[track]['images'])
            parts = PurePosixPath(track).parts
            category = '/'.join(parts[2:-1])
            writer.writerow([parts[1], category, track, f'{track}/annotations.json', ';'.join(members)])

    summary = args.reports / 'hr_extraction_summary.txt'
    lines = [
        '=== LRLPR-26: EXTRACCION SELECTIVA HR ===',
        f'ZIP de origen: {args.zip}',
        f'Destino: {args.output}',
        'Escenario A: 50000 HR, 10000 JSON',
        'Escenario B: 50000 HR, 10000 JSON',
        f'Tracks válidos: {len(tracks)}',
        'Imágenes HR seleccionadas: 100000',
        'Anotaciones JSON seleccionadas: 20000',
        'Imágenes LR extraídas: 0',
        'Imágenes de test extraídas: 0',
        f'Archivos nuevos extraídos: {written}',
        f'Archivos existentes reutilizados (comprobación de tamaño): {already}',
        f'Índice de tracks: {index}',
        'ZIP original intacto. No se ha generado ni modificado el manifest LPO.',
        '=== FINALIZADO ===',
    ]
    summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
