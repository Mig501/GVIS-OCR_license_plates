#!/usr/bin/env python3
"""Inspección ligera y sin extracción del ZIP oficial de UC3M-LP.

Ejemplo: python scripts/uc3m/inspect_uc3m_zip.py
No modifica el ZIP ni extrae datos del dataset.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

DEFAULT_ZIP = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
DEFAULT_OUTPUT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP')
SPLITS = ('train', 'test')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def split_from_path(p: PurePosixPath) -> str | None:
    for part in p.parts[:-1]:
        if part.lower() in SPLITS:
            return part.lower()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', dest='archive', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(f'No existe el ZIP: {args.archive}')
    args.output.mkdir(parents=True, exist_ok=True)

    ext_counter: Counter[str] = Counter()
    split_ext: dict[str, Counter[str]] = defaultdict(Counter)
    first_level: Counter[str] = Counter()
    second_level: Counter[str] = Counter()
    type_examples: dict[str, list[str]] = defaultdict(list)
    image_keys: dict[str, set[str]] = defaultdict(set)
    json_keys: dict[str, set[str]] = defaultdict(set)
    duplicate_keys: list[str] = []
    unexpected: list[str] = []
    names_seen: set[str] = set()
    listed_files = 0
    uncompressed_size = 0

    try:
        with ZipFile(args.archive, 'r') as zf:
            # El MD5 del ZIP ya fue verificado en la descarga: aquí solo leemos el índice.
            for info in zf.infolist():
                if info.is_dir():
                    continue
                listed_files += 1
                uncompressed_size += info.file_size
                name = info.filename
                if name in names_seen:
                    duplicate_keys.append(name)
                names_seen.add(name)

                p = PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts:
                    raise ValueError(f'Ruta insegura dentro del ZIP: {name}')
                ext = p.suffix.lower() or '(sin extensión)'
                ext_counter[ext] += 1
                split = split_from_path(p)
                split_ext[split or '(fuera de train/test)'][ext] += 1
                first_level[p.parts[0] if p.parts else '(vacío)'] += 1
                second_level['/'.join(p.parts[:2])] += 1
                if len(type_examples[ext]) < 12:
                    type_examples[ext].append(name)
                if split and ext in IMAGE_EXTS:
                    image_keys[split].add(str(p.with_suffix('')))
                elif split and ext == '.json':
                    json_keys[split].add(str(p.with_suffix('')))
                elif p.name.lower() not in {'train.txt', 'test.txt'}:
                    unexpected.append(name)
    except BadZipFile as e:
        raise ValueError('El archivo no tiene estructura ZIP válida') from e

    counts = {s: {'images': sum(split_ext[s][e] for e in IMAGE_EXTS),
                  'json': split_ext[s]['.json'],
                  'images_without_json': len(image_keys[s] - json_keys[s]),
                  'json_without_image': len(json_keys[s] - image_keys[s])} for s in SPLITS}
    lines = [
        '=== UC3M-LP: INSPECCION DEL ZIP ===',
        f'Archivo ZIP: {args.archive}',
        f'Tamaño del ZIP: {args.archive.stat().st_size:,} bytes',
        f'Entradas de archivo (sin directorios): {listed_files}',
        f'Tamaño descomprimido anunciado: {uncompressed_size:,} bytes',
        '', 'Distribución por extensión:',
    ]
    for ext, qty in sorted(ext_counter.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f'- {ext}: {qty}')
    lines.extend(['', 'Por partición (imágenes / JSON / imágenes sin JSON / JSON sin imagen):'])
    for split in SPLITS:
        c = counts[split]
        lines.append(f'- {split}: {c["images"]} / {c["json"]} / {c["images_without_json"]} / {c["json_without_image"]}')
    lines.extend(['', 'Primeros niveles de directorios:'])
    for folder, qty in first_level.most_common(15):
        lines.append(f'- {folder}: {qty}')
    lines.extend(['', 'Rutas de segundo nivel más frecuentes:'])
    for folder, qty in second_level.most_common(15):
        lines.append(f'- {folder}: {qty}')
    lines.extend(['', f'Nombres duplicados en ZIP: {len(duplicate_keys)}',
                  f'Archivos inesperados fuera de imágenes/JSON/índices: {len(unexpected)}',
                  'No se han extraído ni modificado archivos.',
                  '=== FINALIZADO ==='])

    (args.output / 'zip_structure_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    examples = []
    for ext, paths in sorted(type_examples.items()):
        examples.append(f'=== {ext} ===')
        examples.extend(paths)
        examples.append('')
    examples.append('=== Rutas inesperadas (máximo 30) ===')
    examples.extend(unexpected[:30])
    (args.output / 'zip_example_paths.txt').write_text('\n'.join(examples) + '\n', encoding='utf-8')

    with (args.output / 'zip_extensions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['extension', 'count'])
        writer.writerows(sorted(ext_counter.items(), key=lambda x: (-x[1], x[0])))

    print('\n'.join(lines))
    print(f'Informe: {args.output / "zip_structure_report.txt"}')
    print(f'Ejemplos: {args.output / "zip_example_paths.txt"}')


if __name__ == '__main__':
    main()
