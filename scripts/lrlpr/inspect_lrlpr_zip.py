#!/usr/bin/env python3
"""Inspect a downloaded LRLPR-26 ZIP without extracting or modifying it.

Run inside the project's SLURM/Apptainer execution setup.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

ZIP_PATH = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/LRLPR-26/LRLPR-26-5opEvJTW.zip')
OUT_DIR = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/LRLPR-26')
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
ANNOTATION_EXT = {'.txt', '.csv', '.json', '.xml', '.yaml', '.yml'}


def write_count_csv(path: Path, column: str, counter: Counter) -> None:
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([column, 'count'])
        writer.writerows(counter.most_common())


def main() -> None:
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f'No existe el ZIP: {ZIP_PATH}')

    ext_counts = Counter()
    first_level = Counter()
    two_levels = Counter()
    three_levels = Counter()
    keyword_counts = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    annotation_examples: list[str] = []
    image_count = 0
    total_files = 0
    uncompressed_bytes = 0

    try:
        with ZipFile(ZIP_PATH) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                total_files += 1
                uncompressed_bytes += info.file_size
                name = info.filename.replace('\\', '/')
                parts = PurePosixPath(name).parts
                if not parts:
                    continue
                ext = PurePosixPath(name).suffix.lower() or '(sin extensión)'
                ext_counts[ext] += 1
                first_level['/'.join(parts[:1])] += 1
                two_levels['/'.join(parts[:2])] += 1
                three_levels['/'.join(parts[:3])] += 1
                if ext in IMAGE_EXT:
                    image_count += 1
                if ext in ANNOTATION_EXT and len(annotation_examples) < 12:
                    annotation_examples.append(name)
                if len(examples[ext]) < 8:
                    examples[ext].append(name)

                # These counts are indicators only: do not infer dataset roles from them.
                segments = {segment.casefold() for segment in parts[:-1]}
                if 'hr' in segments or 'high_resolution' in segments or 'high-res' in segments:
                    keyword_counts['Directorio HR'] += 1
                if 'lr' in segments or 'low_resolution' in segments or 'low-res' in segments:
                    keyword_counts['Directorio LR'] += 1
                if 'train' in segments or 'training' in segments:
                    keyword_counts['Directorio train'] += 1
                if 'test' in segments or 'testing' in segments:
                    keyword_counts['Directorio test'] += 1
                if 'scenario_a' in segments or 'scenario-a' in segments:
                    keyword_counts['Directorio escenario A'] += 1
                if 'scenario_b' in segments or 'scenario-b' in segments:
                    keyword_counts['Directorio escenario B'] += 1
    except BadZipFile as exc:
        raise RuntimeError('El archivo ZIP es inválido') from exc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_count_csv(OUT_DIR / 'zip_extensions.csv', 'extension', ext_counts)
    write_count_csv(OUT_DIR / 'zip_folders_level2.csv', 'first_two_components', two_levels)
    write_count_csv(OUT_DIR / 'zip_folders_level3.csv', 'first_three_components', three_levels)
    with (OUT_DIR / 'zip_example_paths.txt').open('w', encoding='utf-8') as file:
        for ext in sorted(examples):
            file.write(f'=== {ext} ===\n')
            for path in examples[ext]:
                file.write(path + '\n')
            file.write('\n')
        file.write('=== Ejemplos de ficheros de anotación (rutas) ===\n')
        for path in annotation_examples:
            file.write(path + '\n')

    report = [
        '=== LRLPR-26: INSPECCION DE ESTRUCTURA ZIP ===',
        f'ZIP: {ZIP_PATH}',
        f'Entradas de archivo (excluidos directorios): {total_files}',
        f'Imagenes por extension: {image_count}',
        f'Ficheros no imagen: {total_files - image_count}',
        f'Tamano descomprimido: {uncompressed_bytes:,} bytes',
        'Extensiones mas comunes:',
        *[f'- {extension}: {count}' for extension, count in ext_counts.most_common(12)],
        'Primeros directorios:',
        *[f'- {path}: {count}' for path, count in first_level.most_common(15)],
        'Rutas de segundo nivel mas frecuentes:',
        *[f'- {path}: {count}' for path, count in two_levels.most_common(15)],
        'Coincidencias de nombres de directorio (orientativas):',
        *[f'- {key}: {value}' for key, value in sorted(keyword_counts.items())],
        f'Informe completo: {OUT_DIR}',
        'No se han extraido ni modificado archivos.',
        '=== FINALIZADO ===',
    ]
    (OUT_DIR / 'zip_structure_report.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
