#!/usr/bin/env python3
"""Inspecciona de forma no destructiva los JSON de LRLPR-26 HR.

Uso en Legio2:
    python scripts/lrlpr/inspect_lrlpr_annotations.py

No modifica el dataset: solo genera informes en results/inspection/LRLPR-26.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/LRLPR-26/selected_hr')
REPORT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/LRLPR-26')
SCENARIOS = ('Scenario-A', 'Scenario-B')
EXAMPLES_PER_SCENARIO = 3


def shape(value, depth=0):
    """Esquema orientativo de un JSON, sin asumir nombres de campos."""
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(k): shape(v, depth + 1) for k, v in list(value.items())[:24]}
    if isinstance(value, list):
        if not value:
            return 'list[empty]'
        return {'list_length': len(value), 'first_item': shape(value[0], depth + 1)}
    return type(value).__name__


def get_paths(base):
    """Recorre solo annotations.json y mantiene orden reproducible."""
    for dirpath, dirs, files in os.walk(base):
        dirs.sort()
        if 'annotations.json' in files:
            yield Path(dirpath) / 'annotations.json'


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f'No existe el directorio extraído: {DATA_ROOT}')

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = ['=== LRLPR-26: INSPECCION DE ANOTACIONES HR ===', f'Raiz: {DATA_ROOT}']
    counts = Counter()
    countries = defaultdict(Counter)
    top_keys = Counter()
    top_types = Counter()
    invalid = []
    samples = defaultdict(list)
    sample_country = defaultdict(set)

    for scenario in SCENARIOS:
        base = DATA_ROOT / 'train' / scenario
        if not base.is_dir():
            raise FileNotFoundError(f'No existe escenario: {base}')
        for path in get_paths(base):
            rel = path.relative_to(DATA_ROOT)
            # train/Scenario-A/<pais>/track_xxxxx/annotations.json
            country = rel.parts[2] if len(rel.parts) >= 5 else '(sin pais en ruta)'
            counts[scenario] += 1
            countries[scenario][country] += 1
            try:
                with path.open('r', encoding='utf-8-sig') as f:
                    item = json.load(f)
            except (OSError, ValueError, UnicodeError) as exc:
                invalid.append((rel.as_posix(), str(exc)))
                continue

            top_types[type(item).__name__] += 1
            if isinstance(item, dict):
                top_keys.update(item.keys())

            # Ejemplos repartidos por país cuando sea posible.
            if country not in sample_country[scenario] and len(samples[scenario]) < EXAMPLES_PER_SCENARIO:
                samples[scenario].append({'source_path': rel.as_posix(), 'annotation': item})
                sample_country[scenario].add(country)

    lines += ['', 'Anotaciones encontradas:']
    for scenario in SCENARIOS:
        lines.append(f'- {scenario}: {counts[scenario]} JSON')
        for country, n in countries[scenario].most_common():
            lines.append(f'    {country}: {n}')
    lines.append(f'Total: {sum(counts.values())}')
    lines.append(f'JSON con error de lectura/parseo: {len(invalid)}')
    if invalid:
        lines.append('Primeros errores:')
        for path, message in invalid[:10]:
            lines.append(f'  {path}: {message}')
    lines += ['', 'Tipo de objeto JSON raiz:']
    for key, n in top_types.most_common():
        lines.append(f'- {key}: {n}')
    lines += ['', 'Claves de primer nivel (numero de documentos):']
    for key, n in top_keys.most_common():
        lines.append(f'- {key}: {n}')

    examples = []
    lines += ['', 'Ejemplos de estructura (tipos de datos, no valores):']
    for scenario in SCENARIOS:
        for item in samples[scenario]:
            rel = item['source_path']
            lines.append(f'--- {rel}')
            lines.append(json.dumps(shape(item['annotation']), ensure_ascii=False, indent=2)[:5000])
            examples.append(item)

    lines += ['', 'IMPORTANTE:', '- Este paso solo identifica el formato de las anotaciones.',
              '- Todavia no se infiere el texto OCR ni se construyen particiones.',
              '- No se han modificado imagenes, JSON ni manifiestos.']

    report_file = REPORT_ROOT / 'annotation_schema_report.txt'
    examples_file = REPORT_ROOT / 'annotation_examples.json'
    report_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    examples_file.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print('\n'.join(lines[:lines.index('Ejemplos de estructura (tipos de datos, no valores):')]))
    print(f'Ejemplos de anotacion guardados: {len(examples)}')
    print(f'Informe: {report_file}')
    print(f'Ejemplos: {examples_file}')
    print('No se han modificado imagenes ni anotaciones.')
    print('=== FINALIZADO ===')

    if invalid:
        raise SystemExit(1)
    if counts['Scenario-A'] != 10000 or counts['Scenario-B'] != 10000:
        raise SystemExit('ERROR: numero de anotaciones distinto del esperado; revisar informe.')


if __name__ == '__main__':
    main()
