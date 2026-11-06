#!/usr/bin/env python3
"""Inspecciona anotaciones JSON de UC3M-LP directamente en el ZIP.

No extrae imagenes, no modifica el ZIP ni los manifiestos. Genera informes
reducidos que pueden descargarse con la tarea de inspeccion de VS Code.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ZIP_DEFAULT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
OUT_DEFAULT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP')
JSON_RE = re.compile(r'^UC3M-LP/(train|test)/[^/]+\.json$', flags=re.IGNORECASE)


def walk_schema(value, path, field_types, examples, depths=0):
    """Resume la estructura JSON manteniendo los indices de listas agrupados."""
    typename = type(value).__name__
    field_types[path][typename] += 1
    if not isinstance(value, (dict, list)) and path not in examples:
        example = repr(value)
        examples[path] = example[:100] + ('...' if len(example) > 100 else '')
    if depths >= 12:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            walk_schema(v, (path + '.' + str(k)) if path else str(k), field_types, examples, depths + 1)
    elif isinstance(value, list):
        # Los ejemplos por campo deben representar todos los tipos de elementos.
        # En UC3M-LP las listas suelen ser pequenas (objetos y caracteres).
        for item in value[:100]:
            walk_schema(item, path + '[*]', field_types, examples, depths + 1)


def main():
    ap = argparse.ArgumentParser(description='Inspeccionar esquema JSON de UC3M-LP sin extraer el ZIP')
    ap.add_argument('--zip', type=Path, default=ZIP_DEFAULT)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    if not args.zip.is_file():
        raise FileNotFoundError(f'No existe el archivo ZIP: {args.zip}')

    root_types = Counter()
    counts = Counter()
    top_keys = Counter()
    field_types = defaultdict(Counter)
    field_examples = {}
    failures = []
    candidate_examples = defaultdict(list)

    with zipfile.ZipFile(args.zip, 'r') as zf:
        members = [(n, m.group(1).lower()) for n in zf.namelist() if (m := JSON_RE.fullmatch(n))]
        members.sort(key=lambda x: x[0])
        print('=== UC3M-LP: INSPECCION DE ANOTACIONES JSON ===', flush=True)
        print(f'ZIP: {args.zip}', flush=True)
        print(f'Anotaciones JSON encontradas: {len(members)}', flush=True)
        for idx, (name, split) in enumerate(members, 1):
            try:
                with zf.open(name) as fh:
                    ann = json.load(fh)
                counts[split] += 1
                root_types[type(ann).__name__] += 1
                if isinstance(ann, dict):
                    top_keys.update(ann.keys())
                walk_schema(ann, '$', field_types, field_examples)
                # Preferir ejemplos con datos, elegidos por complejidad de contenido.
                if len(candidate_examples[split]) < 2:
                    candidate_examples[split].append((name, ann))
                else:
                    # Sustituir el segundo ejemplo si aparece uno mas completo.
                    prev = candidate_examples[split][1]
                    if len(json.dumps(ann, ensure_ascii=False)) > len(json.dumps(prev[1], ensure_ascii=False)):
                        candidate_examples[split][1] = (name, ann)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                failures.append({'path': name, 'error': f'{type(exc).__name__}: {exc}'})
            if idx % 500 == 0:
                print(f'Anotaciones procesadas: {idx}/{len(members)}', flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'annotation_schema_report.txt'
    examples_path = args.out / 'annotation_examples.json'
    fields_path = args.out / 'annotation_fields.csv'
    errors_path = args.out / 'annotation_errors.csv'

    field_rows = [
        {'field_path': key,
         'types': ', '.join(f'{kind}:{num}' for kind, num in sorted(types.items())),
         'occurrences': sum(types.values()),
         'example': field_examples.get(key, '')}
        for key, types in sorted(field_types.items())
    ]
    with fields_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['field_path', 'types', 'occurrences', 'example'])
        writer.writeheader()
        writer.writerows(field_rows)

    picked = [
        {'source_path': name, 'annotation': annotation}
        for split in ('train', 'test') for name, annotation in candidate_examples[split]
    ]
    examples_path.write_text(json.dumps(picked, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with errors_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['path', 'error'])
        writer.writeheader()
        writer.writerows(failures)

    lines = [
        '=== UC3M-LP: INSPECCION DE ANOTACIONES JSON ===',
        f'ZIP: {args.zip}',
        f'JSON encontrados: {len(members)}',
        f'Train validos: {counts["train"]}',
        f'Test validos: {counts["test"]}',
        f'JSON con errores: {len(failures)}',
        'Tipos de raiz: ' + (', '.join(f'{k}={v}' for k, v in root_types.most_common()) or '(ninguno)'),
        'Claves de primer nivel:',
    ]
    lines += [f'  {key}: {count}' for key, count in top_keys.most_common()]
    lines += [
        f'Campos/rutas de esquema identificados: {len(field_rows)}',
        f'Ejemplos guardados: {len(picked)}',
        f'Informe: {report_path}',
        f'Ejemplos: {examples_path}',
        f'Campos detallados: {fields_path}',
        'No se han extraido imagenes ni modificado archivos originales.',
        '=== FINALIZADO ===',
    ]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    if failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
