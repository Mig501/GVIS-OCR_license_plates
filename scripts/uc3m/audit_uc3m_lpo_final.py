#!/usr/bin/env python3
"""Audita todos los recortes UC3M-LP_LPO y crea hojas de contacto de control.

Uso en Legio2:
    python scripts/uc3m/audit_uc3m_lpo_final.py

Solo lee el manifest.csv y los PNG de processed/UC3M-LP_LPO.
Escribe informes/hojas de contacto en results/inspection/UC3M-LP/lpo_final_audit/.
No modifica el dataset, las etiquetas ni el ZIP original.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
EXPECTED = {'total': 2547, 'train': 2056, 'test': 491, 4: 2405, 5: 105, 6: 37}
ISSUE_FIELDS = ['row', 'image_path', 'issue', 'detail']
INDEX_FIELDS = ['sheet', 'position', 'image_path', 'scenario', 'text_normalized',
                'polygon_vertices', 'width', 'height', 'selection_reason']


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA_ROOT / 'processed/UC3M-LP_LPO')
    parser.add_argument('--report-dir', type=Path, default=PROJECT_ROOT / 'results/inspection/UC3M-LP/lpo_final_audit')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--allow-nonstandard-counts', action='store_true',
                        help='Para pruebas con conjuntos reducidos, no para la auditoría real')
    return parser.parse_args()


def load_font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def safe_resolve(base: Path, relative: str):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('ruta no relativa o con ..')
    full = (base / path).resolve()
    if not full.is_relative_to(base.resolve()):
        raise ValueError('ruta fuera de la carpeta del dataset')
    return full


def audit(dataset: Path, rows: list[dict], allow_nonstandard: bool):
    issues = []
    counts = Counter()
    vertices = Counter()
    widths, heights = [], []
    seen = set()
    valid = []

    def issue(n, row, code, detail):
        issues.append({'row': n, 'image_path': row.get('image_path', ''),
                       'issue': code, 'detail': str(detail)})

    for n, row in enumerate(rows, start=2):
        relative = row.get('image_path', '')
        if relative in seen:
            issue(n, row, 'duplicate_path', 'mismo image_path en más de una fila')
            continue
        seen.add(relative)
        split = row.get('scenario', '')
        if split not in ('train', 'test'):
            issue(n, row, 'invalid_split', repr(split))
            continue
        if not Path(relative).name.startswith(split + '__'):
            issue(n, row, 'split_filename_mismatch', relative)
            continue
        try:
            image_file = safe_resolve(dataset, relative)
        except ValueError as exc:
            issue(n, row, 'unsafe_path', exc)
            continue
        if not image_file.is_file():
            issue(n, row, 'missing_image', image_file)
            continue
        if image_file.suffix.lower() != '.png':
            issue(n, row, 'not_png', relative)
            continue

        try:
            width, height = int(row['width']), int(row['height'])
            v = int(row['polygon_vertices'])
            n_chars = int(row['character_count'])
            known_n = int(row['known_characters'])
            masked_n = int(row['anonymized_characters'])
            with Image.open(image_file) as im:
                if im.format != 'PNG':
                    raise ValueError('imagen declarada PNG con otro formato')
                im.load()  # Fuerza a decodificar todos los píxeles y detectar truncamientos.
                actual_size = im.size
            if actual_size != (width, height):
                raise ValueError(f'tamaño declarado {width}x{height}, real {actual_size}')
            if width < 2 or height < 2:
                raise ValueError('dimensiones menores que 2 píxeles')
            if v < 3:
                raise ValueError('polígono de menos de 3 vértices')

            plate = row['text_normalized'].strip()
            mask = row['known_mask'].strip()
            visible = row['text_visible'].strip()
            if not plate or any(not (c.isalnum() or c == '*') for c in plate):
                raise ValueError('texto normalizado inválido')
            if len(plate) != len(mask) or any(c not in '01' for c in mask):
                raise ValueError('máscara de posiciones inválida')
            if mask != ''.join('0' if c == '*' else '1' for c in plate):
                raise ValueError('known_mask incoherente con los asteriscos')
            if visible != plate.replace('*', ''):
                raise ValueError('text_visible incoherente')
            if (n_chars, known_n, masked_n) != (len(plate), len(visible), plate.count('*')):
                raise ValueError('contadores de caracteres incoherentes')
            if row['label_status'] != ('partial_masked' if masked_n else 'source_annotation_unverified'):
                raise ValueError('label_status incoherente')
            if row.get('crop_method') != 'polygon_bbox_4pct_margin':
                raise ValueError('método de recorte distinto del esperado')
            src = Path(row.get('source_path', ''))
            ann = Path(row.get('annotation_path', ''))
            if not (src.stem == ann.stem and src.parent == ann.parent and
                    src.suffix.lower() == '.jpg' and ann.suffix.lower() == '.json' and
                    src.parent.name == split):
                raise ValueError('JPG/JSON homónimos o partición incoherente')
            if row['text_original'].split('-', 1)[-1].upper() != plate:
                raise ValueError('text_original no coincide con el GT normalizado')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue(n, row, 'invalid_record_or_image', exc)
            continue

        counts[split] += 1
        vertices[v] += 1
        widths.append(width)
        heights.append(height)
        valid.append(row)

    if not allow_nonstandard:
        if len(rows) != EXPECTED['total']:
            issue(0, {}, 'total_count', f'{len(rows)} != {EXPECTED["total"]}')
        for split in ('train', 'test'):
            if counts[split] != EXPECTED[split]:
                issue(0, {}, 'split_count', f'{split}: {counts[split]} != {EXPECTED[split]}')
        for v in (4, 5, 6):
            if vertices[v] != EXPECTED[v]:
                issue(0, {}, 'vertex_count', f'{v}: {vertices[v]} != {EXPECTED[v]}')
        if any(v not in (4, 5, 6) for v in vertices):
            issue(0, {}, 'unexpected_vertex_count', repr(dict(vertices)))
    return issues, valid, counts, vertices, widths, heights


def choose_stratified(rows, split, seed, amount=20):
    rng = random.Random(seed + (0 if split == 'train' else 71))
    target = {4: 12, 5: 4, 6: 4}
    chosen, seen = [], set()
    for v, count in target.items():
        candidates = [r for r in rows if r['scenario'] == split and int(r['polygon_vertices']) == v]
        for r in rng.sample(candidates, min(count, len(candidates))):
            if r['image_path'] not in seen:
                chosen.append((r, f'aleatoria_{v}_vertices'))
                seen.add(r['image_path'])
    remaining = [r for r in rows if r['scenario'] == split and r['image_path'] not in seen]
    for r in rng.sample(remaining, min(amount - len(chosen), len(remaining))):
        chosen.append((r, 'aleatoria_adicional'))
    rng.shuffle(chosen)
    return chosen[:amount]


def choose_edge_cases(rows, count=20):
    # Tres criterios distintos: ancho pequeño, altura pequeña y aspecto extremo.
    candidates = sorted(rows, key=lambda r: int(r['width']))[:100]
    small_height = sorted(rows, key=lambda r: int(r['height']))[:100]
    aspects = [int(r['width']) / int(r['height']) for r in rows]
    median_aspect = statistics.median(aspects)
    extreme = sorted(rows, key=lambda r: abs(math.log((int(r['width']) / int(r['height'])) / median_aspect)), reverse=True)[:100]
    output, seen = [], set()
    for pool, label, n in [(candidates, 'ancho_reducido', 7),
                           (small_height, 'altura_reducida', 7),
                           (extreme, 'aspecto_extremo', 6)]:
        for row in pool:
            if len([x for x, why in output if why == label]) >= n:
                break
            if row['image_path'] not in seen:
                output.append((row, label))
                seen.add(row['image_path'])
    return output[:count]


def create_sheet(dataset, name, selected, report_dir, index):
    columns, cell_w, cell_h = 4, 454, 196
    nrows = math.ceil(len(selected) / columns)
    sheet = Image.new('RGB', (columns * cell_w, max(1, nrows) * cell_h), (242, 244, 247))
    draw = ImageDraw.Draw(sheet)
    font = load_font(17)
    small = load_font(13)
    for pos, (row, reason) in enumerate(selected, 1):
        x = ((pos - 1) % columns) * cell_w
        y = ((pos - 1) // columns) * cell_h
        draw.rounded_rectangle([x + 7, y + 7, x + cell_w - 7, y + cell_h - 7],
                               radius=9, fill=(255, 255, 255), outline=(212, 216, 221), width=2)
        title = (f'{pos:02d}  {row["scenario"]}  {row["text_normalized"]} '
                 f'({row["polygon_vertices"]}v)')
        draw.text((x + 17, y + 14), title[:43], font=font, fill=(28, 32, 37))
        try:
            with Image.open(safe_resolve(dataset, row['image_path'])) as im:
                im.load()
                im = im.convert('RGB')
                im.thumbnail((cell_w - 34, 115), Image.Resampling.LANCZOS)
                x_im = x + (cell_w - im.width) // 2
                y_im = y + 40 + (115 - im.height) // 2
                sheet.paste(im, (x_im, y_im))
        except (OSError, ValueError) as exc:
            draw.text((x + 17, y + 75), f'ERROR: {str(exc)[:40]}', font=small, fill=(160, 0, 0))
        footer = f'{row["width"]}x{row["height"]} px  |  {reason}'
        draw.text((x + 17, y + cell_h - 29), footer[:57], font=small, fill=(90, 96, 105))
        index.append({'sheet': name, 'position': pos, 'image_path': row['image_path'],
                      'scenario': row['scenario'], 'text_normalized': row['text_normalized'],
                      'polygon_vertices': row['polygon_vertices'], 'width': row['width'],
                      'height': row['height'], 'selection_reason': reason})
    path = report_dir / name
    sheet.save(path, format='JPEG', quality=90, subsampling=0)
    sheet.close()
    return path


def write_csv(path, headers, data):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


def main():
    a = get_args()
    manifest = a.dataset / 'manifest.csv'
    if not manifest.is_file():
        raise SystemExit(f'ERROR: no existe el manifiesto: {manifest}')
    a.report_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open('r', encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    required = {'image_path', 'source_path', 'annotation_path', 'scenario',
                'text_original', 'text_normalized', 'text_visible', 'known_mask',
                'width', 'height', 'polygon_vertices', 'character_count',
                'known_characters', 'anonymized_characters', 'label_status', 'crop_method'}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f'ERROR: columnas ausentes del manifest: {sorted(missing)}')

    print('=== UC3M-LP: AUDITORIA FINAL DE RECORTES LPO ===', flush=True)
    print(f'Manifest: {manifest}', flush=True)
    print(f'Registros a comprobar: {len(rows)}', flush=True)
    issues, valid, splits, vertices, widths, heights = audit(a.dataset, rows, a.allow_nonstandard_counts)
    index = []
    sheets = []
    if valid:
        for split in ('train', 'test'):
            selected = choose_stratified(valid, split, a.seed)
            sheets.append(create_sheet(a.dataset, f'uc3m_lpo_{split}_audit.jpg',
                                       selected, a.report_dir, index))
        edge = choose_edge_cases(valid)
        sheets.append(create_sheet(a.dataset, 'uc3m_lpo_edge_cases.jpg',
                                   edge, a.report_dir, index))
    write_csv(a.report_dir / 'audit_issues.csv', ISSUE_FIELDS, issues)
    write_csv(a.report_dir / 'contact_sheet_index.csv', INDEX_FIELDS, index)
    labels = Counter(r['label_status'] for r in valid)
    lines = [
        '=== UC3M-LP: AUDITORIA FINAL DE RECORTES LPO ===',
        f'Filas del manifest: {len(rows)}',
        f'PNG comprobados (decodificacion completa): {len(valid)}',
        f'Train: {splits["train"]}; Test: {splits["test"]}',
        f'Poligonos: 4={vertices[4]}, 5={vertices[5]}, 6={vertices[6]}',
        f'Etiquetas: {dict(labels)}',
        f'Incidencias: {len(issues)}',
        f'Dimensiones ancho (min/mediana/max): {min(widths)}/{statistics.median(widths):g}/{max(widths)}' if widths else 'Sin anchos validos',
        f'Dimensiones alto (min/mediana/max): {min(heights)}/{statistics.median(heights):g}/{max(heights)}' if heights else 'Sin altos validos',
        f'Recortes mostrados en hojas de contacto: {len(index)}',
        'Hojas: ' + ', '.join(str(p) for p in sheets),
        'Solo lectura sobre processed/UC3M-LP_LPO; GT parcial y mascaras sin modificar.',
        'ESTADO: INTEGRIDAD CORRECTA; INSPECCION VISUAL PENDIENTE' if not issues else
        'ESTADO: REVISION NECESARIA - VER audit_issues.csv',
    ]
    summary_file = a.report_dir / 'audit_summary.txt'
    summary_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
