#!/usr/bin/env python3
"""UC3M-LP: visualiza polígonos de 5/6 vértices; no modifica los originales.

El CSV de entrada procede de diagnose_uc3m_polygons.py.
Compatible con el esquema ya auditado de UC3M-LP y los paths de Legio2.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import math
import random
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ZIP_DEFAULT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
REPORTS = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP/image_pairing/same_name_audit/polygon_diagnostics')


def polygon_area(points):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def choose_samples(rows, seed=42, count=24):
    rng = random.Random(seed)
    # Distribución balanceada por partición y número de vértices
    groups = {}
    for row in rows:
        key = (row['split'], len(row['_points']))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < count and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def annotate_candidate(zf, row):
    points = row['_points']
    path = row['image_path']
    with zf.open(path) as f:
        with Image.open(io.BytesIO(f.read())) as original:
            im = original.convert('RGB')
    width, height = im.size
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    pw = xmax - xmin
    ph = ymax - ymin
    margin_x = max(20, int(pw * .35))
    margin_y = max(20, int(ph * 0.65))
    x1 = max(0, xmin - margin_x)
    y1 = max(0, ymin - margin_y)
    x2 = min(width, xmax + margin_x)
    y2 = min(height, ymax + margin_y)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f'ROI vacía: {path}')
    im = im.crop((x1, y1, x2, y2))
    localized = [(x - x1, y - y1) for x, y in points]
    dr = ImageDraw.Draw(im)
    stroke = max(2, int(min(im.size) / 60))
    dr.line(localized + [localized[0]], fill=(255, 65, 55), width=stroke, joint='curve')
    for j, (x, y) in enumerate(localized):
        r = max(3, stroke + 1)
        dr.ellipse((x-r, y-r, x+r, y+r), fill=(255, 220, 35), outline=(40, 40, 40), width=1)
    # Simple recorte rectangular para referencia (no sustituye la rectificación)
    dr.rectangle((xmin-x1, ymin-y1, xmax-x1, ymax-y1), outline=(35, 220, 85), width=max(1, stroke // 2))
    im.thumbnail((400, 170), Image.Resampling.LANCZOS)
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, default=ZIP_DEFAULT)
    parser.add_argument('--issues', type=Path, default=REPORTS / 'polygon_issue_details.csv')
    parser.add_argument('--output', type=Path, default=REPORTS)
    parser.add_argument('--sample', type=int, default=24)
    args = parser.parse_args()

    if not args.zip.is_file():
        raise FileNotFoundError(f'ZIP no encontrado: {args.zip}')
    if not args.issues.is_file():
        raise FileNotFoundError(f'CSV no encontrado: {args.issues}')

    with args.issues.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    counts = Counter()
    repeated = 0
    for row in rows:
        pts = ast.literal_eval(row['poly_points'])
        if not isinstance(pts, list) or len(pts) < 3 or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts):
            raise ValueError(f'Polígono inesperado: {row["image_path"]}')
        row['_points'] = [tuple(float(x) for x in p) for p in pts]
        n = len(pts)
        counts[n] += 1
        repeated += len(set(row['_points'])) < len(row['_points'])
        if polygon_area(row['_points']) <= 0:
            raise ValueError(f'Polígono sin área: {row["image_path"]}')

    sample = choose_samples(rows, count=args.sample)
    args.output.mkdir(parents=True, exist_ok=True)
    print('=== UC3M-LP: REVISION DE POLIGONOS DE VERTICES VARIABLES ===', flush=True)
    print(f'Polígonos en CSV: {len(rows)}', flush=True)
    for n in sorted(counts):
        print(f'- {n} vértices: {counts[n]}', flush=True)
    print(f'Con vértices repetidos: {repeated}', flush=True)
    print(f'Muestras visuales: {len(sample)}', flush=True)

    pages = []
    with zipfile.ZipFile(args.zip) as zf:
        for start in range(0, len(sample), 12):
            sheet = Image.new('RGB', (1304, 1040), (245, 247, 249))
            dr = ImageDraw.Draw(sheet)
            font = ImageFont.load_default()
            page_num = start // 12 + 1
            dr.text((12, 10), f'UC3M-LP | Poligonos de 5/6 vertices | hoja {page_num}', fill=(15, 25, 40), font=font)
            for idx, row in enumerate(sample[start:start+12]):
                col = idx % 3
                line = idx // 3
                x0, y0 = 12 + col * 430, 46 + line * 245
                dr.rectangle((x0, y0, x0+416, y0+232), fill=(255, 255, 255), outline=(195, 202, 208))
                name = row['image_path'].replace('UC3M-LP/', '')
                title = f'{name} | LP {row["lp_index"]} | {len(row["_points"])} vertices'
                dr.text((x0 + 8, y0 + 9), title[:69], fill=(20, 33, 50), font=font)
                dr.text((x0 + 8, y0 + 25), f'GT parcial: {row["lp_id"]}', fill=(55, 65, 80), font=font)
                try:
                    preview = annotate_candidate(zf, row)
                    sheet.paste(preview, (x0 + (416-preview.width)//2, y0 + 52 + (170-preview.height)//2))
                except Exception as ex:
                    dr.text((x0+12, y0+95), f'ERROR {type(ex).__name__}: {str(ex)[:45]}', fill=(180, 10, 10), font=font)
            target = args.output / f'polygon_variable_review_{page_num:02}.jpg'
            sheet.save(target, quality=88)
            pages.append(target)
            print('Hoja generada:', target, flush=True)
    print('Imagenes y anotaciones originales no modificadas.')
    print('=== FINALIZADO ===', flush=True)


if __name__ == '__main__':
    main()
