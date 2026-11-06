#!/usr/bin/env python3
"""Audita el emparejamiento JSON/JPG por mismo nombre dentro del ZIP de UC3M-LP.

Solo lee el ZIP y escribe informes/hojas de contacto. No extrae ni modifica
las fotografias, anotaciones ni manifiestos.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont

DEFAULT_ZIP = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
DEFAULT_OUTPUT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP/image_pairing/same_name_audit')


def valid_point(p, w, h):
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(a, (int, float)) for a in p) and 0 <= p[0] < w and 0 <= p[1] < h


def note(issues, split, name, code, detail):
    issues.append(dict(split=split, json_path=name, issue=code, detail=str(detail)))


def csv_write(path, cols, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def render_sheet(zf, cases, path, title):
    # Cada muestra usa el JSON y JPG con el mismo nombre.
    W, COLS, CW, CH = 1440, 3, 480, 310
    rows = (len(cases) + COLS - 1)//COLS
    page = Image.new('RGB', (W, 56 + rows*CH), '#f6f7f9')
    d = ImageDraw.Draw(page)
    d.text((14, 16), title, fill='#142b45')
    for i, case in enumerate(cases):
        left, top = (i%COLS)*CW, 56+(i//COLS)*CH
        img_ref = case['image_path']
        ann = case['annotation']
        with zf.open(img_ref) as file:
            # Cargar solo esta imagen; descartar al pasar al siguiente caso.
            with Image.open(file) as image:
                image.load()
                image = image.convert('RGB')
                orig_w, orig_h = image.size
                image.thumbnail((CW-26, 235), Image.Resampling.LANCZOS)
                rw, rh = image.size
        ox, oy = left+(CW-rw)//2, top+23+(235-rh)//2
        page.paste(image, (ox,oy))
        draw = ImageDraw.Draw(page)
        for lp in ann.get('lps', []):
            poly = lp.get('poly_coord', [])
            if len(poly) == 4 and all(valid_point(p, orig_w, orig_h) for p in poly):
                coords = [(round(ox+p[0]*rw/orig_w), round(oy+p[1]*rh/orig_h)) for p in poly]
                draw.line(coords+[coords[0]], fill='#00f46a', width=3)
        draw.text((left+10, top+8), f"{case['split']}/{case['stem']}.jpg", fill='#243447')
        draw.text((left+10, top+267), f"{orig_w}x{orig_h}  |  matrículas: {len(ann.get('lps', []))}", fill='#29394d')
        draw.line([(left,top+CH-1),(left+CW,top+CH-1)], fill='#d5d9df', width=1)
    page.save(path, quality=86)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--samples-per-split', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    issues, report_rows = [], []
    candidate_samples = {'train': [], 'test': []}
    counters = Counter()

    print('=== UC3M-LP: VALIDACION EMPAREJAMIENTO POR NOMBRE ===', flush=True)
    with zipfile.ZipFile(args.zip) as zf:
        entries = set(zf.namelist())
        json_paths = sorted(p for p in entries if p.endswith('.json') and '/train/' in p or p.endswith('.json') and '/test/' in p)
        for i, jp in enumerate(json_paths, 1):
            split = 'train' if '/train/' in jp else 'test'
            name = PurePosixPath(jp).stem
            ip = str(PurePosixPath(jp).with_suffix('.jpg'))
            counters['json'] += 1
            counters[f'json_{split}'] += 1
            row = dict(split=split, json_path=jp, same_name_image=ip, image_exists='no', dimensions_match='no', lps=0, invalid_polygons=0, invalid_char_boxes=0)
            try:
                ann = json.loads(zf.read(jp).decode('utf-8-sig'))
            except Exception as ex:
                note(issues, split, jp, 'json_invalid', repr(ex))
                report_rows.append(row)
                continue
            if ip not in entries:
                note(issues, split, jp, 'jpg_missing', ip)
                report_rows.append(row)
                continue
            row['image_exists'] = 'yes'
            counters['image_exists'] += 1
            try:
                with zf.open(ip) as imf:
                    with Image.open(imf) as im:
                        width, height = im.size
                        if im.format != 'JPEG':
                            note(issues, split, jp, 'unexpected_format', im.format)
            except Exception as ex:
                note(issues, split, jp, 'image_read_error', repr(ex))
                report_rows.append(row)
                continue
            if ann.get('imageWidth') == width and ann.get('imageHeight') == height:
                row['dimensions_match'] = 'yes'
                counters['dimensions_match'] += 1
            else:
                note(issues, split, jp, 'dimension_mismatch', f"JSON={ann.get('imageWidth')}x{ann.get('imageHeight')} JPG={width}x{height}")
            lps = ann.get('lps')
            if not isinstance(lps, list):
                note(issues, split, jp, 'lps_not_list', type(lps).__name__)
                report_rows.append(row)
                continue
            row['lps'] = len(lps)
            counters['plates'] += len(lps)
            for pidx, lp in enumerate(lps):
                poly = lp.get('poly_coord', [])
                if not isinstance(poly, list) or len(poly) != 4 or not all(valid_point(p,width,height) for p in poly):
                    row['invalid_polygons'] += 1
                    note(issues, split, jp, 'invalid_polygon', pidx)
                for cidx, char in enumerate(lp.get('characters', [])):
                    bbox = char.get('bbox_coord', [])
                    if not isinstance(bbox, list) or len(bbox) != 2 or not all(valid_point(p,width,height) for p in bbox) or bbox[0][0] >= bbox[1][0] or bbox[0][1] >= bbox[1][1]:
                        row['invalid_char_boxes'] += 1
                        note(issues, split, jp, 'invalid_character_bbox', f'{pidx}:{cidx}')
            counters['invalid_polygons'] += row['invalid_polygons']
            counters['invalid_char_boxes'] += row['invalid_char_boxes']
            if row['dimensions_match'] == 'yes' and row['invalid_polygons'] == 0 and row['invalid_char_boxes'] == 0:
                candidate_samples[split].append(dict(split=split, stem=name, image_path=ip, annotation=ann))
            report_rows.append(row)
            if i % 500 == 0:
                print(f'Comprobados: {i}/{len(json_paths)}', flush=True)
        cols = ['split','json_path','same_name_image','image_exists','dimensions_match','lps','invalid_polygons','invalid_char_boxes']
        csv_write(args.output/'same_name_checks.csv',cols,report_rows)
        csv_write(args.output/'same_name_issues.csv',['split','json_path','issue','detail'],issues)
        sheets = []
        for split in ('train', 'test'):
            sample = rng.sample(candidate_samples[split], min(args.samples_per_split, len(candidate_samples[split])))
            if sample:
                path = args.output/f'same_name_{split}_samples.jpg'
                render_sheet(zf, sample, path, f'UC3M-LP | {split} | JSON y JPG con el mismo nombre (muestra adicional)')
                sheets.append(path.name)
    summary = [
        '=== UC3M-LP: VALIDACION EMPAREJAMIENTO POR NOMBRE ===',
        f"JSON inspeccionados: {counters['json']}",
        f"Train: {counters['json_train']} | Test: {counters['json_test']}",
        f"JPG homonimos encontrados: {counters['image_exists']}",
        f"Dimensiones JPG/JSON coincidentes: {counters['dimensions_match']}",
        f"Matriculas anotadas: {counters['plates']}",
        f"Poligonos invalidos: {counters['invalid_polygons']}",
        f"Cajas de caracteres invalidas: {counters['invalid_char_boxes']}",
        f"Incidencias registradas: {len(issues)}",
        f"Hojas visuales: {', '.join(sheets) or 'ninguna'}",
        'ATENCION: dimensiones y coordenadas validas no demuestran por si solas correspondencia visual.',
        'No se han modificado imagenes, JSON ni manifiestos.',
        '=== FINALIZADO ===',
    ]
    (args.output/'same_name_summary.txt').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    print('\n'.join(summary), flush=True)
    print('Resultados:',args.output)


if __name__ == '__main__':
    main()
