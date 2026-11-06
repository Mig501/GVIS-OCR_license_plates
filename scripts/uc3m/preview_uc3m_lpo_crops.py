#!/usr/bin/env python3
"""Previsualización no destructiva de recortes LPO de UC3M-LP.

Lee fotografías y anotaciones directamente del ZIP original en Legio2.
Empareja JSON y JPG por el MISMO nombre, NO por `imagePath`.
Prueba un recorte rectangular que encierra todos los vértices (4, 5 o 6).
Solo crea muestras e informes en results/inspection; no altera el dataset.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_ZIP = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
DEFAULT_OUT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP/lpo_preview')
VERTICES = (4, 5, 6)
SPLITS = ('train', 'test')


def as_points(poly: object) -> list[tuple[float, float]] | None:
    if not isinstance(poly, list) or len(poly) not in VERTICES:
        return None
    pts = []
    for p in poly:
        if not isinstance(p, (tuple, list)) or len(p) != 2:
            return None
        x, y = p
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            return None
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        pts.append((float(x), float(y)))
    return pts


def bounding_box(points: list[tuple[float, float]], width: int, height: int, padding: float):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmax <= xmin or ymax <= ymin:
        return None
    pad_x = max(2, math.ceil((xmax - xmin) * padding))
    pad_y = max(2, math.ceil((ymax - ymin) * padding))
    x1 = max(0, math.floor(xmin) - pad_x)
    y1 = max(0, math.floor(ymin) - pad_y)
    x2 = min(width, math.ceil(xmax) + pad_x)
    y2 = min(height, math.ceil(ymax) + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def generate_sheet(items: list[dict], dest: Path, label: str):
    cols, cell_w, cell_h = 3, 460, 240
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell_w, 58 + rows * cell_h), '#f6f7fa')
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 14), f'UC3M-LP | Recortes LPO de prueba | {label}', fill='#17283c')
    draw.text((18, 36), 'BBox de todos los vertices; imagenes originales sin editar', fill='#526377')
    for idx, item in enumerate(items):
        x = (idx % cols) * cell_w
        y = 58 + (idx // cols) * cell_h
        draw.rectangle((x + 6, y + 6, x + cell_w - 6, y + cell_h - 6), fill='#ffffff', outline='#ccd7e3', width=2)
        text = (f"{item['split']} / {item['json_stem']} / LP {item['lp_index']} | "
                f"{len(item['points'])} vertices | {item['lp_id']}")
        draw.text((x + 17, y + 15), text[:76], fill='#192d42')
        draw.text((x + 17, y + 35), f"Recorte: {item['crop_width']}x{item['crop_height']} px | {item['image_name']}", fill='#526377')
        crop = item['crop_image']
        im = ImageOps.contain(crop, (cell_w - 32, cell_h - 80), method=Image.Resampling.NEAREST)
        ix = x + (cell_w - im.width) // 2
        iy = y + 65 + (cell_h - 80 - im.height) // 2
        sheet.paste(im, (ix, iy))
    sheet.save(dest, quality=92, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--per-group', type=int, default=4, help='Muestras por split y tipo de vertices (defecto 4)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--padding', type=float, default=0.03, help='Margen relativo por lado (defecto 0.03)')
    args = parser.parse_args()
    if not args.zip.is_file():
        raise FileNotFoundError(f'ZIP no encontrado: {args.zip}')
    if not (1 <= args.per_group <= 20):
        parser.error('--per-group debe estar entre 1 y 20')
    if not 0 <= args.padding <= 0.3:
        parser.error('--padding debe estar entre 0 y 0.3')
    args.output.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output / 'crops'
    samples_dir.mkdir(exist_ok=True)

    print('=== UC3M-LP: PRUEBA DE RECORTES LPO ===', flush=True)
    print(f'ZIP: {args.zip}', flush=True)
    print('Emparejamiento: JSON y JPG con el mismo nombre (NO imagePath)', flush=True)
    print(f'Metodo: bbox de todos los vertices, padding={args.padding:.1%}', flush=True)

    candidates = defaultdict(list)
    scanned_json = 0
    counted_lps = 0
    invalid_geometry = 0
    missing_images = 0

    with zipfile.ZipFile(args.zip) as zf:
        names = set(zf.namelist())
        json_names = sorted(
            n for n in names
            if n.lower().endswith('.json') and any(n.startswith(f'UC3M-LP/{split}/') for split in SPLITS)
        )
        for json_name in json_names:
            split = json_name.split('/')[1]
            img_name = json_name[:-5] + '.jpg'
            if img_name not in names:
                missing_images += 1
                continue
            ann = json.loads(zf.read(json_name))
            scanned_json += 1
            lps = ann.get('lps', [])
            if not isinstance(lps, list):
                invalid_geometry += 1
                continue
            for lp_i, lp in enumerate(lps):
                counted_lps += 1
                if not isinstance(lp, dict):
                    invalid_geometry += 1
                    continue
                pts = as_points(lp.get('poly_coord'))
                if pts is None:
                    invalid_geometry += 1
                    continue
                # Las coordenadas declaradas se validaran contra las dimensiones reales al recortar.
                candidates[(split, len(pts))].append({
                    'split': split,
                    'json_name': json_name,
                    'image_name': img_name,
                    'json_stem': Path(json_name).stem,
                    'lp_index': lp_i,
                    'lp_id': str(lp.get('lp_id', '')),
                    'points': pts,
                    'expected_width': ann.get('imageWidth'),
                    'expected_height': ann.get('imageHeight'),
                })

        print(f'JSON leidos: {scanned_json}; matriculas: {counted_lps}', flush=True)
        for split in SPLITS:
            for n in VERTICES:
                print(f'- {split}, {n} vertices: {len(candidates[(split, n)])}', flush=True)

        chosen = []
        for split in SPLITS:
            for n in VERTICES:
                group = candidates[(split, n)]
                rng = random.Random(args.seed + n + (100 if split == 'test' else 0))
                chosen.extend(rng.sample(group, min(args.per_group, len(group))))

        report_rows = []
        sheets = {split: [] for split in SPLITS}
        issues = []
        for i, rec in enumerate(chosen, 1):
            try:
                with Image.open(io.BytesIO(zf.read(rec['image_name']))) as im:
                    im.load()
                    w, h = im.size
                    if (rec['expected_width'], rec['expected_height']) != (w, h):
                        raise ValueError(f"dimensiones no coinciden: JSON {rec['expected_width']}x{rec['expected_height']} vs JPG {w}x{h}")
                    for x, y in rec['points']:
                        if x < 0 or y < 0 or x > w or y > h:
                            raise ValueError(f'vertice fuera de limites ({x}, {y})')
                    bbox = bounding_box(rec['points'], w, h, args.padding)
                    if bbox is None:
                        raise ValueError('bbox degenerada')
                    crop = im.crop(bbox).convert('RGB')
            except Exception as exc:
                issues.append((rec['image_name'], rec['lp_index'], str(exc)))
                continue
            out_name = f"{rec['split']}__{rec['json_stem']}__lp{rec['lp_index']:02d}__{len(rec['points'])}v.png"
            crop.save(samples_dir / out_name)
            rec.update(crop_width=crop.width, crop_height=crop.height, crop_image=crop)
            sheets[rec['split']].append(rec)
            report_rows.append({
                'split': rec['split'], 'json_path': rec['json_name'], 'image_path': rec['image_name'],
                'lp_index': rec['lp_index'], 'lp_id': rec['lp_id'], 'vertices': len(rec['points']),
                'masked_chars': rec['lp_id'].count('*'), 'crop_method': f'bbox_polygon_padding_{args.padding:.3f}',
                'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3],
                'crop_width': crop.width, 'crop_height': crop.height,
                'sample_path': f'crops/{out_name}',
            })
            print(f"Muestra {i}/{len(chosen)}: {out_name} {crop.width}x{crop.height}", flush=True)

        for split in SPLITS:
            if sheets[split]:
                path = args.output / f'uc3m_lpo_preview_{split}.jpg'
                generate_sheet(sheets[split], path, split)
                print(f'Hoja generada: {path}', flush=True)

    index_path = args.output / 'lpo_preview_index.csv'
    with index_path.open('w', encoding='utf-8', newline='') as f:
        fieldnames = ('split', 'json_path', 'image_path', 'lp_index', 'lp_id', 'vertices', 'masked_chars',
                      'crop_method', 'x1', 'y1', 'x2', 'y2', 'crop_width', 'crop_height', 'sample_path')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    summary = (f'=== UC3M-LP: PRUEBA DE RECORTES LPO ===\n'
               f'JSON procesados: {scanned_json}\n'
               f'Matriculas anotadas: {counted_lps}\n'
               f'Poligonos no utilizables: {invalid_geometry}\n'
               f'JSON sin imagen homonima: {missing_images}\n'
               f'Muestras seleccionadas: {len(chosen)}\n'
               f'Muestras generadas: {len(report_rows)}\n'
               f'Errores durante el recorte: {len(issues)}\n'
               f'Metodo: rectangulo contenedor de todos los vertices, margen {args.padding:.0%}\n'
               f'Recorte de prueba sin rectificacion de perspectiva\n'
               f'Las zonas anonimizadas NO se reconstruyen ni modifican\n'
               f'No se han modificado ZIP, imagenes originales ni manifiestos\n'
               f'=== FINALIZADO ===\n')
    (args.output / 'lpo_preview_summary.txt').write_text(summary, encoding='utf-8')
    print(summary, flush=True)
    if issues:
        for row in issues:
            print('ERROR DE MUESTRA:', *row, sep=' | ', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
