#!/usr/bin/env python3
"""Auditoria visual no destructiva de recortes HR de LRLPR-26.

Lee las imagenes de raw/LRLPR-26/selected_hr, agrupa por escenario/layout,
genera hojas de contacto aleatorias y de alertas geometricas e informes CSV.
Las alertas NO implican exclusion; ninguna imagen ni manifiesto se modifica.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_DATA = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/LRLPR-26/selected_hr')
DEFAULT_OUT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/LRLPR-26/lpo_visual_audit')
GROUPS = [(s, l) for s in ('Scenario-A', 'Scenario-B') for l in ('Brazilian', 'Mercosur')]
SUPPORTED = {'.png', '.jpg', '.jpeg'}


def make_sheet(rows, dst, root, heading, cell_w=350, cell_h=230, columns=4):
    if not rows:
        return False
    nrows = (len(rows) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * cell_w, nrows * cell_h + 42), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 12), heading, fill='black', font=font)
    for idx, item in enumerate(rows):
        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h + 42
        draw.rectangle([x, y, x+cell_w-1, y+cell_h-1], outline='#d0d0d0', width=1)
        try:
            with Image.open(root / item['path']) as im:
                shown = ImageOps.exif_transpose(im).convert('RGB')
                # Ampliacion de vista, sin modificar ni guardar imagen de origen.
                shown.thumbnail((cell_w-18, cell_h-67), Image.Resampling.LANCZOS)
                px = x + (cell_w - shown.width) // 2
                py = y + 6 + (cell_h-67-shown.height)//2
                canvas.paste(shown, (px, py))
        except Exception as exc:
            draw.text((x+10, y+20), f'ERROR: {type(exc).__name__}', fill='red', font=font)
        parts = Path(item['path']).parts
        track = parts[-2] if len(parts)>1 else '?'
        base = parts[-1]
        draw.text((x+8, y+cell_h-54), f'{track}/{base}   {item["width"]}x{item["height"]}', fill='black', font=font)
        reason = item.get('reasons', '') or 'muestra aleatoria'
        draw.text((x+8, y+cell_h-34), reason[:52], fill='#8b2020' if item.get('reasons') else '#444444', font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, quality=88, optimize=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--output', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--per-group', type=int, default=20)
    ap.add_argument('--alerts-per-group', type=int, default=20)
    args = ap.parse_args()
    root, out = args.data_root, args.output
    if not root.is_dir():
        raise FileNotFoundError(f'No existe el dataset HR: {root}')
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    by_group = defaultdict(list)
    unreadable = []
    print('=== LRLPR-26: AUDITORIA VISUAL DE RECORTES HR ===', flush=True)
    for scenario, layout in GROUPS:
        folder = root / 'train' / scenario / layout
        if not folder.is_dir():
            raise FileNotFoundError(f'Falta grupo: {folder}')
        paths = sorted(p for p in folder.glob('track_*/hr-*') if p.is_file() and p.suffix.lower() in SUPPORTED)
        for p in paths:
            relative = p.relative_to(root).as_posix()
            try:
                with Image.open(p) as image:
                    w, h = image.size
                    if w <= 0 or h <= 0:
                        raise ValueError('dimensiones no validas')
                by_group[(scenario, layout)].append({'path': relative, 'scenario': scenario,
                    'layout': layout, 'width': w, 'height': h, 'aspect': w/h, 'area': w*h,
                    'reasons': '', 'score': 0.0})
            except Exception as exc:
                unreadable.append({'path': relative, 'error': f'{type(exc).__name__}: {exc}'})
        print(f'{scenario}/{layout}: {len(paths)} rutas HR; {len(by_group[(scenario,layout)])} abiertas', flush=True)

    flag_counts = Counter()
    alerts = []
    report_lines = ['=== LRLPR-26: AUDITORIA VISUAL DE RECORTES HR ===',
                    f'Semilla aleatoria: {args.seed}',
                    'AVISO: las alertas son heuristicas; no confirman fotografias de vehiculos.', '']
    output_sheets = []
    total_images = 0
    for scenario, layout in GROUPS:
        items = by_group[(scenario,layout)]
        total_images += len(items)
        name = f'{scenario.lower().replace("scenario-", "scenario_")}_{layout.lower()}'
        if not items:
            raise RuntimeError(f'Grupo sin imagenes: {scenario}/{layout}')
        med_area = statistics.median(i['area'] for i in items)
        med_ratio = statistics.median(i['aspect'] for i in items)
        for item in items:
            reasons = []
            if item['aspect'] < 1.6:
                reasons.append('aspecto_bajo')
            if item['aspect'] > 6.0:
                reasons.append('aspecto_alto')
            if item['area'] > med_area * 12:
                reasons.append('area_muy_grande')
            if item['area'] < med_area / 12:
                reasons.append('area_muy_pequena')
            if item['width'] >= 400 or item['height'] >= 200:
                reasons.append('dimensiones_grandes')
            item['reasons'] = ';'.join(reasons)
            if reasons:
                for reason in reasons:
                    flag_counts[reason] += 1
                item['score'] = (8 if 'dimensiones_grandes' in reasons else 0) + \
                     (7 if 'area_muy_grande' in reasons else 0) + \
                     (4 if 'aspecto_bajo' in reasons else 0) + \
                     (3 if 'aspecto_alto' in reasons else 0) + \
                     (1 if 'area_muy_pequena' in reasons else 0) + \
                     min(item['area']/max(med_area, 1), 100)/100
                alerts.append(item)
        sample = rng.sample(items, min(args.per_group, len(items)))
        random_path = out / f'{name}_random.jpg'
        make_sheet(sample, random_path, root, f'{scenario} / {layout}: muestra aleatoria')
        output_sheets.append(random_path)
        group_alerts = sorted([i for i in items if i['reasons']], key=lambda i: (-i['score'], i['path']))
        if group_alerts:
            alert_path = out / f'{name}_alerts.jpg'
            make_sheet(group_alerts[:args.alerts_per_group], alert_path, root, f'{scenario} / {layout}: alertas prioritarias')
            output_sheets.append(alert_path)
        report_lines.append(f'{scenario}/{layout}: {len(items)} HR; mediana {med_area:.0f} px^2; '
            f'aspecto mediano {med_ratio:.2f}; {len(group_alerts)} alertas')
    alert_csv = out / 'geometry_alerts.csv'
    with alert_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','scenario','layout','width','height','aspect','area','reasons'])
        writer.writeheader()
        for item in sorted(alerts, key=lambda i: (-i['score'], i['path'])):
            writer.writerow({key:item[key] for key in writer.fieldnames})
    with (out/'unreadable_images.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path','error'])
        writer.writeheader()
        writer.writerows(unreadable)
    report_lines.extend(['', f'Imagenes inspeccionadas: {total_images}',
                         f'Imagenes con alerta geometrica: {len(alerts)}',
                         f'Imagenes no legibles: {len(unreadable)}',
                         f'Hojas de contacto generadas: {len(output_sheets)}',
                         'Tipos de alerta:'] + [f'  {k}: {v}' for k,v in sorted(flag_counts.items())] +
                        ['', 'Ninguna imagen ni manifiesto modificado.',
                         'Las imagenes no marcadas tambien pueden contener errores de recorte.',
                         'El informe no equivale a validacion visual exhaustiva.'])
    (out/'visual_audit_summary.txt').write_text('\n'.join(report_lines)+'\n', encoding='utf-8')
    print('\n'.join(report_lines), flush=True)
    print(f'Informes: {out}', flush=True)
    print('=== FINALIZADO ===', flush=True)

if __name__ == '__main__':
    main()
