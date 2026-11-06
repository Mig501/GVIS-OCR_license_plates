#!/usr/bin/env python3
"""Diagnostica polígonos fuera de rango de UC3M-LP sin modificar el ZIP.

Emparejamiento: JPG con igual nombre y misma partición que su JSON.
Produce CSV detallado, resumen y hojas de contacto representativas.
No elimina, recorta, corrige ni extrae las imágenes originales.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from PIL import Image, ImageDraw, ImageFont

ZIP_DEFAULT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
OUT_DEFAULT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP/image_pairing/same_name_audit/polygon_diagnostics')


def numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def analyze(poly, width, height):
    if not isinstance(poly, list) or len(poly) != 4:
        return 'poligono_malformado', [], None
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or not all(numeric(v) for v in p) for p in poly):
        return 'coordenadas_malformadas', [], None
    outside = []
    max_overrun = 0.0
    for p in poly:
        x, y = p
        if x < 0:
            outside.append('x_negativa'); max_overrun = max(max_overrun, -x)
        if y < 0:
            outside.append('y_negativa'); max_overrun = max(max_overrun, -y)
        if x >= width:
            outside.append('x_supera_ancho'); max_overrun = max(max_overrun, x - (width-1))
        if y >= height:
            outside.append('y_supera_alto'); max_overrun = max(max_overrun, y - (height-1))
    if not outside:
        return '', [], 0.0
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    intersects = max(xs) >= 0 and max(ys) >= 0 and min(xs) < width and min(ys) < height
    prefix = 'parcialmente_fuera' if intersects else 'totalmente_fuera'
    return prefix, sorted(set(outside)), max_overrun


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_sheets(zf, items, out, max_per_sheet=12):
    if not items:
        return []
    names = []
    cw, ch, cols = 540, 350, 3
    for sheet_idx, start in enumerate(range(0, len(items), max_per_sheet), 1):
        batch = items[start:start+max_per_sheet]
        rows = math.ceil(len(batch)/cols)
        canvas = Image.new('RGB', (cw*cols, 48+ch*rows), '#f5f7f9')
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 16), f'UC3M-LP | Poligonos con incidencias | Hoja {sheet_idx}', fill='#172f46')
        for idx, rec in enumerate(batch):
            left = (idx%cols)*cw
            top = 48+(idx//cols)*ch
            image_path = rec['image_path']
            draw.text((left+10, top+8), f"{rec['split']}/{PurePosixPath(image_path).name} | LP {rec['lp_index']} | {rec['lp_id'][:20]}", fill='#263c4e')
            draw.text((left+10, top+28), f"{rec['issue_type']} | excedente {rec['max_overrun_px']} px", fill='#a63d32')
            try:
                with zf.open(image_path) as stream:
                    with Image.open(stream) as im:
                        # La reduccion JPEG minimiza CPU/RAM; no modifica el original.
                        im.draft('RGB', (950, 650))
                        im = im.convert('RGB')
                        w, h = im.size
                        im.thumbnail((cw-22, 248), Image.Resampling.LANCZOS)
                        preview = im.copy()
                ow, oh = int(rec['width']), int(rec['height'])
                px, py = left+(cw-preview.width)//2, top+53+(248-preview.height)//2
                canvas.paste(preview, (px,py))
                raw = rec['poly_points']
                if raw:
                    poly = json.loads(raw)
                    if len(poly) == 4 and all(len(p)==2 and all(numeric(v) for v in p) for p in poly):
                        pts = [(px+round(p[0]*preview.width/ow), py+round(p[1]*preview.height/oh)) for p in poly]
                        draw.line(pts+[pts[0]], fill='#F20A45', width=3)
                        for pt in pts:
                            draw.ellipse((pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4),fill='#FFC000')
                draw.text((left+10, top+310), f"Foto {rec['width']}x{rec['height']} | {rec['annotation_path']}",fill='#4b5865')
            except Exception as exc:
                draw.text((left+10, top+100), f'No se pudo abrir JPG: {type(exc).__name__}', fill='#ad2a2a')
            draw.line([(left,top+ch-1),(left+cw,top+ch-1)], fill='#cbd3dd', width=1)
        name = f'polygon_samples_{sheet_idx:02d}.jpg'
        canvas.save(out/name, quality=85)
        names.append(name)
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', type=Path, default=ZIP_DEFAULT)
    ap.add_argument('--output', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sample-limit', type=int, default=36)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    issues = []
    counts = Counter()
    with zipfile.ZipFile(args.zip) as zf:
        entries = set(zf.namelist())
        json_files = sorted(name for name in entries if name.lower().endswith('.json') and ('/train/' in name or '/test/' in name))
        for idx, jp in enumerate(json_files,1):
            split = 'train' if '/train/' in jp else 'test'
            ann = json.loads(zf.read(jp).decode('utf-8-sig'))
            img = str(PurePosixPath(jp).with_suffix('.jpg'))
            if img not in entries:
                raise FileNotFoundError(f'Sin JPG con mismo nombre: {jp}')
            w, h = ann['imageWidth'], ann['imageHeight']
            for lp_index, lp in enumerate(ann.get('lps', [])):
                counts['plates'] += 1
                issue_type, outside_kinds, overrun = analyze(lp.get('poly_coord'), w, h)
                if not issue_type:
                    continue
                poly = lp.get('poly_coord')
                issues.append(dict(
                    split=split, annotation_path=jp, image_path=img,
                    lp_index=lp_index, lp_id=str(lp.get('lp_id','')),
                    width=w, height=h,
                    issue_type=issue_type,
                    out_of_bounds=';'.join(outside_kinds),
                    max_overrun_px=round(overrun,2) if overrun is not None else '',
                    poly_points=json.dumps(poly, ensure_ascii=False),
                ))
                counts[f'issues_{split}'] += 1
                counts[f'type_{issue_type}'] += 1
                for kind in outside_kinds:
                    counts[f'kind_{kind}'] += 1
            counts['json'] += 1
            if idx%500 == 0:
                print(f'JSON inspeccionados: {idx}/{len(json_files)}',flush=True)
        fields=['split','annotation_path','image_path','lp_index','lp_id','width','height','issue_type','out_of_bounds','max_overrun_px','poly_points']
        write_csv(args.output/'polygon_issue_details.csv', fields, issues)
        rng = random.Random(args.seed)
        # Casos mas extremos + muestra reproducible del resto, sin repetir imagen/poligono.
        ordered = sorted(issues,key=lambda r: float(r['max_overrun_px']) if r['max_overrun_px']!='' else float('inf'),reverse=True)
        top_count=min(len(ordered), args.sample_limit//2)
        chosen=ordered[:top_count]
        rest=ordered[top_count:]
        chosen.extend(rng.sample(rest,min(args.sample_limit-len(chosen),len(rest))))
        # Ordenamos por particion y ruta para facilitar la comparacion.
        chosen.sort(key=lambda r:(r['split'], r['annotation_path'],r['lp_index']))
        names = render_sheets(zf,chosen,args.output)
    summary = [
        '=== UC3M-LP: DIAGNOSTICO DE POLIGONOS ===',
        f'JSON revisados: {counts["json"]}',
        f'Matriculas anotadas: {counts["plates"]}',
        f'Poligonos que no pasan la validacion de limites: {len(issues)}',
        f' - Train: {counts["issues_train"]}',
        f' - Test: {counts["issues_test"]}',
        'Clasificacion de incidencias:',
    ]
    for code in ('poligono_malformado','coordenadas_malformadas','parcialmente_fuera','totalmente_fuera'):
        summary.append(f' - {code}: {counts[f"type_{code}"]}')
    summary.append('Tipos de limites incumplidos (pueden solaparse):')
    for code in ('x_negativa','y_negativa','x_supera_ancho','y_supera_alto'):
        summary.append(f' - {code}: {counts[f"kind_{code}"]}')
    summary += [
        f'Casos visualizados: {len(chosen)}',
        f'Hojas generadas: {", ".join(names) if names else "ninguna"}',
        'Los poligonos fuera de los limites NO se han reparado automaticamente.',
        'No se han modificado JPG, JSON ni manifiestos.',
        '=== FINALIZADO ===',
    ]
    (args.output/'polygon_diagnostic_summary.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8')
    print('\n'.join(summary),flush=True)
    print('Archivos:',args.output)


if __name__=='__main__':
    main()
