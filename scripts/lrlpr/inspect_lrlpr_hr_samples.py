#!/usr/bin/env python3
"""Audita anotaciones HR de LRLPR-26 y genera muestras visuales sin alterar datos."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/LRLPR-26/selected_hr')
OUTPUT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/LRLPR-26')
EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
SCENARIOS = ('Scenario-A', 'Scenario-B')
LAYOUTS = ('Brazilian', 'Mercosur')
CORNERS = ('top-left', 'top-right', 'bottom-right', 'bottom-left')


def font(size):
    for name in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def pick_two(items):
    items = sorted(items, key=lambda r: r['track'])
    if not items:
        return []
    if len(items) == 1:
        return items
    return [items[0], items[len(items) // 2]]


def make_sheet(scenario, rows, destination, samples):
    cw, ch, cols = 286, 190, 5
    canvas = Image.new('RGB', (cw * cols, ch * len(rows) + 52), '#f4f5f7')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 14), f'LRLPR-26 | {scenario} | muestras HR (sin editar)', font=font(19), fill='#162e3f')
    for row_idx, rec in enumerate(rows):
        for col_idx, image_path in enumerate(rec['images']):
            x, y = col_idx * cw, 52 + row_idx * ch
            draw.rectangle((x + 5, y + 4, x + cw - 5, y + ch - 4), fill='white', outline='#c9d0d7', width=1)
            header = f"{rec['layout']} | {rec['track']} | {rec['plate_text']}"
            draw.text((x + 12, y + 11), header[:42], font=font(12), fill='#1e2f42')
            with Image.open(image_path) as im:
                im.load()
                w, h = im.size
                pic = ImageOps.contain(im.convert('RGB'), (cw - 30, 125), method=Image.Resampling.NEAREST)
                iw, ih = pic.size
                ix, iy = x + (cw - iw) // 2, y + 39 + (125 - ih) // 2
                canvas.paste(pic, (ix, iy))
                original_poly = rec['corners'].get(image_path.name)
                if isinstance(original_poly, dict) and all(k in original_poly for k in CORNERS):
                    try:
                        poly = [(ix + float(original_poly[k][0]) * iw / w,
                                 iy + float(original_poly[k][1]) * ih / h) for k in CORNERS]
                        draw.line(poly + [poly[0]], fill='#ef6a3d', width=2, joint='curve')
                    except (ValueError, TypeError, KeyError, ZeroDivisionError):
                        pass
            draw.text((x + 12, y + ch - 22), f'{image_path.name} | {w}x{h}', font=font(11), fill='#465d70')
            samples.append({'scenario': scenario, 'layout': rec['layout'],
                            'track_id': rec['track'], 'plate_text': rec['plate_text'],
                            'image_path': str(image_path), 'width': w, 'height': h,
                            'has_corners': str(isinstance(rec['corners'].get(image_path.name), dict)).lower()})
    canvas.save(destination, quality=91, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DATA_ROOT)
    ap.add_argument('--output', type=Path, default=OUTPUT_ROOT)
    args = ap.parse_args()
    root, output = args.root, args.output
    annotation_root = root / 'train' if (root / 'train').is_dir() else root
    if not annotation_root.is_dir():
        raise FileNotFoundError(f'No existe el directorio: {annotation_root}')
    json_paths = sorted(annotation_root.rglob('annotations.json'))
    if not json_paths:
        raise RuntimeError(f'No se han encontrado annotations.json en: {annotation_root}')

    groups = defaultdict(list)
    counts = Counter()
    labels = Counter()
    problems = []

    for annotation_path in json_paths:
        relative = annotation_path.relative_to(annotation_root)
        parts = relative.parts
        scenario = next((part for part in parts if part in SCENARIOS), None)
        if scenario is None:
            problems.append(f'Ruta sin escenario conocido: {relative}')
            continue
        layout_dir = next((part for part in parts if part in LAYOUTS), None)
        track = annotation_path.parent.name
        try:
            with annotation_path.open('r', encoding='utf-8') as f:
                ann = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f'JSON no válido: {relative}: {exc}')
            continue
        if not isinstance(ann, dict):
            problems.append(f'Anotación no es objeto: {relative}')
            continue
        layout = ann.get('plate_layout', '')
        plate_text = ann.get('plate_text', '')
        corners = ann.get('corners', {})
        if not isinstance(corners, dict):
            problems.append(f'corners no es objeto: {relative}')
            corners = {}
        if not isinstance(plate_text, str) or not plate_text.strip():
            problems.append(f'plate_text vacío/no texto: {relative}')
            plate_text = ''
        if layout != layout_dir:
            problems.append(f'Layout distinto a carpeta: {relative} ({layout!r} != {layout_dir!r})')
        images = sorted(p for p in annotation_path.parent.iterdir()
                        if p.is_file() and p.name.lower().startswith('hr-') and p.suffix.lower() in EXTS)
        counts[(scenario, 'tracks')] += 1
        counts[(scenario, 'hr_images')] += len(images)
        counts[(scenario, 'tracks_with_corners')] += bool(corners)
        counts[(scenario, 'tracks_without_corners')] += not bool(corners)
        counts[(scenario, layout, 'tracks')] += 1
        if plate_text:
            labels[plate_text.upper().strip()] += 1
        if len(images) != 5:
            problems.append(f'{relative}: {len(images)} archivos HR; esperados 5')
        if scenario == 'Scenario-A':
            missing = [p.name for p in images if p.name not in corners]
            if missing:
                problems.append(f'{relative}: faltan esquinas HR: {missing}')
        groups[(scenario, layout)].append({
            'track': track, 'plate_text': plate_text, 'layout': layout,
            'images': images, 'corners': corners,
        })

    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'hr_annotation_summary.txt'
    lines = ['=== LRLPR-26: AUDITORIA DE HR Y ANOTACIONES ===',
             f'Anotaciones inspeccionadas: {len(json_paths)}']
    for sc in SCENARIOS:
        lines.append(f'{sc}: {counts[(sc, "tracks")]} tracks, {counts[(sc, "hr_images")]} HR, '
                     f'con esquinas: {counts[(sc, "tracks_with_corners")]}, '
                     f'sin esquinas: {counts[(sc, "tracks_without_corners")]}')
        for layout in LAYOUTS:
            lines.append(f'  {layout}: {counts[(sc, layout, "tracks")]} tracks')
    lines += [f'Etiquetas de matricula distintas (por texto): {len(labels)}',
              f'Etiquetas presentes en más de un track: {sum(n > 1 for n in labels.values())}',
              f'Anomalías detectadas: {len(problems)}']
    if problems:
        lines.append('Primeras 40 anomalías:')
        lines.extend(f'- {msg}' for msg in problems[:40])
    selected = []
    try:
        for scenario, suffix in [('Scenario-A', 'a'), ('Scenario-B', 'b')]:
            chosen = []
            for layout in LAYOUTS:
                chosen.extend(pick_two(groups[(scenario, layout)]))
            if not chosen:
                lines.append(f'AVISO: no hay tracks para {scenario}')
                continue
            filename = output / f'hr_samples_scenario_{suffix}.jpg'
            make_sheet(scenario, chosen, filename, selected)
            lines.append(f'Hoja generada: {filename}')
    except (OSError, ValueError) as exc:
        lines.append(f'ERROR generando muestras visuales: {exc}')
        report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        raise
    index = output / 'hr_sample_index.csv'
    with index.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario', 'layout', 'track_id', 'plate_text',
                                                'image_path', 'width', 'height', 'has_corners'])
        writer.writeheader()
        writer.writerows(selected)
    lines.append(f'Índice de muestras: {index}')
    lines.append('No se han modificado las imágenes ni las anotaciones originales.')
    lines.append('=== FINALIZADO ===')
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
