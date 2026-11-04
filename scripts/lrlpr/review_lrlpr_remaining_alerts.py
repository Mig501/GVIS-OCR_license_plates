#!/usr/bin/env python3
"""Genera una hoja de contacto de las 10 alertas geométricas HR pendientes.

Ejecutar en un nodo SLURM del proyecto GVIS, sin modificar datos originales.
"""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/LRLPR-26/selected_hr')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
ALERTS_CSV = PROJECT_ROOT / 'results/inspection/LRLPR-26/lpo_visual_audit/geometry_alerts.csv'
OUT_DIR = PROJECT_ROOT / 'results/inspection/LRLPR-26/lpo_visual_audit'

# Muestras no incluidas en la primera hoja de 20 alertas Brazilian.
REMAINING = [
    'train/Scenario-A/Brazilian/track_02218/hr-001.png',
    'train/Scenario-A/Brazilian/track_00228/hr-005.png',
    'train/Scenario-A/Brazilian/track_02218/hr-002.png',
    'train/Scenario-A/Brazilian/track_00908/hr-003.png',
    'train/Scenario-A/Brazilian/track_01411/hr-005.png',
    'train/Scenario-A/Brazilian/track_00811/hr-001.png',
    'train/Scenario-A/Brazilian/track_02218/hr-003.png',
    'train/Scenario-A/Brazilian/track_02218/hr-004.png',
    'train/Scenario-A/Brazilian/track_02218/hr-005.png',
    'train/Scenario-A/Brazilian/track_01269/hr-002.png',
]


def font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--alerts', type=Path, default=ALERTS_CSV)
    parser.add_argument('--out', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.alerts.is_file():
        raise FileNotFoundError(f'No existe el CSV de alertas: {args.alerts}')
    with args.alerts.open('r', encoding='utf-8-sig', newline='') as f:
        alerts = {row['path']: row for row in csv.DictReader(f)}

    if len(REMAINING) != len(set(REMAINING)):
        raise ValueError('Hay rutas duplicadas en REMAINING')
    absent = [p for p in REMAINING if p not in alerts]
    if absent:
        raise ValueError('Faltan alertas en CSV:\n' + '\n'.join(absent))

    columns, rows = 2, 5
    cell_w, cell_h, gap, margin, header_h = 550, 270, 16, 24, 74
    width = columns * cell_w + (columns - 1) * gap + 2 * margin
    height = header_h + rows * cell_h + (rows - 1) * gap + margin * 2
    canvas = Image.new('RGB', (width, height), '#f2f5f8')
    draw = ImageDraw.Draw(canvas)
    title_f = font(23)
    label_f = font(17)
    small_f = font(14)
    draw.text((margin, 20), 'LRLPR-26 | 10 alertas geometricas restantes', font=title_f, fill='#202b37')
    draw.text((margin, 49), 'Scenario-A / Brazilian | vista ampliada sin alterar datos', font=small_f, fill='#586575')

    for i, rel in enumerate(REMAINING):
        src = args.root / rel
        if not src.is_file():
            raise FileNotFoundError(f'No se encuentra imagen: {src}')
        with Image.open(src) as im:
            im.load()
            original = im.convert('RGB')
        w, h = original.size
        row, col = divmod(i, columns)
        x = margin + col * (cell_w + gap)
        y = header_h + margin + row * (cell_h + gap)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=8,
                               fill='white', outline='#bbc8d6', width=2)
        track = Path(rel).parent.name
        name = Path(rel).name
        draw.text((x + 14, y + 13), f'{i+1:02d}. {track}/{name}', font=label_f, fill='#192a3d')
        meta = alerts[rel]
        draw.text((x + 14, y + 40),
                  f'{w}x{h} px | aspecto {float(meta["aspect"]):.2f} | {meta["reasons"]}',
                  font=small_f, fill='#6a4553')
        # Escala ampliada SOLO para inspeccion; no se generan recortes ni se altera la fuente.
        preview = ImageOps.contain(original, (cell_w - 40, cell_h - 88), Image.Resampling.NEAREST)
        ix = x + (cell_w - preview.width) // 2
        iy = y + 72 + (cell_h - 88 - preview.height) // 2
        canvas.paste(preview, (ix, iy))
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / 'scenario_a_brazilian_remaining_alerts.jpg'
    canvas.save(out_path, quality=95, subsampling=0)
    print('=== LRLPR-26: REVISION DE ALERTAS PENDIENTES ===')
    print(f'Alertas del CSV: {len(alerts)}')
    print(f'Imagenes pendientes incluidas: {len(REMAINING)}')
    print(f'Hoja generada: {out_path}')
    print('Ninguna imagen original ni manifiesto modificado.')
    print('=== FINALIZADO ===')


if __name__ == '__main__':
    main()
