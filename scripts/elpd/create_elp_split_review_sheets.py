#!/usr/bin/env python3
"""Generate visual review sheets for ELPD IDs repeated across original splits.

Read-only with respect to ELPD_LPO and its manifest. Run on Legio2 compute node.
"""
import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_DATASET = Path('/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO')
DEFAULT_REPORT = Path('/home/miguel.sanchez/ocr_matriculas/results/audit/ELPD/split_review')


def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description='ELPD: contact sheets for cross-split plate IDs')
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--groups-per-sheet', type=int, default=5)
    args = parser.parse_args()

    if args.groups_per_sheet < 1:
        parser.error('--groups-per-sheet must be at least 1')
    conflict_file = args.report_dir / 'split_conflicts.csv'
    if not conflict_file.is_file():
        raise SystemExit(f'ERROR: no existe {conflict_file}. Ejecuta antes analyze_elp_split_leakage.py.')
    with conflict_file.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected = {'plate_id_candidate', 'image_paths', 'train_original', 'val_original', 'test_original', 'split_proposed'}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise SystemExit(f'ERROR: columnas CSV insuficientes. Requeridas: {sorted(expected)}')
        groups = list(reader)

    if not groups:
        raise SystemExit('ERROR: no hay conflictos que revisar.')
    group_paths = []
    for group in groups:
        paths = [p.strip() for p in group['image_paths'].split('|') if p.strip()]
        if not paths:
            raise SystemExit(f"ERROR: {group['plate_id_candidate']} no tiene rutas de imagen")
        checked = []
        for rel in paths:
            path = args.dataset_root / rel
            if not path.is_file():
                raise SystemExit(f'ERROR: falta imagen {path}')
            checked.append((rel, path))
        group_paths.append(checked)

    out = args.report_dir / 'visual_review'
    out.mkdir(parents=True, exist_ok=True)
    W = 1460
    HEADER_H = 86
    ROW_H = 207
    BOTTOM_H = 42
    PADDING_X = 18
    CARD_START_X = 240
    CARD_W = 386
    CARD_H = 174
    IMAGE_W = 360
    IMAGE_H = 127
    f_title, f_group, f_text, f_small = font(23), font(20), font(16), font(14)
    count = 0

    for sheet_i in range(math.ceil(len(groups)/args.groups_per_sheet)):
        start = sheet_i * args.groups_per_sheet
        end = min(start + args.groups_per_sheet, len(groups))
        canvas = Image.new('RGB', (W, HEADER_H + (end-start)*ROW_H + BOTTOM_H), '#f3f5f8')
        draw = ImageDraw.Draw(canvas)
        draw.text((PADDING_X, 14), 'ELPD - Revision visual de matrículas entre splits', fill='#182b40', font=f_title)
        draw.text((PADDING_X, 49), f'Hoja {sheet_i+1} de {math.ceil(len(groups)/args.groups_per_sheet)} | grupos {start+1}-{end} de {len(groups)} | sin cambios en dataset', fill='#4b5f72', font=f_text)
        for local_i, ix in enumerate(range(start, end)):
            group = groups[ix]
            plate = group['plate_id_candidate']
            y = HEADER_H + local_i*ROW_H
            draw.rectangle((10, y + 2, W-10, y + ROW_H-7), fill='white', outline='#d9e1e8', width=2)
            draw.text((PADDING_X, y+15), plate, fill='#102f4b', font=f_group)
            splits = ' / '.join(k for k in ('train','val','test') if int(group[f'{k}_original']) > 0)
            draw.text((PADDING_X, y+50), f'Splits: {splits}', fill='#3d536a', font=f_small)
            draw.text((PADDING_X, y+73), f"Propuesta: {group['split_proposed']}", fill='#5c5760', font=f_small)
            draw.text((PADDING_X, y+100), f"Fotos: {len(group_paths[ix])}", fill='#5c5760', font=f_small)
            if len(group_paths[ix]) > 3:
                raise SystemExit(f'ERROR: {plate} tiene {len(group_paths[ix])} fotos; ampliar el diseño antes de usar')
            for j, (rel, path) in enumerate(group_paths[ix]):
                cx = CARD_START_X + j*CARD_W
                cy = y+10
                draw.rectangle((cx, cy, cx+CARD_W-8, cy+CARD_H), fill='#edf1f6', outline='#bac9d6')
                with Image.open(path) as original:
                    img = ImageOps.exif_transpose(original).convert('RGB')
                    ow, oh = img.size
                    ratio = min(IMAGE_W / ow, IMAGE_H / oh)
                    dims = (max(1, round(ow * ratio)), max(1, round(oh * ratio)))
                    filt = Image.Resampling.NEAREST if ratio >= 1 else Image.Resampling.LANCZOS
                    img = img.resize(dims, filt)
                    x_img = cx + 8 + (IMAGE_W - img.width)//2
                    y_img = cy + 7 + (IMAGE_H - img.height)//2
                    canvas.paste(img, (x_img, y_img))
                file_name = Path(rel).name
                draw.text((cx+10, cy+138), file_name[:35], fill='#273f54', font=f_small)
                draw.text((cx+10, cy+155), f'Original: {ow}x{oh}', fill='#546c80', font=f_small)
                count += 1
        draw.text((PADDING_X, canvas.height-31), 'Comparar visualmente: mismo vehiculo, mismo texto, duplicados o capturas diferentes.', fill='#40566b', font=f_small)
        dest = out/f'conflicts_{sheet_i+1:02d}.jpg'
        canvas.save(dest, quality=93, subsampling=0)
        print(f'Generada: {dest}')

    notes = out / 'split_review_notes.csv'
    if not notes.exists():
        with notes.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['plate_id_candidate','same_vehicle_visual','labels_match_visual','approved_split','notes'])
            writer.writeheader()
            for group in groups:
                writer.writerow({'plate_id_candidate':group['plate_id_candidate'],
                                 'same_vehicle_visual':'pending', 'labels_match_visual':'pending',
                                 'approved_split':'', 'notes':''})
        print(f'Plantilla de revision: {notes}')
    else:
        print(f'Plantilla existente conservada sin cambios: {notes}')

    print('\n=== REVISION VISUAL DE SOLAPAMIENTOS ELPD ===')
    print(f'IDs compartidos: {len(groups)}')
    print(f'Imagenes incluidas: {count}')
    print(f'Hojas generadas: {math.ceil(len(groups)/args.groups_per_sheet)}')
    print(f'Directorio: {out}')
    print('No se han modificado imagenes, particiones ni manifest.csv.')
    print('=== FINALIZADO ===')


if __name__ == '__main__':
    main()
