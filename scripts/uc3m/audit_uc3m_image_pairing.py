#!/usr/bin/env python3
"""Audita las referencias JSON-fotografia de UC3M-LP mediante comparaciones visuales.

Lee el ZIP original SIN extraerlo. Compara, para muestras de train y test,
la foto con el mismo nombre que el JSON y la declarada en `imagePath`.
Dibuja los poligonos de `lps` sobre ambas candidatas y muestra acercamientos.
No decide automaticamente que emparejamiento es correcto y no modifica datos.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps

ZIP_DEFAULT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP/UC3M-LP.zip')
OUT_DEFAULT = Path('/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP/image_pairing')

CSV_FIELDS = [
    'split', 'json_path', 'json_stem_image', 'declared_imagePath',
    'declared_image', 'stem_exists', 'declared_exists',
    'same_candidate', 'number_of_plates', 'image_width_json', 'image_height_json',
]
SAMPLE_FIELDS = CSV_FIELDS + [
    'stem_image_size', 'declared_image_size', 'selection_reason',
    'visual_decision', 'visual_notes',
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def valid_image_path(parent: PurePosixPath, image_path: object) -> str:
    if not isinstance(image_path, str) or not image_path.strip():
        return ''
    name = PurePosixPath(image_path.replace('\\', '/'))
    if name.is_absolute() or '..' in name.parts:
        return ''
    # Los ejemplos UC3M-LP usan nombres relativos a la carpeta train/test.
    return (parent / name).as_posix()


def pick_samples(rows: list[dict], count: int, seed: int) -> list[dict]:
    by_split = {'train': [], 'test': []}
    for row in rows:
        if row['split'] in by_split:
            by_split[row['split']].append(row)
    forced = {'train': {'00001', '00677'}, 'test': {'00000', '00591'}}
    rng = random.Random(seed)
    selected = []
    for split in ('train', 'test'):
        available = by_split[split]
        chosen = []
        for stem in sorted(forced[split]):
            match = next((r for r in available if PurePosixPath(r['json_path']).stem == stem), None)
            if match:
                chosen.append((match, 'ejemplo_preferente'))
        options = [r for r in available if r not in [x[0] for x in chosen]]
        # Preferir casos en los que el nombre y imagePath discrepan y ambas fotos existen.
        conflicting = [r for r in options if r['same_candidate'] == 'no'
                       and r['stem_exists'] == 'yes' and r['declared_exists'] == 'yes']
        rng.shuffle(conflicting)
        for row in conflicting[:max(0, count - len(chosen))]:
            chosen.append((row, 'discrepancia_muestreada'))
        if len(chosen) < count:
            rest = [r for r in options if r not in [x[0] for x in chosen]]
            rng.shuffle(rest)
            for row in rest[:count - len(chosen)]:
                chosen.append((row, 'muestra_aleatoria'))
        selected.extend([{**row, 'selection_reason': reason} for row, reason in chosen])
    return selected


def decode_preview(zf: ZipFile, path: str, max_decode=(2000, 1500)):
    if not path:
        return None, ''
    try:
        raw = zf.read(path)
        with Image.open(io.BytesIO(raw)) as im:
            original_size = im.size
            # JPEG: reduce la resolucion ya en el decoder y evita memorias enormes.
            im.draft('RGB', max_decode)
            im.load()
            result = ImageOps.exif_transpose(im).convert('RGB')
            if result.width > max_decode[0] or result.height > max_decode[1]:
                result.thumbnail(max_decode, Image.Resampling.LANCZOS)
        return (result, original_size), f'{original_size[0]}x{original_size[1]}'
    except Exception as exc:
        return None, 'ERROR: ' + str(exc)[:100]


def get_polygons(annotation: dict):
    polys = []
    for lp in annotation.get('lps', []):
        if not isinstance(lp, dict):
            continue
        points = lp.get('poly_coord')
        if isinstance(points, list) and len(points) == 4 and all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)
            for p in points
        ):
            polys.append((str(lp.get('lp_id', '?')), points))
    return polys


def card(zf: ZipFile, path: str, annotation: dict, label: str, cache: dict):
    card_w, card_h = 620, 250
    canvas = Image.new('RGB', (card_w, card_h), '#f1f4f8')
    dr = ImageDraw.Draw(canvas)
    dr.rectangle((0, 0, card_w - 1, card_h - 1), outline='#b6c1cf', width=2)
    dr.text((12, 10), label, fill='#152f43')
    dr.text((12, 29), PurePosixPath(path).name if path else 'Sin ruta valida', fill='#344f63')
    if path not in cache:
        cache[path] = decode_preview(zf, path)
    loaded, dimension_string = cache[path]
    dr.text((390, 29), dimension_string[:29], fill='#4d5b6a')
    if loaded is None:
        dr.text((20, 120), 'Imagen no disponible / fallo de lectura', fill='#ab3030')
        return canvas, dimension_string

    source, orig_size = loaded
    ann_w = annotation.get('imageWidth')
    ann_h = annotation.get('imageHeight')
    if not isinstance(ann_w, (int, float)) or ann_w <= 0:
        ann_w = orig_size[0]
    if not isinstance(ann_h, (int, float)) or ann_h <= 0:
        ann_h = orig_size[1]
    # Poligonos en coordenadas del JSON; normalizamos por las dimensiones
    # del JSON para poder compararlos con ambas candidatas incluso si difieren.
    sx, sy = source.width / ann_w, source.height / ann_h
    annotated = source.copy()
    overlay = ImageDraw.Draw(annotated)
    polys = get_polygons(annotation)
    for i, (_, points) in enumerate(polys):
        pts = [(int(x * sx), int(y * sy)) for x, y in points]
        if len(pts) == 4:
            overlay.line(pts + [pts[0]], fill='#00ff44' if i == 0 else '#f9d633', width=6)
    overview = annotated.copy()
    overview.thumbnail((285, 185), Image.Resampling.LANCZOS)
    canvas.paste(overview, (12 + (285 - overview.width) // 2, 53 + (180 - overview.height) // 2))
    dr.text((18, 230), f'Poligonos: {len(polys)}', fill='#344f63')

    # Acercamiento sobre el primer poligono de matricula, con margen extra.
    if polys:
        pts = [(x * sx, y * sy) for x, y in polys[0][1]]
        left, top = min(x for x, y in pts), min(y for x, y in pts)
        right, bottom = max(x for x, y in pts), max(y for x, y in pts)
        padx = max(5, (right - left) * .35)
        pady = max(5, (bottom - top) * .65)
        rect = (max(0, int(left - padx)), max(0, int(top - pady)),
                min(source.width, int(right + padx)), min(source.height, int(bottom + pady)))
        if rect[2] > rect[0] and rect[3] > rect[1]:
            cut = annotated.crop(rect)
            cut.thumbnail((298, 178), Image.Resampling.LANCZOS)
            # Ampliar solo para la visualizacion, no modifica datos.
            if cut.width < 170 and cut.height < 100:
                factor = min(298 / cut.width, 178 / cut.height)
                cut = cut.resize((int(cut.width * factor), int(cut.height * factor)), Image.Resampling.NEAREST)
            canvas.paste(cut, (310 + (298 - cut.width) // 2, 53 + (180 - cut.height) // 2))
    dr.text((315, 230), 'Vista cercana del poligono (verde)', fill='#344f63')
    return canvas, dimension_string


def make_sheet(zf: ZipFile, selected: list[dict], docs: dict, dest: Path):
    rows = len(selected)
    h_head, h_row = 76, 280
    sheet = Image.new('RGB', (1256, h_head + h_row * rows), 'white')
    dr = ImageDraw.Draw(sheet)
    dr.text((14, 12), 'UC3M-LP | Comparacion JSON -> imagen (sin modificar datos)', fill='#152f43')
    dr.text((14, 38), 'IZQ: misma numeracion JSON/JPG | DER: campo imagePath del JSON', fill='#42556a')
    cache = {}
    for i, row in enumerate(selected):
        y = h_head + i * h_row
        name = row['json_path']
        ann = docs[name]
        split = row['split']
        dr.text((14, y), f'{i + 1:02d}  {split}/{PurePosixPath(name).name}   '
                f'imagePath={row["declared_imagePath"]!s}   matriculas={row["number_of_plates"]}',
                fill='#1c3547')
        for j, (field, label) in enumerate((('json_stem_image', 'Candidato 1: mismo nombre'),
                                           ('declared_image', 'Candidato 2: imagePath'))):
            image, size = card(zf, row[field], ann, label, cache)
            sheet.paste(image, (8 + 624 * j, y + 23))
            row['stem_image_size' if j == 0 else 'declared_image_size'] = size
        row['visual_decision'] = 'pending'  # same_stem / imagePath / ambos / ninguno / incierto
        row['visual_notes'] = ''
    sheet.save(dest, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', type=Path, default=ZIP_DEFAULT)
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--samples-per-split', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not args.zip.is_file():
        raise FileNotFoundError(f'ZIP no encontrado: {args.zip}')
    if not 1 <= args.samples_per_split <= 30:
        raise ValueError('--samples-per-split debe estar entre 1 y 30')
    args.out.mkdir(parents=True, exist_ok=True)
    print('=== UC3M-LP: AUDITORIA DE CORRESPONDENCIA JSON-IMAGEN ===', flush=True)
    rows, docs = [], {}
    with ZipFile(args.zip) as zf:
        paths = set(zf.namelist())
        js = sorted(n for n in paths if n.lower().endswith('.json')
                    and PurePosixPath(n).parent.name in {'train', 'test'})
        bad_jsons = []
        for name in js:
            try:
                doc = json.loads(zf.read(name))
                if not isinstance(doc, dict):
                    raise ValueError('Raiz no dict')
            except Exception as exc:
                bad_jsons.append((name, str(exc)))
                continue
            parent = PurePosixPath(name).parent
            same_stem = parent.joinpath(PurePosixPath(name).stem + '.jpg').as_posix()
            image_path = doc.get('imagePath', '')
            declared = valid_image_path(parent, image_path)
            row = {
                'split': parent.name,
                'json_path': name,
                'json_stem_image': same_stem,
                'declared_imagePath': image_path,
                'declared_image': declared,
                'stem_exists': 'yes' if same_stem in paths else 'no',
                'declared_exists': 'yes' if declared in paths else 'no',
                'same_candidate': 'yes' if same_stem == declared else 'no',
                'number_of_plates': len(doc.get('lps', [])) if isinstance(doc.get('lps'), list) else 0,
                'image_width_json': doc.get('imageWidth', ''),
                'image_height_json': doc.get('imageHeight', ''),
            }
            rows.append(row)
            docs[name] = doc
        write_csv(args.out / 'image_mapping_overview.csv', CSV_FIELDS, rows)
        selected = pick_samples(rows, args.samples_per_split, args.seed)
        samples = []
        for split in ('train', 'test'):
            pick = [r for r in selected if r['split'] == split]
            if pick:
                dest = args.out / f'pairing_{split}.jpg'
                make_sheet(zf, pick, docs, dest)
                print(f'Hoja generada: {dest}', flush=True)
                samples.extend(pick)
        write_csv(args.out / 'image_mapping_review.csv', SAMPLE_FIELDS, samples)
    counts = Counter(r['split'] for r in rows)
    different = Counter(r['split'] for r in rows if r['same_candidate'] == 'no')
    nonexist_stem = sum(r['stem_exists'] == 'no' for r in rows)
    nonexist_declared = sum(r['declared_exists'] == 'no' for r in rows)
    summary = [
        '=== UC3M-LP: AUDITORIA DE CORRESPONDENCIA JSON-IMAGEN ===',
        f'JSON leidos: {len(rows)}',
        f'JSON invalidos: {len(bad_jsons)}',
        f'Train: {counts["train"]}; Test: {counts["test"]}',
        f'Nombre JSON y campo imagePath diferentes: {sum(different.values())}',
        f'  Train: {different["train"]}; Test: {different["test"]}',
        f'Candidato de mismo nombre no encontrado: {nonexist_stem}',
        f'Candidato imagePath no encontrado: {nonexist_declared}',
        f'Muestras visuales generadas: {len(samples)}',
        'ATENCION: diferencias de nombres NO prueban que imagePath sea erroneo.',
        'La correspondencia debe confirmarse visualmente comparando los dos candidatos.',
        'No se han extraido ni modificado fotografias, anotaciones o manifiestos.',
        '=== FINALIZADO ===',
    ]
    (args.out / 'image_mapping_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('\n'.join(summary), flush=True)


if __name__ == '__main__':
    main()
