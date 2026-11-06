#!/usr/bin/env python3
"""Genera recortes LPO y un manifiesto trazable de UC3M-LP.

- Lee directamente el ZIP original (sin extraer fotos completas).
- Asocia imagen/JSON por nombre homónimo, no por imagePath.
- Admite polígonos de 4, 5, 6 o más vértices (bbox de todos sus puntos).
- Conserva máscaras '*' y GT parcial sin reconstruir caracteres ocultos.
- No altera ZIP, fotografías ni JSON originales.

Ejemplo SLURM: python scripts/uc3m/prepare_uc3m_lpo.py
"""
import argparse
import csv
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
COLUMNS = [
    'image_path', 'source_path', 'annotation_path', 'text_original',
    'text_normalized', 'text_visible', 'known_mask', 'dataset', 'track_id',
    'vehicle_id', 'scenario', 'view_id', 'label_status', 'crop_method',
    'width', 'height', 'polygon_vertices', 'character_count',
    'known_characters', 'anonymized_characters',
]
ISSUE_COLUMNS = ['annotation_path', 'source_path', 'lp_index', 'issue', 'detail']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path,
                   default=DATA_ROOT / 'raw/UC3M-LP/UC3M-LP.zip')
    p.add_argument('--out-dir', type=Path,
                   default=DATA_ROOT / 'processed/UC3M-LP_LPO')
    p.add_argument('--report-dir', type=Path,
                   default=PROJECT_ROOT / 'results/inspection/UC3M-LP/lpo_preparation')
    p.add_argument('--margin-frac', type=float, default=0.04,
                   help='Margen relativo al ancho/alto de la matrícula (0.04 = 4%%)')
    p.add_argument('--expected-count', type=int, default=2547)
    return p.parse_args()


def polygon_box(points, image_size, margin):
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError('polígono con menos de 3 puntos')
    xy = []
    for p in points:
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
            raise ValueError('coordenadas de polígono no válidas')
        xy.append((float(p[0]), float(p[1])))
    x0, x1 = min(p[0] for p in xy), max(p[0] for p in xy)
    y0, y1 = min(p[1] for p in xy), max(p[1] for p in xy)
    w, h = image_size
    if x1 <= x0 or y1 <= y0:
        raise ValueError('polígono degenerado')
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise ValueError('polígono excede los límites de imagen')
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (max(0, math.floor(x0 - mx)), max(0, math.floor(y0 - my)),
           min(w, math.ceil(x1 + mx)), min(h, math.ceil(y1 + my)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError('recorte demasiado pequeño')
    return box


def atomic_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    a = parse_args()
    if not 0 <= a.margin_frac <= .5:
        raise ValueError('--margin-frac debe estar entre 0 y 0.5')
    if not a.zip.is_file():
        raise FileNotFoundError(a.zip)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = a.out_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    a.report_dir.mkdir(parents=True, exist_ok=True)

    rows, issues = [], []
    splits = Counter()
    verts = Counter()
    labels = Counter()
    read_images = 0
    reused_images = 0
    with zipfile.ZipFile(a.zip) as z:
        names = set(z.namelist())
        json_names = sorted(n for n in names if n.lower().endswith('.json')
                            and n.startswith('UC3M-LP/')
                            and n.split('/')[1] in {'train', 'test'})
        print('=== UC3M-LP: PREPARACION LPO ===', flush=True)
        print('ZIP:', a.zip, flush=True)
        print('JSON encontrados:', len(json_names), flush=True)
        for ji, ann_name in enumerate(json_names, 1):
            rel = Path(ann_name).relative_to('UC3M-LP')
            split, stem = rel.parts[0], rel.stem
            image_name = str(Path(ann_name).with_suffix('.jpg'))
            if image_name not in names:
                issues.append(dict(annotation_path=ann_name, source_path=image_name,
                                   lp_index='', issue='missing_image', detail='JPG homónimo no encontrado'))
                continue
            try:
                ann = json.loads(z.read(ann_name).decode('utf-8-sig'))
                lps = ann['lps']
                if not isinstance(lps, list):
                    raise ValueError('lps no es una lista')
            except (ValueError, KeyError, UnicodeDecodeError) as exc:
                issues.append(dict(annotation_path=ann_name, source_path=image_name,
                                   lp_index='', issue='invalid_json', detail=str(exc)))
                continue
            # Una sola descompresión de cada fotografía, aunque tenga varias matrículas.
            try:
                with z.open(image_name) as image_file:
                    with Image.open(image_file) as image:
                        image.load()
                        if (image.width != ann.get('imageWidth')
                                or image.height != ann.get('imageHeight')):
                            raise ValueError(f'dimensiones no coinciden JPG={image.size}, JSON=' 
                                             f'{ann.get("imageWidth")}x{ann.get("imageHeight")}')
                        read_images += 1
                        for lp_i, lp in enumerate(lps):
                            try:
                                polygon = lp['poly_coord']
                                bbox = polygon_box(polygon, image.size, a.margin_frac)
                                plate_id = str(lp['lp_id']).strip()
                                # AN-/AD- es metadato, no forma parte de la matrícula.
                                plate_text = plate_id.split('-', 1)[-1].upper()
                                if not plate_text or not all(c.isalnum() or c == '*' for c in plate_text):
                                    raise ValueError(f'etiqueta no válida: {plate_id!r}')
                                known = ''.join(c for c in plate_text if c != '*')
                                masks = plate_text.count('*')
                                characters = lp.get('characters', [])
                                if not isinstance(characters, list):
                                    raise ValueError('characters no es una lista')
                                explicit_text = ''.join(str(c.get('char_id', '')) for c in characters).upper()
                                # Se registra discrepancia, pero se conserva el GT fuente.
                                if explicit_text and explicit_text != known:
                                    issues.append(dict(annotation_path=ann_name,
                                                       source_path=image_name, lp_index=lp_i,
                                                       issue='character_mismatch',
                                                       detail=f'lp_id={known}; chars={explicit_text}'))
                                file_name = f'{split}__{stem}__lp{lp_i:02d}.png'
                                dest = image_dir / file_name
                                dims = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                                if dest.is_file():
                                    with Image.open(dest) as existing:
                                        existing.verify()
                                    with Image.open(dest) as existing:
                                        if existing.size != dims:
                                            raise ValueError(f'recorte previo de tamaño distinto: {existing.size} != {dims}')
                                    reused_images += 1
                                else:
                                    cropped = image.crop(bbox)
                                    # PNG sin pérdida adicional frente a los JPG originales.
                                    cropped.save(dest, format='PNG', optimize=False)
                                    cropped.close()
                                rows.append(dict(
                                    image_path=f'images/{file_name}', source_path=image_name,
                                    annotation_path=ann_name, text_original=plate_id,
                                    text_normalized=plate_text, text_visible=known,
                                    known_mask=''.join('0' if c == '*' else '1' for c in plate_text),
                                    dataset='UC3M-LP', track_id='', vehicle_id='',
                                    scenario=split, view_id=f'{stem}__lp{lp_i:02d}',
                                    label_status='partial_masked' if masks else 'source_annotation_unverified',
                                    crop_method='polygon_bbox_4pct_margin',
                                    width=dims[0], height=dims[1], polygon_vertices=len(polygon),
                                    character_count=len(plate_text), known_characters=len(known),
                                    anonymized_characters=masks,
                                ))
                                splits[split] += 1
                                verts[len(polygon)] += 1
                                labels['partial_masked' if masks else 'unmasked'] += 1
                            except (ValueError, KeyError, OSError, TypeError) as exc:
                                issues.append(dict(annotation_path=ann_name,
                                                   source_path=image_name, lp_index=lp_i,
                                                   issue='crop_error', detail=str(exc)))
            except (UnidentifiedImageError, ValueError, OSError) as exc:
                issues.append(dict(annotation_path=ann_name, source_path=image_name,
                                   lp_index='', issue='image_error', detail=str(exc)))
            if ji % 250 == 0:
                print(f'JSON procesados: {ji}/{len(json_names)} | recortes: {len(rows)} | incidencias: {len(issues)}', flush=True)

    issues_path = a.report_dir / 'lpo_preparation_issues.csv'
    atomic_csv(issues_path, ISSUE_COLUMNS, issues)
    summary = [
        '=== UC3M-LP: PREPARACION LPO ===',
        f'JSON inspeccionados: {len(json_names)}',
        f'Fotografias procesadas: {read_images}',
        f'Matriculas recortadas: {len(rows)}',
        f'Recortes ya existentes validados: {reused_images}',
        f'Train: {splits["train"]}; Test: {splits["test"]}',
        'Vertices: ' + ', '.join(f'{n}={num}' for n, num in sorted(verts.items())),
        'Etiquetas: ' + ', '.join(f'{k}={v}' for k, v in sorted(labels.items())),
        f'Incidencias: {len(issues)}',
        f'Imagenes procesadas: {image_dir}',
        'Solo se ha usado JPG homonimo de cada JSON (imagePath no utilizado).',
        'Se conservan asteriscos y posiciones de GT parcial.',
        'No se han modificado las fotografias ni anotaciones originales.',
    ]
    if issues or len(rows) != a.expected_count:
        summary.append(f'ESTADO: REVISION NECESARIA (esperadas {a.expected_count} matrículas)')
        summary.append('El manifest NO se publica mientras existan incidencias o recuento incorrecto.')
        result_ok = False
    else:
        atomic_csv(a.out_dir / 'manifest.csv', COLUMNS, rows)
        summary.append(f'Manifest: {a.out_dir / "manifest.csv"}')
        summary.append('ESTADO: PREPARACION COMPLETADA; VALIDACION VISUAL PENDIENTE')
        result_ok = True
    (a.report_dir / 'lpo_preparation_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
    print('\n'.join(summary), flush=True)
    if not result_ok:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
