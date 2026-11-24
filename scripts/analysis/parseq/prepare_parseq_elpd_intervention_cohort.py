#!/usr/bin/env python3
"""Paso 6A · PARSeq / ELPD V17: cohorte para intervenciones visuales.

Prepara una muestra reproducible de inserciones puras y predicciones correctas
para estudiar el papel de bandas, escudos y otras regiones graficas.

NO aplica mascaras ni ejecuta inferencia: antes de enmascarar nuevas imagenes
hay que validar visualmente que las ROI no cubren ningun caracter del GT.

Uso en Legio2 (sin argumentos):
  python scripts/analysis/parseq/prepare_parseq_elpd_intervention_cohort.py

Prueba solo de metadatos (sin imagenes):
  python .../prepare_parseq_elpd_intervention_cohort.py --prepare-only

No modifica checkpoints, anotaciones, resultados de baselines ni imagenes originales.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import shutil

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_CSV = PROJECT / 'results/audit/ELPD/gt_corrections_v17/parseq/predictions_verified_v17.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/cohorte_intervenciones_v1'
SEED = 42
N_PER_INSERTION = 8
N_CONTROLS = 16
PRIOR_STUDIED = {
    'images/train__ABA1234.JPG',
    'images/test__DANKE82.png',
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def insertion_kind(gt: str, pred: str) -> str | None:
    """Exactamente el criterio de insercion pura usado en el paso 3."""
    if len(pred) <= len(gt):
        return None
    ops = [op for op in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes()
           if op[0] != 'equal']
    if not ops or any(op[0] != 'insert' for op in ops):
        return None
    if len(ops) == 1 and ops[0][1] == 0:
        return 'inicial'
    if len(ops) == 1 and ops[0][1] == len(gt):
        return 'final'
    return 'interna'


def safe_image(root: Path, rel: str) -> Path:
    r = Path(rel)
    if r.is_absolute() or '..' in r.parts or len(r.parts) != 2 or r.parts[0] != 'images':
        raise ValueError(f'Ruta inesperada en el CSV: {rel}')
    p = (root / r).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise FileNotFoundError(f'Imagen ELPD no encontrada: {p}')
    return p


def load_v17(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    if not path.is_file():
        raise FileNotFoundError(f'CSV V17 no disponible: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        needed = {'image_path', 'ground_truth', 'prediction_normalized', 'exact_match', 'scenario'}
        if not needed.issubset(rdr.fieldnames or []):
            raise ValueError(f'Faltan columnas del CSV: {needed - set(rdr.fieldnames or [])}')
        rows = list(rdr)
    if len(rows) != 733 or len({r['image_path'] for r in rows}) != 733:
        raise RuntimeError('Se esperan 733 casos unicos del GT verificado V17')
    if Counter(r['exact_match'] for r in rows) != {'1': 517, '0': 216}:
        raise RuntimeError('El CSV no reproduce la evaluacion final de PARSeq (517/216)')
    by_kind: dict[str, list[dict]] = {'inicial': [], 'interna': [], 'final': [], 'correcta': []}
    for r in rows:
        if r['exact_match'] == '1':
            if r['ground_truth'] != r['prediction_normalized']:
                raise RuntimeError(f'Caso correcto inconsistente: {r["image_path"]}')
            by_kind['correcta'].append(r)
        else:
            kind = insertion_kind(r['ground_truth'], r['prediction_normalized'])
            if kind:
                by_kind[kind].append(r)
    if {k: len(by_kind[k]) for k in ('inicial', 'interna', 'final')} != {
        'inicial': 46, 'interna': 34, 'final': 47
    }:
        raise RuntimeError('El recuento de inserciones puras difiere del paso 3')
    return rows, by_kind


def select_cases(by_kind: dict[str, list[dict]], seed: int,
                 per_kind: int, n_controls: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    for kind in ('inicial', 'interna', 'final'):
        eligible = [r for r in by_kind[kind] if r['image_path'] not in PRIOR_STUDIED]
        if len(eligible) < per_kind:
            raise ValueError(f'Insuficientes casos de insercion {kind}')
        for r in rng.sample(sorted(eligible, key=lambda v: v['image_path']), per_kind):
            cases.append({'data': r, 'group': 'insercion', 'type': kind,
                          'matched_to': '', 'matching': ''})
    # Seleccionar un grupo de controles emparejados por longitud y, si es posible, split.
    errors_for_match = rng.sample(cases, n_controls)
    available_controls = sorted(by_kind['correcta'], key=lambda v: v['image_path'])
    for error in errors_for_match:
        reference = error['data']
        target_length = len(reference['ground_truth'])
        same_length_and_split = [r for r in available_controls
                                 if len(r['ground_truth']) == target_length
                                 and r['scenario'] == reference['scenario']]
        same_length = [r for r in available_controls
                       if len(r['ground_truth']) == target_length]
        if same_length_and_split:
            pool, match = same_length_and_split, 'longitud_y_split'
        elif same_length:
            pool, match = same_length, 'solo_longitud'
        else:
            # La ausencia de controles de esa longitud invalida el emparejamiento.
            raise RuntimeError(f'No hay control correcto con longitud {target_length}')
        control = rng.choice(pool)
        available_controls.remove(control)
        cases.append({'data': control, 'group': 'correcta_control', 'type': 'correcta',
                      'matched_to': reference['image_path'], 'matching': match})
    if len({c['data']['image_path'] for c in cases}) != len(cases):
        raise RuntimeError('Hay imagenes duplicadas en la seleccion')
    summary = {'selection_seed': seed, 'insertions_per_type': per_kind, 'controls': n_controls,
               'matched_controls': dict(Counter(c['matching'] for c in cases
                                                 if c['group'] == 'correcta_control')),
               'excluded_from_new_sample': sorted(PRIOR_STUDIED)}
    return cases, summary


def font(size: int):
    from PIL import ImageFont
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def sheet(rows: list[dict], out: Path, sheet_number: int):
    """Hoja visual para revisar graficos sin alterar ni anotar los pixeles."""
    from PIL import Image, ImageDraw, ImageOps
    width, height = 1320, 1090
    canvas = Image.new('RGB', (width, height), (252, 252, 252))
    d = ImageDraw.Draw(canvas)
    d.text((26, 14), f'ELPD V17 · PARSeq · PASO 6A · Hoja {sheet_number:02d}',
           font=font(24), fill=(23, 37, 48))
    d.text((26, 49), 'Revisar banda azul / escudos / otros graficos antes de definir mascaras.',
           font=font(16), fill=(75, 79, 88))
    tile_w, tile_h, pad_x, pad_y, top = 624, 230, 20, 20, 87
    for k, r in enumerate(rows):
        left = pad_x + (k % 2) * (tile_w + pad_x)
        upper = top + (k // 2) * (tile_h + pad_y)
        d.rounded_rectangle((left, upper, left + tile_w, upper + tile_h), radius=10,
                            outline=(194, 201, 210), fill=(255, 255, 255), width=2)
        tag = 'ERROR' if r['group'] == 'insercion' else 'CONTROL'
        color = (157, 57, 41) if tag == 'ERROR' else (38, 109, 84)
        d.text((left + 13, upper + 9), f'{r["id"]} · {tag} · {r["type"]}',
               font=font(17), fill=color)
        d.text((left + 13, upper + 37),
               f'GT: {r["ground_truth"]}    OCR: {r["prediction_normalized"]}',
               font=font(18), fill=(24, 41, 62))
        image = r['image']
        # Mantener proporcion y no recortar caracteres; fondo uniforme para lectura.
        target_box = (tile_w - 25, 120)
        preview = ImageOps.contain(image, target_box, Image.Resampling.NEAREST)
        x = left + (tile_w - preview.width) // 2
        y = upper + 72 + (120 - preview.height) // 2
        canvas.paste(preview, (x, y))
        d.text((left + 13, upper + 204),
               f'{r["scenario"]} · {r["image_path"].split("/")[-1][:53]}',
               font=font(13), fill=(105, 110, 120))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def run(args):
    from PIL import Image
    rows, by_kind = load_v17(args.csv)
    cases, selection_summary = select_cases(by_kind, args.seed,
                                             args.per_kind, args.n_controls)
    if args.out.exists():
        raise FileExistsError(f'El directorio ya existe; no sobrescribimos: {args.out}')
    args.out.mkdir(parents=True)
    records = []
    for i, selected in enumerate(cases, 1):
        r = selected['data']
        rid = ('E' if selected['group'] == 'insercion' else 'C') + f'{i:03d}'
        records.append({
            'id': rid,
            'image_path': r['image_path'],
            'scenario': r['scenario'],
            'group': selected['group'],
            'type': selected['type'],
            'ground_truth': r['ground_truth'],
            'prediction_normalized': r['prediction_normalized'],
            'original_exact_match': r['exact_match'],
            'gt_length': len(r['ground_truth']),
            'pred_length': len(r['prediction_normalized']),
            'matched_to': selected['matched_to'],
            'matching': selected['matching'],
            'gt_validation_status': r.get('gt_validation_status_v17', ''),
        })
    core_fields = ['id', 'image_path', 'scenario', 'group', 'type', 'ground_truth',
                   'prediction_normalized', 'original_exact_match', 'gt_length', 'pred_length',
                   'matched_to', 'matching', 'gt_validation_status']
    write_csv(args.out / 'seleccion_casos.csv', records, core_fields)
    roi_fields = ['id', 'image_path', 'group', 'type', 'roi_estado', 'elemento_grafico',
                  'objetivo_x0', 'objetivo_y0', 'objetivo_x1', 'objetivo_y1',
                  'control_x0', 'control_y0', 'control_x1', 'control_y1',
                  'observaciones']
    roi_template = [dict(id=r['id'], image_path=r['image_path'], group=r['group'],
                         type=r['type'], roi_estado='pendiente_revision',
                         elemento_grafico='', objetivo_x0='', objetivo_y0='',
                         objetivo_x1='', objetivo_y1='', control_x0='', control_y0='',
                         control_x1='', control_y1='', observaciones='') for r in records]
    write_csv(args.out / 'plantilla_roi.csv', roi_template, roi_fields)
    count_selected = Counter(r['type'] for r in records)
    summary = {
        'paso': '6A', 'dataset': 'ELPD V17 verificado', 'model': 'PARSeq pretrained',
        'baseline_images': len(rows), 'baseline_correct': len(by_kind['correcta']),
        'baseline_incorrect': 216,
        'baseline_pure_insertions': {k: len(by_kind[k]) for k in ('inicial','interna','final')},
        'cohort_total': len(records), 'cohort_by_type': dict(count_selected),
        'cohort_unique_paths': len({r['image_path'] for r in records}),
        'csv_reference_sha256': file_sha256(args.csv),
        'original_weights_or_gt_changed': False, 'inference_executed': False,
        'roi_coordinates_reference': 'Image resized to 128x32 px; x0,y0 inclusive; x1,y1 exclusive',
        'selection': selection_summary,
        'note': 'Muestra exploratoria, no aleatoria representativa del conjunto completo. '
                'ROI pendiente de revision humana antes de enmascarar. '
                'No comparar correcciones de las imagenes seleccionadas con metricas independientes.',
    }
    (args.out / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n',
                                           encoding='utf-8')
    print('=== PASO 6A | PARSEQ ELPD V17 | SELECCION Y REVISION DE ROI ===', flush=True)
    print('Baseline:', len(rows), 'imagenes; correctas:', len(by_kind['correcta']),
          '| inserciones puras:', summary['baseline_pure_insertions'], flush=True)
    print('Seleccion:', len(records), '| por grupo:', dict(count_selected), flush=True)
    print('CSV de seleccion y plantilla ROI creados', flush=True)
    if args.prepare_only:
        print('PREPARACION METADATOS COMPLETADA (NO SE HAN CARGADO IMAGENES)', flush=True)
        return
    out_images = args.out / 'imagenes_originales'
    out_sheets = args.out / 'hojas'
    out_images.mkdir()
    out_sheets.mkdir()
    for r in records:
        path = safe_image(args.image_root, r['image_path'])
        # Preservar los bytes originales y su extension.
        target = out_images / f'{r["id"]}{path.suffix}'
        shutil.copy2(path, target)
        with Image.open(path) as im:
            im.load()
            r['image'] = im.convert('RGB')
    for batch_start in range(0, len(records), 8):
        sheet_no = batch_start // 8 + 1
        sheet(records[batch_start:batch_start+8], out_sheets / f'hoja_{sheet_no:02d}.png', sheet_no)
    print(f'[OK] Imagenes originales copiadas: {len(records)}', flush=True)
    print(f'[OK] Hojas de contacto: {(len(records)+7)//8}', flush=True)
    print('RESULTADO: COHORTE PREPARADA; ROI PENDIENTES DE VALIDACION', flush=True)
    print('SALIDA:', args.out, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--per-kind', type=int, default=N_PER_INSERTION)
    ap.add_argument('--n-controls', type=int, default=N_CONTROLS)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    if args.per_kind < 1 or args.n_controls < 1 or args.n_controls > 3 * args.per_kind:
        ap.error('Se requieren casos por tipo >=1 y controles entre 1 y 3*casos_por_tipo')
    run(args)


if __name__ == '__main__':
    main()
