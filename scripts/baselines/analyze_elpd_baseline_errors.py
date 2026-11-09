#!/usr/bin/env python3
"""Analyse OCR failures and make visual review sheets for ALL ELPD baselines.

No GPU, model loading, pip installs, or changes to datasets are required.
Inputs:  results/baselines/ELPD/<model>/predictions.csv and the ELPD_LPO crops.
Outputs: results/baselines/ELPD/<model>/error_analysis/{errors.csv,
         error_analysis.md, contact_sheet_errors_*.jpg,
         contact_sheet_high_confidence_errors.jpg,
         contact_sheet_low_confidence_correct.jpg}.

Run via CPU task with no extra arguments. Optionally use --model parseq_pretrained.
The classification of edit operations is ONE valid optimal Levenshtein alignment;
there may be other equally optimal alignments, so it is descriptive, not causal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import textwrap

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATA = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
BASELINES_REL = Path('results/baselines/ELPD')
IMAGES_REL = Path('processed/ELPD_LPO')


def align_errors(gt: str, predicted: str):
    """Deterministic optimal Levenshtein edit script, gt -> predicted."""
    n, m = len(gt), len(predicted)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (gt[i - 1] != predicted[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    i, j = n, m
    changes = []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (gt[i - 1] != predicted[j - 1]):
            if gt[i - 1] != predicted[j - 1]:
                changes.append(('substitution', f'{gt[i - 1]}>{predicted[j - 1]}'))
            i -= 1
            j -= 1
        elif j and dp[i][j] == dp[i][j - 1] + 1:
            changes.append(('insertion', f'+{predicted[j - 1]}'))
            j -= 1
        else:
            changes.append(('deletion', f'-{gt[i - 1]}'))
            i -= 1
    changes.reverse()
    assert len(changes) == dp[n][m]
    return changes


def error_category(changes):
    types = {operation for operation, _ in changes}
    if not types:
        return 'correct'
    if len(types) > 1:
        return 'mixed'
    return next(iter(types))


def read_predictions(path: Path):
    required = {
        'image_path', 'ground_truth', 'prediction_normalized', 'exact_match',
        'edit_distance', 'gt_review_status', 'confidence',
    }
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: columnas ausentes: {sorted(missing)}')
        items = list(reader)
    if not items:
        raise ValueError(f'{path}: predictions.csv vacio')
    for r in items:
        image_rel = Path(r['image_path'])
        if image_rel.is_absolute() or '..' in image_rel.parts or image_rel.parts[0] != 'images':
            raise ValueError(f'Ruta insegura en predicciones: {r["image_path"]}')
        gt, pred = r['ground_truth'], r['prediction_normalized']
        if not gt or '*' in gt:
            raise ValueError(f'GT vacio o enmascarado en ELPD: {r["image_path"]}')
        changes = align_errors(gt, pred)
        dist = len(changes)
        if int(r['edit_distance']) != dist or int(r['exact_match']) != int(dist == 0):
            raise ValueError(f'Inconsistencia en distancia o EM: {r["image_path"]}')
        r['error_category'] = error_category(changes)
        r['edit_operations'] = ' '.join(detail for _, detail in changes)
        r['confidence_num'] = float(r['confidence'])
    return items


def font(size, bold=False):
    from PIL import ImageFont
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def draw_sheet(rows, outfile: Path, image_root: Path, title: str):
    from PIL import Image, ImageDraw, ImageOps
    if not rows:
        return
    ncols = 3
    nrows = 4
    w, h = 470, 245
    page_w, page_h = 3 * w + 18, 4 * h + 85
    bg = Image.new('RGB', (page_w, page_h), '#f5f6f8')
    draw = ImageDraw.Draw(bg)
    draw.text((14, 14), title, fill='#171b21', font=font(23, True))
    draw.text((14, 46), 'Recortes originales sin editar; miniaturas solo para revision visual',
              fill='#505865', font=font(13))
    for idx, r in enumerate(rows):
        x = 9 + (idx % ncols) * w
        y = 72 + (idx // ncols) * h
        draw.rounded_rectangle((x, y, x + w - 10, y + h - 10), radius=9,
                               fill='white', outline='#cbd4df', width=2)
        name = Path(r['image_path']).name
        draw.text((x + 10, y + 9), f'{name[:42]}', fill='#17202e', font=font(16, True))
        gt = r['ground_truth']
        pred = r['prediction_normalized']
        draw.text((x + 10, y + 38), f'GT:   {gt}', fill='#225c45', font=font(17, True))
        draw.text((x + 10, y + 63), f'Pred: {pred}', fill='#a33c35', font=font(17, True))
        tag = (f'D={r["edit_distance"]} | {r["error_category"]} | '
               f'conf={r["confidence_num"]:.3f}')
        draw.text((x + 10, y + 91), tag, fill='#343a43', font=font(13))
        source = image_root / r['image_path']
        if source.is_file():
            try:
                with Image.open(source) as orig:
                    image = ImageOps.contain(orig.convert('RGB'), (w - 28, 106),
                                             method=Image.Resampling.LANCZOS)
                bg.paste(image, (x + (w - 10 - image.width) // 2,
                                 y + 117 + (106 - image.height) // 2))
            except Exception as e:
                draw.text((x + 10, y + 150), 'NO SE PUDO ABRIR: ' + str(e)[:42],
                          fill='#a33c35', font=font(12))
        else:
            draw.text((x + 10, y + 150), 'IMAGEN NO ENCONTRADA',
                      fill='#a33c35', font=font(16, True))
    bg.save(outfile, quality=91, optimize=True)


def save_csv(path: Path, items):
    fields = ['image_path', 'gt_review_status', 'ground_truth', 'prediction_raw',
              'prediction_normalized', 'edit_distance', 'error_category',
              'edit_operations', 'confidence']
    with path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows({k: x.get(k, '') for k in fields} for x in items)


def analyze(model_dir: Path, images_root: Path):
    predictions = model_dir / 'predictions.csv'
    items = read_predictions(predictions)
    wrong = [x for x in items if x['error_category'] != 'correct']
    correct = [x for x in items if x['error_category'] == 'correct']
    wrong_sorted = sorted(wrong, key=lambda x: (-int(x['edit_distance']),
                                                   x['image_path']))
    counts = Counter(x['error_category'] for x in wrong)
    types = Counter()
    for x in wrong:
        for op, _ in align_errors(x['ground_truth'], x['prediction_normalized']):
            types[op] += 1
    output = model_dir / 'error_analysis'
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / 'errors.csv', wrong_sorted)
    for i in range(0, len(wrong_sorted), 12):
        draw_sheet(wrong_sorted[i:i + 12],
                   output / f'contact_sheet_errors_{i // 12 + 1:02d}.jpg',
                   images_root, f'{model_dir.name} - ELPD - errores {i + 1}-{min(i + 12, len(wrong_sorted))}')
    draw_sheet(sorted(wrong, key=lambda r: (-r['confidence_num'], r['image_path']))[:12],
               output / 'contact_sheet_high_confidence_errors.jpg', images_root,
               f'{model_dir.name} - errores con mayor score de confianza')
    draw_sheet(sorted(correct, key=lambda r: (r['confidence_num'], r['image_path']))[:12],
               output / 'contact_sheet_low_confidence_correct.jpg', images_root,
               f'{model_dir.name} - aciertos con menor score de confianza')
    totals = sum(int(x['edit_distance']) for x in items)
    char_count = sum(len(x['ground_truth']) for x in items)
    md = [f'# Analisis de errores: {model_dir.name} / ELPD', '',
          '## Resumen', '',
          f'- Imagenes evaluadas: **{len(items)}**',
          f'- Exactas: **{len(correct)}**; incorrectas: **{len(wrong)}**',
          f'- EM: **{100 * len(correct) / len(items):.2f}%**',
          f'- CER: **{100 * totals / char_count:.2f}%** ({totals}/{char_count})',
          f'- Operaciones de edicion (una alineacion optima): ' +
            ', '.join(f'{k}={types[k]}' for k in ['insertion', 'substitution', 'deletion']),
          '', '## Categorias por imagen', '',
          '| Tipo | Numero |', '|---|---:|']
    for name in ('insertion', 'substitution', 'deletion', 'mixed'):
        md.append(f'| {name} | {counts[name]} |')
    md += ['', '## Imagenes con errores', '',
           '| Imagen | GT | Prediccion | Ediciones | Tipo | Confianza |',
           '|---|---|---|---:|---|---:|']
    for x in wrong_sorted:
        md.append(f'| `{Path(x["image_path"]).name}` | `{x["ground_truth"]}` | '
                  f'`{x["prediction_normalized"]}` | {x["edit_distance"]} | '
                  f'{x["error_category"]} | {x["confidence_num"]:.3f} |')
    md += ['', '## Interpretacion y limites', '',
           '- Las categorias son cambios de Levenshtein, **no** causas visuales demostradas.',
           '- Para secuencias con caracteres repetidos puede haber varias alineaciones igualmente optimas.',
           '- `confidence` procede del modelo; no esta calibrada como probabilidad de acierto.',
           '- Revisar las hojas de contacto para clasificar iluminacion, desenfoque, perspectiva, fondo y GT.',
           '- Las miniaturas son solo de inspeccion: las imagenes de origen no se modifican.',
           '- Este analisis es solo para **GT completo ELPD**. UC3M-LP requerira evaluacion parcial.', '']
    (output / 'error_analysis.md').write_text('\n'.join(md), encoding='utf-8')
    missing = [x['image_path'] for x in items if not (images_root / x['image_path']).is_file()]
    print(f'\n=== {model_dir.name} ===')
    print(f'Imagenes: {len(items)} | correctas: {len(correct)} | errores: {len(wrong)}')
    print('Categorias:', dict(counts))
    print('Operaciones:', dict(types))
    print('Imagenes de origen no encontradas:', len(missing))
    print('Resultados:', output)
    return len(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--image-root', type=Path, default=DATA / IMAGES_REL)
    parser.add_argument('--model', help='Nombre del subdirectorio: parseq_pretrained, svtrv2_tiny, etc.')
    args = parser.parse_args()
    base = args.project / BASELINES_REL
    if not base.is_dir():
        parser.error(f'No existe el directorio de resultados: {base}')
    if not args.image_root.is_dir():
        parser.error(f'No existe el directorio de imagenes: {args.image_root}')
    model_dirs = [base / args.model] if args.model else sorted(
        p for p in base.iterdir() if p.is_dir() and (p / 'predictions.csv').is_file())
    if not model_dirs:
        parser.error(f'No hay baselines con predictions.csv en: {base}')
    issues = 0
    print('=== GVIS: ANALISIS VISUAL DE ERRORES ELPD (SOLO CPU) ===')
    for model in model_dirs:
        if not (model / 'predictions.csv').is_file():
            parser.error(f'Falta predictions.csv en {model}')
        issues += analyze(model, args.image_root)
    print('\nESTADO:', 'REVISION DE IMAGENES PENDIENTE' if not issues else
          'ADVERTENCIA: faltan imagenes de origen en las hojas')
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
