#!/usr/bin/env python3
"""Genera evidencias visuales revisables de errores OCR en ELPD completo.

Usa CSV existentes de las baselines zero-shot. NO ejecuta modelos, NO modifica
imagenes originales ni manifiestos. Produce hojas de contacto con TODOS los
fallos respecto al GT disponible y ejemplos por patron para documentar en Word.

AVISO: muchas etiquetas ELPD son candidatas extraidas del nombre del archivo.
Las categorias de error son diagnosticos textuales automatizados; una asociacion
con escudos, bandas o calidad visual requiere revision humana de las imagenes.

Sin argumentos ejecuta automaticamente sobre los modelos disponibles en Legio2.
Uso opcional: --model parseq|svtrv2_tiny|svtrv2_small|mdiff4str_small|all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_DEFAULT = Path('/home/miguel.sanchez/ocr_matriculas')
DATA_DEFAULT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO')
MODELS = {
    'parseq': 'parseq_pretrained_full_735',
    'svtrv2_tiny': 'svtrv2_tiny_full_735',
    'svtrv2_small': 'svtrv2_small_full_735',
    'mdiff4str_small': 'mdiff4str_small_full_735',
}
DISPLAY = {
    'parseq': 'PARSeq',
    'svtrv2_tiny': 'SVTRv2-Tiny',
    'svtrv2_small': 'SVTRv2-Small',
    'mdiff4str_small': 'MDiff4STR-Small',
}
CATEGORIES = {
    'truncamiento_severo': 'Predicciones muy cortas / truncamientos graves',
    'omisiones': 'Eliminaciones u omisiones respecto al GT',
    'insercion_prefijo': 'Caracteres extra al inicio',
    'insercion_sufijo': 'Caracteres extra al final',
    'insercion_interna': 'Caracteres extra dentro de la secuencia',
    'duplicacion': 'Duplicaciones o repeticiones aparentes',
    'confusion_similar': 'Confusiones entre caracteres de forma similar',
    'sustitucion': 'Otras sustituciones',
    'mixtos': 'Predicciones con varios tipos de edicion',
    'confianza_alta': 'Discrepancias con confianza elevada',
    'casos_extremos': 'Discrepancias grandes para revisar visualmente',
    'posibles_elementos_graficos': 'Prefijos/sufijos para revisar escudos o bandas (NO confirmado)',
}
SIMILAR = [set(x) for x in ('I1', 'O0', 'B8', 'S5', 'Z2', 'G6')]
ALLOWED_SOURCES = {'strict_verified', 'strict_corrected', 'external_verified', 'filename_candidate'}


def fnt(size: int):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def levenshtein_ops(gt: str, pred: str) -> list[tuple[str, str, str]]:
    """Una alineacion minima entre GT y pred (no unica en general)."""
    n, m = len(gt), len(pred)
    mat = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        mat[i][0] = i
    for j in range(m + 1):
        mat[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            mat[i][j] = min(mat[i - 1][j] + 1, mat[i][j - 1] + 1,
                            mat[i - 1][j - 1] + (gt[i - 1] != pred[j - 1]))
    out = []
    i, j = n, m
    while i or j:
        if i and j and mat[i][j] == mat[i - 1][j - 1] + (gt[i - 1] != pred[j - 1]):
            if gt[i-1] != pred[j-1]:
                out.append(('S', gt[i-1], pred[j-1]))
            i -= 1
            j -= 1
        elif i and mat[i][j] == mat[i - 1][j] + 1:
            out.append(('D', gt[i-1], ''))
            i -= 1
        else:
            out.append(('I', '', pred[j-1]))
            j -= 1
    return out[::-1]


def near_visual(a: str, b: str) -> bool:
    return any(a in v and b in v and a != b for v in SIMILAR)


def detect_duplication(gt: str, pred: str, ops: list[tuple[str, str, str]]) -> bool:
    # Duplicaciones aparentes: requieren validacion visual y del GT.
    if not any(t == 'I' for t, _, _ in ops):
        return False
    for length in (4, 3, 2):
        if any(pred[i:i+length] == pred[i+length:i+2*length]
               for i in range(len(pred) - 2*length + 1)):
            return True
    def runs(s):
        counts = Counter()
        i = 0
        while i < len(s):
            j = i + 1
            while j < len(s) and s[j] == s[i]:
                j += 1
            counts[s[i]] = max(counts[s[i]], j-i)
            i = j
        return counts
    a, b = runs(gt), runs(pred)
    return any(v >= 2 and v > a[c] for c, v in b.items())


def categorize(r):
    gt, pred = r['ground_truth'], r['prediction_normalized']
    ops = levenshtein_ops(gt, pred)
    totals = Counter(t for t, _, _ in ops)
    cats = set()
    if len(pred) <= 3 and len(gt)-len(pred) >= 3:
        cats.add('truncamiento_severo')
    if totals['D']:
        cats.add('omisiones')
    if totals['I']:
        if pred.endswith(gt) and len(pred) > len(gt):
            cats.add('insercion_prefijo')
        elif pred.startswith(gt) and len(pred) > len(gt):
            cats.add('insercion_sufijo')
        elif len(pred) > len(gt) and gt in pred:
            cats.add('insercion_interna')
        else:
            cats.add('insercion_interna')
        if detect_duplication(gt, pred, ops):
            cats.add('duplicacion')
        if pred != gt and len(pred)>len(gt):
            cats.add('posibles_elementos_graficos')
    if totals['S']:
        if any(near_visual(a, b) for op, a, b in ops if op == 'S'):
            cats.add('confusion_similar')
        if any(not near_visual(a, b) for op, a, b in ops if op == 'S'):
            cats.add('sustitucion')
    if len([k for k in ('S','I','D') if totals[k]]) >= 2:
        cats.add('mixtos')
    if r['confidence_num'] is not None and r['confidence_num'] >= 0.85:
        cats.add('confianza_alta')
    if len(ops) >= 3:
        cats.add('casos_extremos')
    return cats, totals, ops


def load_csv(path, image_root):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        dr = csv.DictReader(f)
        required = {'image_path', 'ground_truth', 'prediction_normalized', 'exact_match', 'edit_distance'}
        if not required.issubset(dr.fieldnames or []):
            raise ValueError(f'{path}: faltan columnas {sorted(required-set(dr.fieldnames or []))}')
        rows = list(dr)
    if len(rows) != 735:
        raise ValueError(f'{path}: se esperaban 735 filas y hay {len(rows)}')
    seen = set()
    for row in rows:
        rel = row['image_path'].strip()
        part = Path(rel)
        if part.is_absolute() or '..' in part.parts or not rel.startswith('images/'):
            raise ValueError(f'Ruta insegura: {rel}')
        if rel in seen:
            raise ValueError(f'Imagen repetida en CSV: {rel}')
        seen.add(rel)
        if not (image_root / rel).is_file():
            raise FileNotFoundError(image_root / rel)
        row['image_path'] = rel
        row['ground_truth'] = row['ground_truth'].strip()
        row['prediction_normalized'] = row['prediction_normalized'].strip()
        if not row['ground_truth'] or not row['ground_truth'].isascii() or not row['ground_truth'].isalnum():
            raise ValueError(f'GT mal formado: {rel}')
        gt, pred = row['ground_truth'], row['prediction_normalized']
        if any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in gt+pred):
            raise ValueError(f'GT o prediccion no normalizada: {rel}')
        if str(int(pred == gt)) != row['exact_match']:
            raise ValueError(f'Exact Match inconsistente en {rel}')
        if len(levenshtein_ops(gt, pred)) != int(row['edit_distance']):
            raise ValueError(f'Distancia de edicion inconsistente en {rel}')
        try:
            row['confidence_num'] = float(row.get('confidence', ''))
        except (ValueError, TypeError):
            row['confidence_num'] = None
        row.setdefault('gt_source', 'not_specified')
        row.setdefault('scenario', 'not_specified')
    return rows


def sheet_tile(r, image_root: Path, model: str, width=520, height=224):
    tile = Image.new('RGB', (width, height), '#ffffff')
    d = ImageDraw.Draw(tile)
    small, mid = fnt(17), fnt(21)
    d.rectangle((0, 0, width-1, height-1), outline='#bbbbbb', width=2)
    image_height = 112 if height <= 224 else height - 112
    image_path = image_root / r['image_path']
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        vis = ImageOps.contain(im, (width-24, image_height), method=Image.Resampling.LANCZOS)
        x = (width-vis.width)//2
        tile.paste(vis, (x, 8+(image_height-vis.height)//2))
    y = 15 + image_height
    d.text((12, y), f'GT: {r["ground_truth"]}  |  PRED: {r["prediction_normalized"] or "(VACIA)"}',
           font=mid, fill='#151515')
    y += 30
    d.text((12, y), f'{r["image_path"][-46:]}', font=small, fill='#404040')
    y += 26
    conf = r.get('confidence_num')
    confidence = f'{conf:.3f}' if conf is not None else 'ND'
    d.text((12, y), f'Ediciones={r["edit_distance"]} | Conf={confidence} | GT={r["gt_source"]}',
           font=small, fill='#555555')
    return tile


def save_sheet(rows, file, image_root, title, cols=3, per_page=12):
    if not rows:
        return
    tw, th = 520, 224
    nrows = math.ceil(len(rows)/cols)
    canvas = Image.new('RGB', (cols*tw, 60+nrows*th), '#f2f2f2')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), title[:90], font=fnt(26), fill='#111111')
    for i, r in enumerate(rows):
        canvas.paste(sheet_tile(r, image_root, '', tw, th), ((i%cols)*tw, 60+(i//cols)*th))
    file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file, quality=90, optimize=True)


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def rank(r, category):
    # Se prioriza una referencia auditada y un error facil de comparar.
    gt_source = r.get('gt_source', '')
    audit = 0 if gt_source in ('strict_verified','strict_corrected','external_verified') else 1
    length = int(r['edit_distance'])
    conf = r['confidence_num'] if r['confidence_num'] is not None else 0
    if category in ('truncamiento_severo', 'casos_extremos'):
        return (audit, -length, -conf, r['image_path'])
    if category == 'confianza_alta':
        return (audit, -conf, length, r['image_path'])
    if category in ('confusion_similar','sustitucion'):
        return (audit, length, -conf, r['image_path'])
    return (audit, length, -conf, r['image_path'])


def export_model(model, rows, image_root, out_root, examples_per_category, page_size):
    model_out = out_root / model
    all_errs = [r for r in rows if r['exact_match'] == '0']
    cats_by_image = {}
    all_records = []
    grouped = defaultdict(list)
    for r in all_errs:
        cats, operations, _ = categorize(r)
        cats_by_image[r['image_path']] = cats
        for cat in cats:
            grouped[cat].append(r)
        all_records.append({
            'image_path': r['image_path'], 'ground_truth':r['ground_truth'],
            'prediction_normalized':r['prediction_normalized'],
            'gt_source':r.get('gt_source',''), 'scenario':r.get('scenario',''),
            'edit_distance':r['edit_distance'], 'confidence':r.get('confidence',''),
            'n_insertions_alignment':operations['I'],
            'n_deletions_alignment':operations['D'],
            'n_substitutions_alignment':operations['S'],
            'categories_candidate':';'.join(sorted(cats)),
            'requires_visual_review':'yes',
        })
    write_rows(model_out/'all_errors.csv', all_records,
               ['image_path','ground_truth','prediction_normalized','gt_source','scenario',
                'edit_distance','confidence','n_insertions_alignment',
                'n_deletions_alignment','n_substitutions_alignment',
                'categories_candidate','requires_visual_review'])

    for offset in range(0, len(all_errs), page_size):
        selected = all_errs[offset:offset+page_size]
        save_sheet(selected,
                   model_out/'all_errors_contact_sheets'/f'errors_{1+offset//page_size:03d}.jpg',
                   image_root, f'{DISPLAY[model]} | Discrepancias vs GT | {offset+1}-{offset+len(selected)}')

    selected_rows = []
    selected_images = set()
    for cat, title in CATEGORIES.items():
        choices = sorted(grouped.get(cat,[]), key=lambda r: rank(r, cat))[:examples_per_category]
        if not choices:
            continue
        save_sheet(choices, model_out/'category_sheets'/f'{cat}.jpg', image_root,
                   f'{DISPLAY[model]} | {title}')
        for r in choices:
            rel = r['image_path']
            ident = hashlib.sha1(rel.encode()).hexdigest()[:10]
            stem = Path(rel).stem[:50]
            basename = f'{stem}_{ident}'
            output_label = model_out/'labeled_examples'/f'{basename}.png'
            output_original = model_out/'original_examples'/f'{basename}{Path(rel).suffix.lower()}'
            if rel not in selected_images:
                selected_images.add(rel)
                one = sheet_tile(r, image_root, model, 1200, 330)
                output_label.parent.mkdir(parents=True, exist_ok=True)
                output_original.parent.mkdir(parents=True, exist_ok=True)
                # Preservar una COPIA binaria del recorte original, sin anotaciones.
                shutil.copy2(image_root/rel, output_original)
                one.save(output_label)
            selected_rows.append({
                'category':cat, 'category_description':title,
                'image_path':rel, 'ground_truth':r['ground_truth'],
                'prediction_normalized':r['prediction_normalized'],
                'gt_source':r.get('gt_source',''), 'scenario':r.get('scenario',''),
                'edit_distance':r['edit_distance'], 'confidence':r.get('confidence',''),
                'original_export_path':str(output_original.relative_to(out_root)),
                'labeled_example_path':str(output_label.relative_to(out_root)),
                'image_checked':'', 'gt_checked':'', 'visual_observation':'',
            })
    write_rows(model_out/'selected_examples_for_review.csv', selected_rows,
               ['category','category_description','image_path','ground_truth',
                'prediction_normalized','gt_source','scenario','edit_distance','confidence',
                'original_export_path','labeled_example_path','image_checked','gt_checked',
                'visual_observation'])
    summary = {
        'model': DISPLAY[model],
        'images': len(rows),
        'correct': len(rows)-len(all_errs),
        'discrepancies_vs_available_gt':len(all_errs),
        'EM_percent':round(100*(len(rows)-len(all_errs))/len(rows),4),
        'levenshtein_total':sum(int(r['edit_distance']) for r in rows),
        'gt_characters':sum(len(r['ground_truth']) for r in rows),
        'CER_percent':round(100*sum(int(r['edit_distance']) for r in rows)/sum(len(r['ground_truth']) for r in rows),4),
        'gt_sources': dict(Counter(r.get('gt_source','unknown') for r in rows)),
        'categories_not_mutually_exclusive': {cat:len(grouped[cat]) for cat in CATEGORIES},
        'individual_exported_cases':len(selected_images),
        'selected_example_rows':len(selected_rows),
        'note': 'Conteos automáticos descriptivos; etiquetas filename_candidate no verificadas visualmente. '
                'No inferir causas visuales solo por alineamiento textual.',
    }
    (model_out/'analysis_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
    # Paquete autocontenido para compartir las hojas y ejemplos en el chat.
    bundle = out_root / f'{model}_visual_review.zip'
    with ZipFile(bundle, 'w', compression=ZIP_DEFLATED, compresslevel=5) as zf:
        for file in sorted(model_out.rglob('*')):
            if file.is_file():
                zf.write(file, arcname=str(file.relative_to(out_root)))
    summary['review_bundle'] = str(bundle)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['all',*MODELS],default='all')
    ap.add_argument('--project-root',type=Path,default=PROJECT_DEFAULT)
    ap.add_argument('--image-root',type=Path,default=DATA_DEFAULT)
    ap.add_argument('--output-root',type=Path,default=None)
    ap.add_argument('--examples-per-category',type=int,default=8)
    ap.add_argument('--page-size',type=int,default=12)
    args = ap.parse_args()
    if args.examples_per_category < 1 or args.page_size < 1:
        ap.error('examples-per-category y page-size deben ser positivos')
    out_root = args.output_root or args.project_root/'results/inspection/ELPD/full_baselines_visual'
    print('=== GVIS: EJEMPLOS VISUALES DE ERRORES / ELPD COMPLETO ===',flush=True)
    print('Resultados:',out_root,flush=True)
    chosen = [args.model] if args.model != 'all' else list(MODELS)
    used = {}
    for model in chosen:
        infile = args.project_root/'results/baselines/ELPD'/MODELS[model]/'predictions.csv'
        if not infile.is_file() and args.model == 'all':
            print(f'[PENDIENTE] {DISPLAY[model]}: {infile}',flush=True)
            continue
        rows = load_csv(infile, args.image_root)
        summary = export_model(model,rows,args.image_root,out_root,
                               args.examples_per_category,args.page_size)
        used[model] = rows
        print(f'[OK] {DISPLAY[model]}: {summary["images"]} imagenes, '
              f'{summary["discrepancies_vs_available_gt"]} discrepancias, '
              f'{summary["individual_exported_cases"]} originales seleccionados; '
              f'EM={summary["EM_percent"]:.2f}%, CER={summary["CER_percent"]:.2f}%',flush=True)
    if not used:
        raise FileNotFoundError('No se ha encontrado ningun predictions.csv completo de ELPD')

    # Matriculas con discrepancias simultaneas, comparando exactamente la misma imagen.
    if len(used)>=2:
        per_model = {m:{r['image_path']:r for r in rows} for m,rows in used.items()}
        common=set.intersection(*(set(d) for d in per_model.values()))
        shared=[]
        for rel in sorted(common):
            refs={per_model[m][rel]['ground_truth'] for m in used}
            if len(refs)!=1:
                raise ValueError(f'GT diferentes entre modelos en {rel}: {refs}')
            incorrect=[m for m in used if per_model[m][rel]['exact_match']=='0']
            if len(incorrect)==len(used):
                item={'image_path':rel,'ground_truth':next(iter(refs)),'num_models':len(incorrect)}
                item.update({f'{m}_prediction':per_model[m][rel]['prediction_normalized'] if m in used else '' for m in MODELS})
                shared.append(item)
        fields=['image_path','ground_truth','num_models']+[f'{m}_prediction' for m in MODELS]
        write_rows(out_root/'shared_failures_available_models.csv',shared,fields)
        if shared:
            one_model=next(iter(used))
            row_lookup={r['image_path']:r for r in used[one_model]}
            for start in range(0,min(len(shared),48),12):
                save_sheet([row_lookup[e['image_path']] for e in shared[start:start+12]],
                           out_root/'shared_failures_contact_sheets'/f'shared_{1+start//12:03d}.jpg',
                           args.image_root,
                           f'Discrepancias compartidas de {len(used)} modelos (GT candidato o auditado)')
        print(f'Discrepancias compartidas: {len(shared)} (entre {len(used)} modelos disponibles)',flush=True)
    print('RESULTADO: MATERIAL VISUAL GENERADO; PENDIENTE REVISION HUMANA',flush=True)
    print('DESCARGAR CON TASK: GVIS: Descargar muestras ELPD',flush=True)


if __name__=='__main__':
    main()
