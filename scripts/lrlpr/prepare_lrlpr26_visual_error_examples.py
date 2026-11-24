#!/usr/bin/env python3
"""LRLPR-26 HR: exportar una pequena muestra visual de errores de 4 baselines.

No ejecuta inferencia ni modifica el dataset. Elige de forma reproducible
predicciones discrepantes, copia solo los recortes seleccionados y crea hojas
comparativas con GT + 4 modelos. Es una seleccion ilustrativa, no estadistica.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import shutil
import zipfile

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_MANIFEST = DATASETS / 'processed/LRLPR-26_LPO/val_manifest.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'raw/LRLPR-26/selected_hr'
DEFAULT_RESULTS = PROJECT / 'results/baselines/LRLPR-26'
DEFAULT_OUT = PROJECT / 'results/inspection/LRLPR-26/error_examples_val_v1'
MODEL_DIRS = {
    'PARSeq': 'parseq_pretrained_val10k',
    'SVTRv2-Tiny': 'svtrv2_tiny_val10k',
    'SVTRv2-Small': 'svtrv2_small_val10k',
    'MDiff4STR-Small': 'mdiff4str_small_val10k',
}
GROUPS = [(s, l) for s in ('Scenario-A', 'Scenario-B')
          for l in ('Brazilian', 'Mercosur')]
CATEGORIES = [
    ('sustitucion_simple', 'Sustitucion simple'),
    ('confusion_I1_O0_B8', 'Confusion I/1, O/0 o B/8'),
    ('insercion_adicional', 'Insercion de caracteres'),
    ('duplicacion_consecutiva', 'Duplicacion consecutiva'),
    ('truncamiento', 'Truncamiento'),
    ('prediccion_vacia', 'Prediccion vacia'),
    ('confianza_alta_incorrecta', 'Error con confianza >= 0.90'),
]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Falta archivo: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f'Faltan columnas {sorted(missing)} en {path}')
        return list(r)


def valid_source_path(source: str) -> Path:
    p = Path(source)
    if p.is_absolute() or '..' in p.parts or len(p.parts) != 5:
        raise ValueError(f'Ruta relativa invalida: {source}')
    if (p.parts[0], p.parts[1], p.parts[2]) not in [
        ('train', s, l) for s, l in GROUPS
    ] or not p.parts[3].startswith('track_') or not p.parts[4].startswith('hr-'):
        raise ValueError(f'Ruta HR inesperada: {source}')
    return p


def confusion_similar(gt: str, pred: str) -> bool:
    if len(gt) != len(pred):
        return False
    differences = [(a, b) for a, b in zip(gt, pred) if a != b]
    return len(differences) == 1 and frozenset(differences[0]) in (
        frozenset('I1'), frozenset('O0'), frozenset('B8'))


def duplicate_one(gt: str, pred: str) -> bool:
    return len(pred) == len(gt) + 1 and any(
        pred[i] == pred[i + 1] and pred[:i] + pred[i+1:] == gt
        for i in range(len(pred) - 1)
    )


def is_category(category: str, row: dict[str, str]) -> bool:
    gt, pred = row['ground_truth'], row['prediction_normalized']
    if gt == pred:
        return False
    ed = int(row['edit_distance'])
    if category == 'sustitucion_simple':
        return len(gt) == len(pred) and ed == 1 and not confusion_similar(gt, pred)
    if category == 'confusion_I1_O0_B8':
        return len(gt) == len(pred) and ed == 1 and confusion_similar(gt, pred)
    if category == 'insercion_adicional':
        return len(pred) > len(gt) and ed <= 3 and not duplicate_one(gt, pred)
    if category == 'duplicacion_consecutiva':
        return duplicate_one(gt, pred)
    if category == 'truncamiento':
        return 0 < len(pred) < len(gt)
    if category == 'prediccion_vacia':
        return len(pred) == 0
    if category == 'confianza_alta_incorrecta':
        return float(row['confidence'] or 0) >= .90 and ed <= 2
    return False


def stable_key(s: str) -> str:
    return hashlib.sha256(('lrlpr26-v1-seed42:' + s).encode()).hexdigest()


def select_examples(preds: dict, manifest: dict):
    used_sources, used_tracks = set(), set()
    selections = []
    for model in MODEL_DIRS:
        own_groups = Counter()
        for category, label in CATEGORIES:
            cands = [r for r in preds[model].values() if is_category(category, r)
                     and r['source_path'] not in used_sources]
            if not cands:
                continue
            def rank(r):
                group = (r['scenario'], r['plate_layout'])
                track = r['track_id']
                return (
                    track in used_tracks,
                    own_groups[group],
                    # Confianza alta: escoger casos especialmente fiables para el modelo.
                    -float(r['confidence'] or 0) if category == 'confianza_alta_incorrecta' else 0,
                    # Evitar errores enormes salvo truncamientos, donde son ilustrativos.
                    int(r['edit_distance']) if category not in ('truncamiento','prediccion_vacia') else 0,
                    stable_key(r['source_path']),
                )
            r = min(cands, key=rank)
            source = r['source_path']
            selections.append({'selected_for': model, 'category': category,
                               'category_label': label, 'source_path': source})
            used_sources.add(source)
            used_tracks.add(r['track_id'])
            own_groups[(r['scenario'], r['plate_layout'])] += 1
    # Una comparativa en cada escenario/estilo de matricula: fallan los cuatro.
    for scenario, layout in GROUPS:
        cands = [r for r in preds['PARSeq'].values()
                 if r['scenario'] == scenario and r['plate_layout'] == layout
                 and r['source_path'] not in used_sources
                 and all(preds[m][r['source_path']]['exact_match'] == '0' for m in MODEL_DIRS)]
        if not cands:
            continue
        r = min(cands, key=lambda x: (
            x['track_id'] in used_tracks,
            sum(int(preds[m][x['source_path']]['edit_distance']) for m in MODEL_DIRS),
            stable_key(x['source_path'])))
        src = r['source_path']
        selections.append({'selected_for': 'COMPARACION',
                           'category': 'errores_compartidos',
                           'category_label': 'Fallo en los 4 modelos',
                           'source_path': src})
        used_sources.add(src)
        used_tracks.add(r['track_id'])
    # Unir GT + predicciones para todas las vistas, por el mismo source_path.
    rows = []
    for i, s in enumerate(selections, start=1):
        src = s['source_path']
        md = manifest[src]
        row = {'numero': f'{i:02d}', **s,
               'scenario': md['scenario'], 'plate_layout': md['plate_layout'],
               'track_id': md['track_id'], 'view_id': md['view_id'],
               'ground_truth': md['text_normalized']}
        for model in MODEL_DIRS:
            p = preds[model][src]
            row[model] = p['prediction_normalized']
            row[f'confianza_{model}'] = p['confidence']
            row[f'acierto_{model}'] = p['exact_match']
        rows.append(row)
    return rows


def load_inputs(manifest_path: Path, results: Path):
    mrows = read_csv(manifest_path, {'source_path', 'track_id', 'scenario', 'plate_layout',
                                    'text_normalized', 'view_id', 'image_path'})
    if len(mrows) != 10000:
        raise ValueError(f'Manifest de validacion inesperado: {len(mrows)} != 10000')
    manifest = {}
    for r in mrows:
        src = r['source_path']
        valid_source_path(src)
        if src in manifest:
            raise ValueError(f'Imagen duplicada en manifest: {src}')
        if not r['text_normalized'] or len(r['text_normalized']) != 7:
            raise ValueError(f'GT inesperado: {src}')
        manifest[src] = r
    preds = {}
    for model, folder in MODEL_DIRS.items():
        csv_path = results / folder / 'predictions.csv'
        rows = read_csv(csv_path, {'source_path', 'ground_truth', 'prediction_normalized',
                                   'exact_match', 'edit_distance', 'confidence',
                                   'scenario', 'plate_layout', 'track_id'})
        if len(rows) != len(manifest):
            raise ValueError(f'{model}: filas {len(rows)} != {len(manifest)}')
        index = {}
        for r in rows:
            src = r['source_path']
            if src in index or src not in manifest:
                raise ValueError(f'{model}: ruta repetida o ausente {src}')
            if r['ground_truth'] != manifest[src]['text_normalized']:
                raise ValueError(f'{model}: GT diferente en {src}')
            if int(r['exact_match']) != int(r['ground_truth'] == r['prediction_normalized']):
                raise ValueError(f'{model}: exact_match inconsistente {src}')
            index[src] = r
        preds[model] = index
        print(f'[OK] {model}: {len(index)} predicciones, '
              f'{sum(x["exact_match"] == "0" for x in index.values())} fallos', flush=True)
    return manifest, preds


def fonts():
    from PIL import ImageFont
    for dirname in ('/usr/share/fonts/truetype/dejavu/', '/usr/share/fonts/TTF/'):
        try:
            f = Path(dirname) / 'DejaVuSans.ttf'
            if f.is_file():
                return (ImageFont.truetype(str(f), 25),
                        ImageFont.truetype(str(f), 19),
                        ImageFont.truetype(str(f), 17))
        except OSError:
            pass
    f = ImageFont.load_default()
    return f, f, f


def generate_sheet(sheet_path: Path, subset: list[dict], image_root: Path):
    from PIL import Image, ImageDraw, ImageOps
    big, regular, small = fonts()
    width, header, height_per = 1760, 115, 248
    canvas = Image.new('RGB', (width, header + len(subset) * height_per), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, header), fill='#172c4b')
    draw.text((26, 20), f'LRLPR-26 HR | Ejemplos de errores | {sheet_path.stem}',
              fill='white', font=big)
    draw.text((26, 64), 'GT: anotacion original NO auditada | Rojo: discrepancia OCR | Verde: coincidencia textual',
              fill='#e1eaf6', font=small)
    for offset, row in enumerate(subset):
        y = header + offset * height_per
        draw.rectangle((0, y, width, y + height_per),
                       fill='#f4f6fa' if offset % 2 == 0 else '#ffffff')
        draw.line((0, y, width, y), fill='#d8dde7', width=2)
        draw.text((26, y + 13), f"{row['numero']} · {row['category_label']}",
                  fill='#1c3457', font=regular)
        draw.text((26, y + 47), f"{row['scenario']} / {row['plate_layout']} / {row['track_id'].split('/')[-1]} | {row['view_id']}",
                  fill='#4b5563', font=small)
        source = image_root / valid_source_path(row['source_path'])
        with Image.open(source) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
            actual_res = im.size
            im.thumbnail((690, 160), Image.Resampling.NEAREST)
            # Ampliamos el recorte sin cambiar el archivo original.
            factor = min(690 / max(1, im.width), 160 / max(1, im.height))
            im = im.resize((max(1, round(im.width * factor)),
                            max(1, round(im.height * factor))), Image.Resampling.NEAREST)
            canvas.paste(im, (26 + (700 - im.width)//2, y + 76 + (160 - im.height)//2))
        draw.text((742, y + 20), 'GT: ' + row['ground_truth'], fill='#195492', font=big)
        for j, model in enumerate(MODEL_DIRS):
            pred = row[model] if row[model] else '[VACIA]'
            fill = '#16803f' if row[f'acierto_{model}'] == '1' else '#c12626'
            draw.text((742, y + 59 + j * 40), f'{model}: {pred}', fill=fill, font=regular)
        draw.text((1540, y+20), f'{actual_res[0]} x {actual_res[1]} px',
                  fill='#556070', font=small)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(sheet_path, quality=91, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    p.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    p.add_argument('--baselines-root', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--plan-only', action='store_true', help='Solo calcular muestra y CSV, sin imagenes')
    args = p.parse_args()
    print('=== LRLPR-26: EJEMPLOS VISUALES DE ERRORES, 4 MODELOS ===', flush=True)
    manifest, preds = load_inputs(args.manifest, args.baselines_root)
    rows = select_examples(preds, manifest)
    if not rows:
        raise RuntimeError('No se seleccionaron errores')
    args.out.mkdir(parents=True, exist_ok=True)
    tab = args.out / 'seleccion_ejemplos.csv'
    with tab.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[OK] Casos elegidos: {len(rows)} (distintos, semilla logica 42)', flush=True)
    print('[OK] Seleccion:', dict(Counter(r['selected_for'] for r in rows)), flush=True)
    if args.plan_only:
        print('[OK] Plan generado; no se han copiado imagenes', flush=True)
        return
    image_dir = args.out / 'imagenes_originales'
    for row in rows:
        rel = valid_source_path(row['source_path'])
        src = args.image_root / rel
        if not src.is_file():
            raise FileNotFoundError(f'No se encuentra recorte HR: {src}')
        dst = image_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f'[OK] Recortes originales copiados: {len(rows)}', flush=True)
    sheets = []
    for key in list(MODEL_DIRS) + ['COMPARACION']:
        subset = [r for r in rows if r['selected_for'] == key]
        if not subset:
            continue
        tag = 'comparacion' if key == 'COMPARACION' else key.lower().replace('-', '_')
        path = args.out / 'hojas' / f'{tag}.jpg'
        generate_sheet(path, subset, args.image_root)
        sheets.append(path)
    readme = args.out / 'LEEME.txt'
    readme.write_text(
        'LRLPR-26 | Muestra visual ilustrativa (NO evaluacion de frecuencias)\n'
        'Se seleccionan casos con discrepancia entre GT y prediccion normalizada\n'
        'y ejemplos compartidos por los 4 modelos. Todos proceden de val_manifest.csv.\n'
        'Las etiquetas son annotation_unverified: cualquier discrepancia puede\n'
        'ser un error de OCR o de GT. Revisar visualmente antes de clasificar causas.\n'
        'No se modifica ningun original; no se ejecuta inferencia.\n'
        'En las hojas: rojo = discrepancia frente a GT, verde = coincide con GT.\n'
        'La confianza no esta calibrada entre modelos.\n'
        'Las categorias pueden solaparse; la muestra no es aleatoria ni\n'
        'representativa de la incidencia de errores.\n'
        'No redistribuir imagenes sin verificar los terminos de uso del dataset.\n',
        encoding='utf-8')
    archive = args.out / 'LRLPR26_Ejemplos_Errores_4_Modelos.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in [tab, readme, *sheets, *sorted(image_dir.rglob('*'))]:
            if file.is_file():
                z.write(file, file.relative_to(args.out))
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f'ZIP corrupto: {bad}')
    print(f'[OK] Hojas de contacto: {len(sheets)}', flush=True)
    print(f'[OK] ZIP: {archive} ({archive.stat().st_size / (1024*1024):.2f} MiB)', flush=True)
    print('RESULTADO: MUESTRA VISUAL PREPARADA; REVISION HUMANA PENDIENTE', flush=True)


if __name__ == '__main__':
    main()
