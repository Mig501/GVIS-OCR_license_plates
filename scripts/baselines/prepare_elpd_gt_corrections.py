#!/usr/bin/env python3
"""Prepara auditoría visual de etiquetas GT potencialmente erróneas en ELPD (735).

Lee las cuatro predicciones zero-shot ya generadas y las imágenes originales.
NO modifica GT, manifiestos, imágenes ni resultados previos. La coincidencia
entre modelos solo PRIORIZA la revisión; NO se considera una verdad de referencia.

Salida (por defecto): results/audit/ELPD/gt_correction_review/
  review_queue.csv             Cola COMPLETA de 653 candidatos + 2 externos
  contact_sheets/             Imágenes originales para revisar TODA la cola
  priority_review_pack.zip    CSV, hojas e imágenes de los casos prioritarios
  review_summary.json         Estadísticas y procedencia
  README_revision.txt         Instrucciones de revisión humana

Requiere Pillow (ya utilizado por el proyecto para las hojas de contacto).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
IMAGES = Path('/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO')
OUT_REL = Path('results/audit/ELPD/gt_correction_review')
EXPECTED_COUNT = 735
EXPECTED_MANIFEST_SHA256 = '5854abd860c89a17e35ef225627b17725c60d23088197f280e34eb5272097f13'
MODEL_DIRS = {
    'parseq': 'parseq_pretrained_full_735',
    'svtrv2_tiny': 'svtrv2_tiny_full_735',
    'svtrv2_small': 'svtrv2_small_full_735',
    'mdiff4str_small': 'mdiff4str_small_full_735',
}
MODEL_NAMES = {'parseq': 'PARSeq', 'svtrv2_tiny': 'SVTRv2-Tiny',
               'svtrv2_small': 'SVTRv2-Small', 'mdiff4str_small': 'MDiff4STR-Small'}
ALLOWED_SOURCES = {'strict_verified', 'strict_corrected', 'external_verified', 'filename_candidate'}
REVIEW_SOURCES = {'external_verified', 'filename_candidate'}
FONT_LOCATIONS = (
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
)


def font(size: int):
    for f in FONT_LOCATIONS:
        if Path(f).is_file():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_path', 'ground_truth', 'gt_source', 'prediction_normalized', 'confidence', 'scenario'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'Columnas ausentes en {path}: {sorted(required - set(reader.fieldnames or []))}')
        return list(reader)


def hash_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()


def validate_and_merge(project: Path, images: Path):
    found = {}
    for name, folder in MODEL_DIRS.items():
        root = project / 'results/baselines/ELPD' / folder
        summary_path = root / 'summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('dataset_size') != EXPECTED_COUNT:
            raise ValueError(f'{name}: dataset_size != 735')
        if summary.get('manifest_sha256') != EXPECTED_MANIFEST_SHA256:
            raise ValueError(f'{name}: no coincide el manifiesto base')
        raw = read_csv(root / 'predictions.csv')
        if len(raw) != EXPECTED_COUNT:
            raise ValueError(f'{name}: {len(raw)} filas (esperadas 735)')
        indexed = {r['image_path']: r for r in raw}
        if len(indexed) != EXPECTED_COUNT:
            raise ValueError(f'{name}: rutas duplicadas en predictions.csv')
        found[name] = indexed

    paths = set(found['parseq'])
    for name, indexed in found.items():
        if set(indexed) != paths:
            raise ValueError(f'{name}: no contiene exactamente las mismas imágenes que PARSeq')
    sources = Counter()
    joined = []
    for rel in sorted(paths):
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'images':
            raise ValueError(f'Ruta insegura: {rel}')
        if not (images / rel).is_file():
            raise FileNotFoundError(f'No existe imagen original: {images / rel}')
        refs = [found[n][rel] for n in MODEL_DIRS]
        gts = {r['ground_truth'] for r in refs}
        srs = {r['gt_source'] for r in refs}
        splits = {r['scenario'] for r in refs}
        if len(gts) != 1 or len(srs) != 1 or len(splits) != 1:
            raise ValueError(f'Las referencias o splits difieren entre modelos: {rel}')
        gt, src, split = next(iter(gts)), next(iter(srs)), next(iter(splits))
        if src not in ALLOWED_SOURCES:
            raise ValueError(f'GT source inesperada: {rel} / {src}')
        sources[src] += 1
        if src not in REVIEW_SOURCES:
            continue  # Proteger las 80 etiquetas estrictamente auditadas.
        preds = {n: found[n][rel]['prediction_normalized'] for n in MODEL_DIRS}
        votes = Counter(p for p in preds.values() if p and p != gt)
        preferred = sorted(votes.items(), key=lambda it: (-it[1], it[0]))
        if preferred:
            max_votes = preferred[0][1]
            tied = [x for x, count in preferred if count == max_votes]
            suggestion = preds['parseq'] if preds['parseq'] in tied else tied[0]
        else:
            max_votes = 0
            suggestion = ''
        if max_votes == 4:
            prio, desc = 1, '4 modelos coinciden en una alternativa diferente del GT'
        elif max_votes == 3:
            prio, desc = 2, '3 modelos coinciden en una alternativa diferente del GT'
        elif max_votes == 2 and preds['parseq'] == suggestion:
            prio, desc = 3, '2 modelos, incluido PARSeq, coinciden en alternativa'
        elif max_votes == 2:
            prio, desc = 4, '2 modelos coinciden en una alternativa'
        elif max_votes == 1:
            prio, desc = 5, 'Desacuerdo sin consenso; requiere inspección visual'
        else:
            prio, desc = 6, 'Sin desacuerdo; revisar de todos modos para completar auditoría'
        try:
            confidences = {n: float(found[n][rel]['confidence']) for n in MODEL_DIRS}
        except ValueError as ex:
            raise ValueError(f'Confianza no numérica: {rel}') from ex
        joined.append({
            'priority_num': prio, 'priority_reason': desc,
            'image_path': rel, 'scenario': split, 'gt_source': src,
            'gt_before_review': gt,
            'suggested_gt_NOT_VERIFIED': suggestion,
            'agreeing_models': max_votes,
            'models_matching_current_gt': sum(p == gt for p in preds.values()),
            **{f'pred_{n}': preds[n] for n in MODEL_DIRS},
            **{f'confidence_{n}': f'{confidences[n]:.6f}' for n in MODEL_DIRS},
            'review_decision': 'pending',       # pending | confirm_gt | correct | unreadable
            'image_checked': '',               # yes solo tras verificar imagen original
            'corrected_gt': '',                # completar SOLO si review_decision=correct
            'reviewer': '',
            'review_notes': '',
        })
    joined.sort(key=lambda r: (r['priority_num'], -r['agreeing_models'],
                               -float(r['confidence_parseq']), r['image_path']))
    return joined, sources


def resize_plate(image: Image.Image, width: int, height: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert('RGB')
    if image.width <= 0 or image.height <= 0:
        raise ValueError('Dimensiones inválidas')
    ratio = min(width / image.width, height / image.height)
    newsize = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return image.resize(newsize, Image.Resampling.LANCZOS)


def render_sheet(rows: list[dict], image_root: Path, file: Path):
    # 2 columnas × 4 filas, fuente grande, matrícula original conservada en color.
    cw, ch, cols, rows_per_page = 800, 290, 2, 4
    canvas = Image.new('RGB', (cw * cols, ch * rows_per_page), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    strong, normal, small = font(24), font(19), font(16)
    for i, r in enumerate(rows):
        x, y = (i % cols) * cw, (i // cols) * ch
        draw.rectangle((x+8, y+8, x+cw-8, y+ch-8), outline='#bdc7d3', width=2)
        draw.text((x+20, y+14), f"#{r['review_order']}  P{r['priority_num']}  {r['image_path'].split('/')[-1]}", fill='#1d2939', font=strong)
        draw.text((x+20, y+47), f"GT actual: {r['gt_before_review']}", fill='#9a3412', font=strong)
        draw.text((x+20, y+79), f"Alternativa NO validada: {r['suggested_gt_NOT_VERIFIED'] or '—'} ({r['agreeing_models']}/4)", fill='#1e40af', font=normal)
        with Image.open(image_root / r['image_path']) as original:
            p = resize_plate(original, 745, 124)
        canvas.paste(p, (x+20+(745-p.width)//2, y+112+(124-p.height)//2))
        draw.text((x+20, y+244), f"PARSeq:{r['pred_parseq'][:15]}  Tiny:{r['pred_svtrv2_tiny'][:15]}", fill='#344054', font=small)
        draw.text((x+20, y+266), f"Small:{r['pred_svtrv2_small'][:15]}  MDiff:{r['pred_mdiff4str_small'][:15]}", fill='#344054', font=small)
    file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file, format='JPEG', quality=88, optimize=True)


def csv_bytes(rows: list[dict], cols: list[str]) -> bytes:
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode('utf-8-sig')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=PROJECT)
    p.add_argument('--image-root', type=Path, default=IMAGES)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--priority-images', type=int, default=48)
    args = p.parse_args()
    if args.priority_images < 1:
        p.error('--priority-images debe ser >=1')
    project = args.project_root
    out = args.out or project / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    queue_path = out / 'review_queue.csv'
    if queue_path.exists():
        raise FileExistsError(f'La cola ya existe. No se sobrescribe para proteger una revisión manual: {queue_path}')
    print('=== GVIS: PREPARACION REVISION GT ELPD 735 ===', flush=True)
    data, source_counts = validate_and_merge(project, args.image_root)
    if source_counts != Counter({'strict_verified': 76, 'strict_corrected': 4,
                                 'filename_candidate': 653, 'external_verified': 2}):
        raise ValueError(f'Origen GT inesperado: {source_counts}')
    cols = ['review_order', 'priority_num', 'priority_reason', 'contact_sheet',
            'image_path', 'scenario', 'gt_source', 'gt_before_review',
            'suggested_gt_NOT_VERIFIED', 'agreeing_models', 'models_matching_current_gt']
    cols += [f'pred_{n}' for n in MODEL_DIRS]
    cols += [f'confidence_{n}' for n in MODEL_DIRS]
    cols += ['review_decision', 'image_checked', 'corrected_gt', 'reviewer', 'review_notes']
    for i, row in enumerate(data):
        row['review_order'] = i+1
        row['contact_sheet'] = f"contact_sheets/gt_review_{i//8 + 1:03d}.jpg"
    pages = []
    for i in range(0, len(data), 8):
        page = out / 'contact_sheets' / f'gt_review_{i//8 + 1:03d}.jpg'
        render_sheet(data[i:i+8], args.image_root, page)
        pages.append(page)
    initial = data[:min(args.priority_images, len(data))]
    pack = out / 'priority_review_pack.zip'
    with ZipFile(pack, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('priority_review.csv', csv_bytes(initial, cols))
        for page in pages[:math.ceil(len(initial)/8)]:
            z.write(page, arcname='contact_sheets/' + page.name)
        for row in initial:
            orig = args.image_root / row['image_path']
            z.write(orig, arcname='original_images/' + Path(row['image_path']).name)
    queue_path.write_bytes(csv_bytes(data, cols))
    count_priority = Counter(r['priority_num'] for r in data)
    summary = {
        'dataset': 'ELPD 735 (manifiesto completo; GT auditado superpuesto)',
        'manifest_sha256': EXPECTED_MANIFEST_SHA256,
        'total_images': EXPECTED_COUNT,
        'protected_strict_audited': 80,
        'review_queue_size': len(data),
        'source_counts': dict(source_counts),
        'priority_counts': dict(sorted(count_priority.items())),
        'priority_review_pack_images': len(initial),
        'contact_sheets': len(pages),
        'auto_corrections_applied': 0,
        'warning': 'Consenso OCR no es evidencia suficiente para cambiar GT; solo revisión humana de la imagen original.',
    }
    (out / 'review_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    guide = (
        'REVISION HUMANA DEL GROUND TRUTH ELPD\n'
        '=====================================\n'
        '1. Abrir priority_review_pack.zip y comprobar visualmente cada matrícula.\n'
        '2. Para cada fila de review_queue.csv, rellenar:\n'
        '   review_decision = confirm_gt | correct | unreadable | pending\n'
        '   image_checked = yes (solo tras ver la imagen original)\n'
        '   corrected_gt = NUEVAETIQUETA (solo si decision=correct; A-Z0-9)\n'
        '   reviewer y review_notes = identificacion / justificacion\n'
        '3. La sugerencia basada en consenso de modelos NO es una correccion.\n'
        '4. Mantener las 80 etiquetas estrictamente auditadas sin cambios.\n'
        '5. El CSV completo contiene tambien los casos donde los 4 modelos\n'
        '   coinciden con el GT: todos los candidatos requieren auditoria.\n'
        '6. No modificar las columnas de image_path, gt_before_review ni gt_source.\n'
        '7. Tras completar revisiones, validar y generar un nuevo manifiesto,\n'
        '   sin alterar external_eval_manifest.csv ni las predicciones originales.\n'
    )
    (out / 'README_revision.txt').write_text(guide, encoding='utf-8')
    print('GT original:', dict(source_counts), flush=True)
    print('Cola para revision:', len(data), 'imagenes', flush=True)
    print('Prioridades:', dict(sorted(count_priority.items())), flush=True)
    print('Hojas de contacto:', len(pages), flush=True)
    print('CSV:', queue_path, flush=True)
    print('ZIP prioritario:', pack, flush=True)
    print('RESULTADO: REVISION PREPARADA; 0 GT MODIFICADOS', flush=True)


if __name__ == '__main__':
    main()
