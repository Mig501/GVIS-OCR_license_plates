#!/usr/bin/env python3
"""Paso 5 | PARSeq x ELPD V17: intervenciones visuales controladas.

Objetivo: comprobar si modificar SOLO elementos graficos cambia las inserciones
ya observadas en la etapa autoregresiva de PARSeq (paso 3).

Configuracion por defecto (sin argumentos para la task GVIS: Ejecutar en GPU):
  python scripts/analysis/parseq/analyze_parseq_elpd_visual_interventions.py

Preflight (sin GPU, valida paths, regiones, crea vistas previas):
  python scripts/analysis/parseq/analyze_parseq_elpd_visual_interventions.py --prepare-only

- No modifica checkpoint, GT ni datos originales.
- Usa el mismo Resize bicubico y Normalize(0.5, 0.5) de los pasos 3/4.
- Interviene sobre el RGB de 32x128 que recibe PARSeq, NO sobre la foto sin
  preprocesar. Por ello las coordenadas en RECTANGULOS son en pixeles 128x32.
- Dos casos / dos regiones por caso / dos tratamientos por region:
    * relleno_gris: borrar el contenido con RGB=(235,235,235)
    * desenfoque: aplicar GaussianBlur(radius=3) solo a la region
- La region 'control' es otra region NO alfanumerica, no un control causal
  perfecto. Las modificaciones introducen cambios de distribucion; incluso
  un cambio de prediccion no atribuye la causa a una capa concreta.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_COMPARISON = PROJECT / 'results/analysis/PARSeq/ELPD/insertion_stages_v1/comparacion_AR_refinamiento.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/visual_interventions_v1'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'

# (x0,y0,x1,y1), límites superiores incluidos, inferiores excluidos.
# Se han elegido sobre los recortes originales ya redimensionados a 128x32;
# no deben cubrir caracteres de matrícula. Inspeccionar vistas previas.
CASES = [
    {
        'numero': 1,
        'slug': 'ABA1234',
        'image_path': 'images/train__ABA1234.JPG',
        'gt': 'ABA1234',
        'ar_esperado': 'BABA1234',
        'ref_esperado': 'BABA1234',
        'regiones': [
            {'nombre': 'banda_azul_izquierda', 'tipo': 'objetivo', 'bbox': (0, 4, 11, 30),
             'descripcion': 'Banda azul con distintivo europeo y letra D; no tapa la A real'},
            {'nombre': 'escudo_central', 'tipo': 'control_alternativo', 'bbox': (40, 3, 54, 28),
             'descripcion': 'Distintivos centrales no alfanumericos, otra region grafica'},
        ],
    },
    {
        'numero': 2,
        'slug': 'DANKE82',
        'image_path': 'images/test__DANKE82.png',
        'gt': 'DANKE82',
        'ar_esperado': 'DANSKE82',
        'ref_esperado': 'DANSKE82',
        'regiones': [
            {'nombre': 'escudos_centrales', 'tipo': 'objetivo', 'bbox': (50, 8, 61, 28),
             'descripcion': 'Distintivos entre la N y la K; no tapa caracteres del GT'},
            {'nombre': 'margen_izquierdo', 'tipo': 'control_alternativo', 'bbox': (0, 8, 11, 28),
             'descripcion': 'Margen y borde izquierdo, sin caracteres de la matricula'},
        ],
    },
]
MODOS = ('relleno_gris', 'desenfoque')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_plate(text: Any) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())


def safe_image(root: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or not path.parts or path.parts[0] != 'images' or '..' in path.parts:
        raise ValueError(f'Ruta insegura: {rel}')
    path = (root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera de ELPD: {rel}')
    if not path.is_file():
        raise FileNotFoundError(f'No se encuentra el recorte: {path}')
    return path


def load_references(csv_path: Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'Falta CSV del paso 3: {csv_path}')
    with csv_path.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        needed = {'image_path', 'ground_truth', 'AR_sin_refinamiento',
                  'AR_mas_1_refinamiento', 'coincide_con_prediccion_guardada_v17'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV no contiene las columnas esperadas: {needed - set(reader.fieldnames or [])}')
        rows = {r['image_path']: r for r in reader}
    for case in CASES:
        row = rows.get(case['image_path'])
        if row is None:
            raise ValueError(f'Falta el caso del paso 3: {case["image_path"]}')
        expected = (case['gt'], case['ar_esperado'], case['ref_esperado'])
        actual = (row['ground_truth'], row['AR_sin_refinamiento'], row['AR_mas_1_refinamiento'])
        if expected != actual or row['coincide_con_prediccion_guardada_v17'] != '1':
            raise ValueError(f'La referencia del paso 3 no coincide para {case["slug"]}: {actual}')


def make_variants(img, regions):
    """Produce dicts (imagen, region, modo). El original siempre es la referencia."""
    from PIL import Image, ImageFilter
    variants = [{'id': 'original', 'region': 'ninguna', 'tipo_region': 'original',
                 'modo': 'sin_cambios', 'bbox': None, 'image': img.copy()}]
    for region in regions:
        x0, y0, x1, y1 = region['bbox']
        if not (0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height):
            raise ValueError(f'ROI fuera de la imagen: {region}')
        for modo in MODOS:
            variant = img.copy()
            if modo == 'relleno_gris':
                variant.paste(Image.new('RGB', (x1 - x0, y1 - y0), (235, 235, 235)), (x0, y0))
            else:
                crop = img.crop((x0, y0, x1, y1))
                blurred = crop.filter(ImageFilter.GaussianBlur(radius=3))
                variant.paste(blurred, (x0, y0))
            variants.append({
                'id': f'{region["nombre"]}__{modo}',
                'region': region['nombre'], 'tipo_region': region['tipo'],
                'modo': modo, 'bbox': (x0, y0, x1, y1), 'image': variant,
            })
    return variants


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def is_subsequence(a: str, b: str) -> bool:
    """Solo distingue insercion pura (a subsecuencia de b); no atribuye posicion."""
    it = iter(b)
    return all(any(c == d for d in it) for c in a)


def classify(gt: str, prediction: str) -> str:
    if gt == prediction:
        return 'correcta'
    if len(prediction) > len(gt) and is_subsequence(gt, prediction):
        return 'insercion_pura_persiste'
    if len(prediction) > len(gt):
        return 'longitud_excesiva_con_otros_errores'
    if len(prediction) == len(gt):
        return 'longitud_correcta_pero_error'
    return 'longitud_insuficiente'


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def make_preview(case: dict, variants: list[dict], out_path: Path, predictions: dict | None = None):
    """Muestra el parche realmente alterado, con borde rojo sin tapar el RGB guardado."""
    from PIL import Image, ImageDraw
    scale = 8
    tile_w, tile_h = 1024, 256
    outer_margin, top, caption_h, gap = 25, 78, 84, 18
    width = tile_w + outer_margin * 2
    height = top + len(variants) * (tile_h + caption_h + gap)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((outer_margin, 9), f'PASO 5 | ELPD | {case["slug"]} | GT={case["gt"]}',
              fill=(15, 35, 55), font=font(26))
    draw.text((outer_margin, 44), f'AR original={case["ar_esperado"]} | "ROI" en coordenadas de 128x32',
              fill=(75, 75, 75), font=font(18))
    for i, variant in enumerate(variants):
        base_y = top + i * (tile_h + caption_h + gap)
        text = variant['id']
        pred = predictions.get(variant['id']) if predictions else None
        if pred:
            text += f' | AR={pred["ar"]} | Refinado={pred["refinado"]}'
        draw.text((outer_margin, base_y + 3), text, fill=(10, 45, 68), font=font(20))
        if variant['bbox'] is not None:
            draw.text((outer_margin, base_y + 31), f'Region={variant["tipo_region"]}; bbox={variant["bbox"]}',
                      fill=(100, 72, 64), font=font(15))
        image_y = base_y + caption_h
        expanded = variant['image'].resize((tile_w, tile_h), resample=Image.Resampling.NEAREST)
        canvas.paste(expanded, (outer_margin, image_y))
        if variant['bbox'] is not None:
            x0, y0, x1, y1 = variant['bbox']
            draw.rectangle((outer_margin + x0 * scale, image_y + y0 * scale,
                            outer_margin + x1 * scale - 1, image_y + y1 * scale - 1),
                           outline=(235, 45, 45), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, optimize=True)


def predict(model, core, tensor):
    """Rerun exacto con refinamiento 0/1; restaurar siempre la configuracion."""
    import torch
    old_refine = int(core.refine_iters)
    outputs = {}
    try:
        for name, n_iters in [('ar', 0), ('refinado', 1)]:
            core.refine_iters = n_iters
            with torch.inference_mode():
                logits = model(tensor)
                probs = logits.softmax(-1)
                text, confidence = model.tokenizer.decode(probs)
            norm = normalize_plate(text[0])
            outputs[name] = {
                'prediction': norm,
                'raw': text[0],
                'confidence': float(confidence[0].prod().item()),
                'probs': probs[0].detach().cpu(),
            }
    finally:
        core.refine_iters = old_refine
    return outputs


def run(args):
    from PIL import Image

    load_references(args.comparison)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'No sobrescribiremos resultados: {args.out}')
    print('=== PASO 5 | PARSEQ ELPD V17 | INTERVENCIONES VISUALES ===', flush=True)
    print('CSV paso 3:', args.comparison, flush=True)
    print('Dataset:', args.image_root, flush=True)
    print('Resultado:', args.out, flush=True)
    originals = {}
    for case in CASES:
        source = safe_image(args.image_root, case['image_path'])
        with Image.open(source) as image:
            img = image.convert('RGB').resize((128, 32), resample=Image.Resampling.BICUBIC)
        originals[case['slug']] = img
    if args.out.exists():
        raise FileExistsError(f'Carpeta ya existente (evitamos pisarla): {args.out}')
    args.out.mkdir(parents=True)
    (args.out / 'comparativas').mkdir()
    (args.out / 'imagenes_procesadas').mkdir()
    plan = {'casos': [{k: v for k, v in case.items()} for case in CASES],
            'preprocesado_px': [128, 32], 'modos': list(MODOS),
            'nota': 'Control alternativo no equivale a placebo. No atribuir causa a un componente solo con estos datos.'}
    (args.out / 'plan_regiones.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    all_variants = {}
    for case in CASES:
        variants = make_variants(originals[case['slug']], case['regiones'])
        all_variants[case['slug']] = variants
        make_preview(case, variants, args.out / 'comparativas' / f'{case["numero"]:02d}_{case["slug"]}_preview.png')
        for var in variants:
            var['image'].save(args.out / 'imagenes_procesadas' / f'{case["slug"]}__{var["id"]}.png')
        print(f'[OK] {case["slug"]}: {len(variants)} variantes (1 original + 4 intervenidas)', flush=True)
    if args.prepare_only:
        print('RESULTADO: PREPARACION VALIDA; VISTAS PREVIAS GENERADAS SIN GPU', flush=True)
        return

    import numpy as np
    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; ejecutar en GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True).eval().cuda()
    core = model.model
    if not bool(core.decode_ar) or int(core.refine_iters) != 1:
        raise RuntimeError('Checkpoint debe usar decode_ar=True y refine_iters=1 como en paso 3')
    if tuple(int(x) for x in model.hparams.img_size) != (32, 128):
        raise RuntimeError(f'Dimensiones de entrada inesperadas: {model.hparams.img_size}')
    rgb_to_tensor = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    resize_reference = T.Resize((32, 128), T.InterpolationMode.BICUBIC)
    eos_id = int(model.tokenizer.eos_id)
    rows = []
    predictions_for_figures = {}
    for case in CASES:
        slug, gt = case['slug'], case['gt']
        path = safe_image(args.image_root, case['image_path'])
        with Image.open(path) as source_img:
            reference_preprocess = resize_reference(source_img.convert('RGB'))
        if not np.array_equal(np.asarray(reference_preprocess), np.asarray(originals[slug])):
            raise RuntimeError(f'Preprocesado PIL diferente del Resize de torchvision: {slug}')
        predictions_for_figures[slug] = {}
        for var in all_variants[slug]:
            x = rgb_to_tensor(var['image']).unsqueeze(0).cuda()
            predicted = predict(model, core, x)
            ar, ref = predicted['ar']['prediction'], predicted['refinado']['prediction']
            if var['id'] == 'original' and (ar != case['ar_esperado'] or ref != case['ref_esperado']):
                raise RuntimeError(f'REPRODUCCION FALLIDA para {slug}: AR={ar} REF={ref}; '
                                   f'esperado AR={case["ar_esperado"]} REF={case["ref_esperado"]}')
            predictions_for_figures[slug][var['id']] = {'ar': ar, 'refinado': ref}
            row = {
                'image_path': case['image_path'], 'caso': slug, 'gt': gt, 'variante': var['id'],
                'tipo_region': var['tipo_region'], 'region': var['region'], 'modo': var['modo'],
                'bbox_x0_y0_x1_y1': str(var['bbox']) if var['bbox'] else '',
                'AR': ar, 'refinado': ref,
                'distancia_AR': edit_distance(gt, ar), 'distancia_refinado': edit_distance(gt, ref),
                'clasificacion_AR': classify(gt, ar), 'clasificacion_refinado': classify(gt, ref),
                'correcto_AR': int(ar == gt), 'correcto_refinado': int(ref == gt),
                'insercion_eliminada_AR': int(len(ar) <= len(gt)),
                'insercion_eliminada_refinado': int(len(ref) <= len(gt)),
                'confianza_AR': f'{predicted["ar"]["confidence"]:.6f}',
                'confianza_refinado': f'{predicted["refinado"]["confidence"]:.6f}',
            }
            for stage, tag in [('ar', 'AR'), ('refinado', 'refinado')]:
                probs = predicted[stage]['probs']
                row['p_EOS_tras_gt_'+tag] = (
                    f'{float(probs[len(gt), eos_id]):.8f}' if len(probs) > len(gt) else '')
            rows.append(row)
            print(f'[{slug} | {var["id"]}] AR={ar}, refinado={ref}, '
                  f'EM_AR={ar==gt}, EM_REF={ref==gt}', flush=True)
        make_preview(case, all_variants[slug],
                     args.out / 'comparativas' / f'{case["numero"]:02d}_{slug}_resultados.png',
                     predictions_for_figures[slug])
    columns = ['image_path','caso','gt','variante','tipo_region','region','modo','bbox_x0_y0_x1_y1',
               'AR','refinado','distancia_AR','distancia_refinado','clasificacion_AR',
               'clasificacion_refinado','correcto_AR','correcto_refinado',
               'insercion_eliminada_AR','insercion_eliminada_refinado',
               'confianza_AR','confianza_refinado','p_EOS_tras_gt_AR','p_EOS_tras_gt_refinado']
    write_csv(args.out / 'comparacion_intervenciones.csv', rows, columns)
    summary = {
        'paso': 5, 'modelo': 'PARSeq oficial pretrained', 'dataset': 'ELPD V17',
        'image_size': [32,128], 'casos': len(CASES), 'originales_reproducidos': len(CASES),
        'variantes_totales': len(rows), 'variantes_modificadas': len(rows)-len(CASES),
        'sha256_comparacion_paso3': sha256(args.comparison),
        'checkpoint_modificado': False, 'GT_modificado': False,
        'resultados_por_caso': {
            case['slug']: [{k: v for k, v in r.items() if k in (
                'variante', 'tipo_region', 'AR','refinado','clasificacion_AR','clasificacion_refinado',
                'correcto_AR','correcto_refinado','p_EOS_tras_gt_AR')} for r in rows if r['caso']==case['slug']]
            for case in CASES
        },
        'limitaciones': [
            'Una correccion al modificar la imagen prueba sensibilidad a la region, no que un unico bloque de la red sea responsable.',
            'Las imagenes intervenidas pueden estar fuera de la distribucion natural.',
            'El control es una region grafica alternativa, no placebo perfecto.',
            'Se evaluan dos casos seleccionados, no una estimacion poblacional.'
        ],
    }
    (args.out / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('RESULTADO: 2 ORIGINALES REPRODUCIDOS; COMPARACIONES DE INTERVENCION GUARDADAS', flush=True)
    print('SALIDA:', args.out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true', help='Valida y genera vistas previas sin usar GPU')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
