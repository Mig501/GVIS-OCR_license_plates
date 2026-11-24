#!/usr/bin/env python3
"""PARSeq x ELPD GT V17: origen temporal de inserciones en el decoder.

Compara AR sin refinamiento y AR + 1 refinamiento sobre los mismos recortes
(una imagen por llamada para evitar interacciones de parada temprana por lote).
No modifica pesos, checkpoints, dataset, CSV de referencia ni codigo oficial.

Uso habitual: python scripts/baselines/analyze_parseq_elpd_insertion_stages.py
Preflight CSV (sin CUDA): ... --prepare-only
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import time

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_CSV = PROJECT / 'results/audit/ELPD/gt_corrections_v17/parseq/predictions_verified_v17.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/insertion_stages_v1'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'

PRIORITY = {
    'inicial': ['images/train__ABA1234.JPG', 'images/test__ES484LJ.jpg', 'images/test__ESCC690.png'],
    'interna': ['images/test__DANKE82.png', 'images/test__BL654BP.png', 'images/test__5888881.png'],
    'final': ['images/test__CN498JC.png', 'images/test__BZ310SW.jpg', 'images/test__CV194VA.jpg'],
}


def normalize_plate(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def insertion_type(gt: str, pred: str) -> str | None:
    """Only *pure* insertions: GT is recovered by deleting extra symbols from pred."""
    if len(pred) <= len(gt):
        return None
    ops = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2
           in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes() if tag != 'equal']
    if not ops or not all(tag == 'insert' for tag, *_ in ops):
        return None
    if len(ops) == 1 and ops[0][1] == 0:
        return 'inicial'
    if len(ops) == 1 and ops[0][1] == len(gt):
        return 'final'
    return 'interna'


def read_and_select(path: Path, max_per_type: int):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        required = {'image_path', 'ground_truth', 'prediction_normalized', 'exact_match'}
        missing = required - set(rdr.fieldnames or [])
        if missing:
            raise ValueError(f'CSV V17 sin columnas {sorted(missing)}: {path}')
        rows = list(rdr)
    if len(rows) != 733:
        raise ValueError(f'Se esperan 733 predicciones verificadas de V17, hay {len(rows)}')
    if len({row['image_path'] for row in rows}) != len(rows):
        raise ValueError('Hay rutas de imagen duplicadas')
    if sum(row['exact_match'] == '0' for row in rows) != 216:
        raise ValueError('El CSV no coincide con los 216 fallos verificados de PARSeq V17')

    by_type = {'inicial': [], 'interna': [], 'final': []}
    for row in rows:
        if row['exact_match'] != '0':
            continue
        gt = row['ground_truth'].strip()
        pred = row['prediction_normalized'].strip()
        kind = insertion_type(gt, pred)
        if kind:
            row['tipo_insercion_pura'] = kind
            by_type[kind].append(row)

    selected = []
    for kind in ('inicial', 'interna', 'final'):
        members = {r['image_path']: r for r in by_type[kind]}
        chosen = []
        for name in PRIORITY[kind]:
            if name in members and name not in chosen:
                chosen.append(name)
        chosen.extend(sorted(n for n in members if n not in chosen))
        selected.extend(members[n] for n in chosen[:max_per_type])
    return rows, selected, {k: len(v) for k, v in by_type.items()}


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    with path.open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(2 ** 20), b''):
            h.update(chunk)
        return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                    help='Predicciones verificadas de PARSeq, ELPD V17 (733 filas)')
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--max-per-type', type=int, default=4,
                    help='Ejemplos por posicion: inicial/interna/final')
    ap.add_argument('--prepare-only', action='store_true',
                    help='Validar CSV y plan de 12 casos sin cargar imagenes ni GPU')
    args = ap.parse_args()
    if args.max_per_type < 1:
        ap.error('--max-per-type debe ser positivo')
    if not args.csv.is_file():
        raise FileNotFoundError(f'Falta el CSV de PARSeq V17: {args.csv}')

    rows, selected, counts = read_and_select(args.csv, args.max_per_type)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'La carpeta de salida ya contiene archivos: {args.out}. No se sobrescribe.')
    args.out.mkdir(parents=True, exist_ok=True)

    select_cols = ['image_path', 'tipo_insercion_pura', 'ground_truth', 'prediction_normalized',
                   'confidence', 'gt_validation_status_v17']
    write_csv(args.out / 'casos_seleccionados.csv', selected, select_cols)
    print('=== PARSEQ ELPD V17: PRE-AR vs POST-REFINAMIENTO ===', flush=True)
    print('CSV verificado:', len(rows), 'filas, 216 predicciones incorrectas', flush=True)
    print('Inserciones puras en el CSV:', counts, flush=True)
    print('Ejemplos seleccionados:', len(selected), flush=True)
    print('SHA256 CSV:', sha256(args.csv), flush=True)
    if args.prepare_only:
        print('RESULTADO: SELECCION PREPARADA; SIN INFERENCIA', flush=True)
        return

    import torch
    from PIL import Image
    from torchvision import transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; ejecutar con GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True)
    model = model.eval().cuda()
    core = model.model
    if not bool(core.decode_ar):
        raise RuntimeError('El checkpoint no tiene decode_ar=True; no se corresponde con la baseline')
    default_refine = int(core.refine_iters)
    if default_refine != 1:
        raise RuntimeError(f'Baseline esperada con refine_iters=1; modelo cargado tiene {default_refine}')
    print('Modelo cargado en:', torch.cuda.get_device_name(0), flush=True)
    print('decode_ar:', core.decode_ar, '| refine_iters original:', default_refine, flush=True)
    transform = T.Compose([
        T.Resize(tuple(model.hparams.img_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    eos_id = model.tokenizer.eos_id
    token_vocab = model.tokenizer._itos
    comparison = []
    step_rows = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for idx, row in enumerate(selected, 1):
                rel = row['image_path']
                relpath = Path(rel)
                if relpath.is_absolute() or '..' in relpath.parts or relpath.parts[0] != 'images':
                    raise ValueError(f'Ruta insegura en CSV: {rel}')
                image_path = args.image_root / relpath
                if not image_path.is_file():
                    raise FileNotFoundError(f'Falta el recorte de ELPD: {image_path}')
                with Image.open(image_path) as im:
                    x = transform(im.convert('RGB')).unsqueeze(0).cuda()
                preds = {}
                logits_map = {}
                for label, iters in [('AR_sin_refinamiento', 0), ('AR_mas_1_refinamiento', 1)]:
                    core.refine_iters = iters
                    logits = model(x)  # misma imagen, checkpoint, preprocess y decoding AR
                    probs = logits.softmax(-1).cpu()[0]
                    text, conf = model.tokenizer.decode(probs.unsqueeze(0))
                    preds[label] = {'raw': text[0], 'normalized': normalize_plate(text[0]),
                                    'confidence': round(float(conf[0].prod().item()), 6)}
                    logits_map[label] = probs
                    for pos, dist in enumerate(probs):
                        token_id = int(dist.argmax().item())
                        step_rows.append({
                            'image_path': rel, 'etapa': label, 'posicion': pos + 1,
                            'token_max': token_vocab[token_id],
                            'prob_token_max': round(float(dist[token_id]), 6),
                            'prob_eos': round(float(dist[eos_id]), 6),
                        })
                gt = row['ground_truth']
                before = preds['AR_sin_refinamiento']['normalized']
                after = preds['AR_mas_1_refinamiento']['normalized']
                reference = row['prediction_normalized']
                original_matches = after == reference
                if len(before) > len(gt) and len(after) > len(gt):
                    verdict = 'insercion_ya_en_AR_y_persiste'
                elif len(before) <= len(gt) and len(after) > len(gt):
                    verdict = 'insercion_aparece_en_refinamiento'
                elif len(before) > len(gt) and len(after) <= len(gt):
                    verdict = 'refinamiento_elimina_exceso_de_longitud'
                else:
                    verdict = 'otro_comportamiento'
                comparison.append({
                    'image_path': rel, 'tipo_insercion_pura': row['tipo_insercion_pura'],
                    'ground_truth': gt, 'prediccion_original_v17': reference,
                    'AR_sin_refinamiento': before, 'AR_mas_1_refinamiento': after,
                    'confianza_AR': preds['AR_sin_refinamiento']['confidence'],
                    'confianza_refinamiento': preds['AR_mas_1_refinamiento']['confidence'],
                    'coincide_con_prediccion_guardada_v17': int(original_matches),
                    'clasificacion_temporal': verdict if original_matches else 'NO_REPRODUCIDO',
                    'p_eos_AR_en_posicion_longitud_GT_mas_1': round(float(logits_map['AR_sin_refinamiento'][len(gt), eos_id]), 6)
                    if len(logits_map['AR_sin_refinamiento']) > len(gt) else '',
                    'p_eos_refinamiento_en_posicion_longitud_GT_mas_1': round(float(logits_map['AR_mas_1_refinamiento'][len(gt), eos_id]), 6)
                    if len(logits_map['AR_mas_1_refinamiento']) > len(gt) else '',
                })
                print(f'[{idx:02d}/{len(selected)}] {rel} | GT={gt} | AR={before} | REFINADO={after} | '
                      f'V17={reference} | reproducido={original_matches}', flush=True)
    finally:
        core.refine_iters = default_refine

    compare_cols = ['image_path', 'tipo_insercion_pura', 'ground_truth', 'prediccion_original_v17',
                    'AR_sin_refinamiento', 'AR_mas_1_refinamiento', 'confianza_AR',
                    'confianza_refinamiento', 'coincide_con_prediccion_guardada_v17',
                    'clasificacion_temporal', 'p_eos_AR_en_posicion_longitud_GT_mas_1',
                    'p_eos_refinamiento_en_posicion_longitud_GT_mas_1']
    step_cols = ['image_path', 'etapa', 'posicion', 'token_max', 'prob_token_max', 'prob_eos']
    write_csv(args.out / 'comparacion_AR_refinamiento.csv', comparison, compare_cols)
    write_csv(args.out / 'probabilidades_por_posicion.csv', step_rows, step_cols)
    summary = {
        'dataset': 'ELPD V17 verificado', 'total_verified': len(rows),
        'original_parseq_errors': 216, 'pure_insertion_counts': counts,
        'selected': len(selected), 'selected_by_type': dict(Counter(r['tipo_insercion_pura'] for r in selected)),
        'reproduced_original_predictions': sum(int(r['coincide_con_prediccion_guardada_v17']) for r in comparison),
        'classification_only_reproduced': dict(Counter(r['clasificacion_temporal'] for r in comparison
                                                       if r['clasificacion_temporal'] != 'NO_REPRODUCIDO')),
        'duration_seconds': round(time.perf_counter() - started, 2),
        'original_weights_unchanged': True, 'original_files_unchanged': True,
        'caution': 'Un cambio de etapa localiza cuando aparece la insercion; no demuestra causa en encoder/atencion.',
    }
    (args.out / 'resumen.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Reproduccion de predicciones V17:', summary['reproduced_original_predictions'], '/', len(selected), flush=True)
    print('Resultado por etapa:', summary['classification_only_reproduced'], flush=True)
    print('RESULTADO: COMPARACION AR/REFINAMIENTO GENERADA (PESOS INTACTOS)', flush=True)
    print('SALIDA:', args.out, flush=True)


if __name__ == '__main__':
    main()
