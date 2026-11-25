#!/usr/bin/env python3
"""PASO 9 · PARSeq/ELPD V17: ablation 0 vs 1 iteracion de refinamiento.

Compara 733 GT verificados con el mismo checkpoint oficial, imagen y
preprocesado; una imagen por forward para que la parada autoregresiva
no dependa de otras imagenes del batch.

Ruta: scripts/analysis/parseq/analyze_parseq_elpd_refinement_ablation.py
Tarea GVIS GPU: no necesita argumentos; salida en results/analysis/PARSeq/ELPD/.
Preflight sin GPU: python <script> --prepare-only --csv /ruta/al/CSV.csv

No modifica pesos, GT, checkpoints ni CSV previos.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
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
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/refinement_ablation_v1'
NORMALIZE_RE = re.compile(r'[^A-Z0-9]')


def normalize(raw: str) -> str:
    return NORMALIZE_RE.sub('', raw.upper())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_v17(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        required = {'image_path', 'scenario', 'ground_truth', 'prediction_raw',
                    'prediction_normalized', 'exact_match'}
        if missing := (required - set(r.fieldnames or [])):
            raise ValueError(f'Columnas ausentes en V17: {sorted(missing)}')
        rows = list(r)
    if len(rows) != 733:
        raise ValueError(f'Se esperaban 733 registros V17, recibidos {len(rows)}')
    if len({r['image_path'] for r in rows}) != 733:
        raise ValueError('Hay image_path duplicados en el V17')
    for row in rows:
        gt = row['ground_truth']
        pred = row['prediction_normalized']
        if normalize(gt) != gt or normalize(pred) != pred:
            raise ValueError(f'GT o prediccion no normalizados: {row["image_path"]}')
        if str(int(gt == pred)) != row['exact_match']:
            raise ValueError(f'exact_match inconsistente: {row["image_path"]}')
        if row['scenario'] not in ('train', 'val', 'test'):
            raise ValueError(f'split desconocido: {row["scenario"]}')
    if sum(r['exact_match'] == '1' for r in rows) != 517:
        raise ValueError('No coinciden los 517 aciertos PARSeq de ELPD V17')
    if sum(len(r['ground_truth']) for r in rows) != 5057:
        raise ValueError('No coinciden los 5057 caracteres GT de ELPD V17')
    return rows


def safe_image(root: Path, rel: str) -> Path:
    p = Path(rel)
    if (p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'images'):
        raise ValueError(f'Ruta de imagen insegura: {rel}')
    path = root / p
    if not path.is_file():
        raise FileNotFoundError(f'Imagen ELPD no encontrada: {path}')
    return path


def edit_distance(a: str, b: str) -> int:
    """Levenshtein unitario, sin dependencia de paquetes externos."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[-1] + 1, prev[j] + 1,
                               prev[j - 1] + (ca != cb)))
        prev = current
    return prev[-1]


def category(gt: str, pred: str) -> str:
    """Categorias mutuamente excluyentes; tipos PUROS segun SequenceMatcher."""
    if gt == pred:
        return 'correcta'
    ops = [op for op, *_ in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes()
           if op != 'equal']
    if ops and all(op == 'insert' for op in ops):
        return 'insercion_pura'
    if ops and all(op == 'delete' for op in ops):
        return 'omision_pura'
    if ops and all(op == 'replace' for op in ops) and len(gt) == len(pred):
        return 'sustitucion_pura'
    return 'mixto'


def insertion_subtype(gt: str, pred: str) -> str:
    """Misma definicion usada en el Paso 8 para tipos de insercion pura."""
    if category(gt, pred) != 'insercion_pura':
        return ''
    differences = [(tag, i1, i2, j1, j2)
                   for tag, i1, i2, j1, j2 in
                   SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes()
                   if tag != 'equal']
    if len(differences) == 1 and differences[0][1] == 0:
        return 'inicial'
    if len(differences) == 1 and differences[0][1] == len(gt):
        return 'final'
    return 'interna'


def parse_prediction(probs, vocab: list[str], eos_id: int) -> dict:
    """Argmax logit hasta EOS, CONSERVANDO simbolos antes de normalizar."""
    if probs.ndim != 2 or probs.shape[1] > len(vocab):
        raise ValueError(f'Dimensiones de logits invalidas: {tuple(probs.shape)}')
    ids = probs.argmax(-1).tolist()
    eos_at = ids.index(eos_id) if eos_id in ids else None
    tokens = [vocab[i] for i in ids[:eos_at if eos_at is not None else len(ids)]]
    raw = ''.join(tokens)
    return {
        'raw': raw,
        'normalized': normalize(raw),
        'eos_position_1based': eos_at + 1 if eos_at is not None else None,
        'n_logits': len(ids),
        'tokens': tokens,
    }


def save_csv(path: Path, rows: list[dict], headers: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def compute_metrics(rows: list[dict], prefix: str) -> dict:
    n = len(rows)
    if not n:
        return {'n': 0}
    chars = sum(len(r['ground_truth']) for r in rows)
    correct = sum(r[f'{prefix}_exact_match'] for r in rows)
    edits = sum(r[f'{prefix}_edit_distance'] for r in rows)
    counts = Counter(r[f'{prefix}_tipo_error'] for r in rows)
    insertions = Counter(r[f'{prefix}_subtipo_insercion'] for r in rows
                         if r[f'{prefix}_tipo_error'] == 'insercion_pura')
    return {
        'n': n,
        'correctas': correct,
        'incorrectas': n - correct,
        'em_pct': round(100 * correct / n, 4),
        'cer_pct': round(100 * edits / chars, 4),
        'ediciones': edits,
        'caracteres_gt': chars,
        'errores_por_categoria_exclusiva': dict(counts),
        'inserciones_puras_por_tipo': dict(insertions),
        'omisiones_puras': counts['omision_pura'],
        'inserciones_puras': counts['insercion_pura'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    p.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--prepare-only', action='store_true', help='Validar CSV sin GPU')
    args = p.parse_args()

    rows = read_v17(args.csv)
    print('=== PASO 9 | PARSeq ELPD V17 | AR vs REFINAMIENTO ===', flush=True)
    print('Dataset: 733 GT verificados | 517 correctas | 216 incorrectas', flush=True)
    print('Splits:', dict(Counter(r['scenario'] for r in rows)), flush=True)
    print('SHA256 CSV V17:', sha256(args.csv), flush=True)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Salida ya contiene resultados, no se sobrescribe: {args.out}')
    if args.prepare_only:
        print('PREFLIGHT OK: 733 filas validadas. NO se ha ejecutado inferencia.', flush=True)
        return

    import torch
    from PIL import Image
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: utilizar GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True)
    model.eval().cuda()
    core = model.model
    if not bool(core.decode_ar) or int(core.refine_iters) != 1:
        raise RuntimeError(f'Checkpoint/configuracion inesperados: '
                           f'decode_ar={core.decode_ar}, refine_iters={core.refine_iters}')
    if tuple(int(v) for v in model.hparams.img_size) != (32, 128):
        raise RuntimeError('Esperada entrada PARSeq 32x128')
    tokenizer = model.tokenizer
    vocab = list(tokenizer._itos)
    eos_id = int(tokenizer.eos_id)
    transform = T.Compose([
        T.Resize(tuple(model.hparams.img_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    print('GPU:', torch.cuda.get_device_name(0), flush=True)

    results = []
    mismatches = []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for i, source in enumerate(rows, start=1):
                rel = source['image_path']
                gt = source['ground_truth']
                with Image.open(safe_image(args.image_root, rel)) as im:
                    x = transform(im.convert('RGB')).unsqueeze(0).cuda()
                stages = {}
                for stage, refine_iters in (('AR', 0), ('REF', 1)):
                    core.refine_iters = refine_iters
                    logits = model(x)
                    probs = logits.softmax(-1).float().cpu()[0]
                    pred = parse_prediction(probs, vocab, eos_id)
                    official, _confidence = tokenizer.decode(probs.unsqueeze(0))
                    if official[0] != pred['raw']:
                        raise RuntimeError(f'Decode oficial distinto a tokens crudos en '
                                           f'{rel} ({stage}): {official[0]!r} vs {pred["raw"]!r}')
                    stages[stage] = pred
                ar, ref = stages['AR'], stages['REF']
                reproduced = int(ref['normalized'] == source['prediction_normalized']
                                 and ref['raw'] == source['prediction_raw'])
                if not reproduced:
                    mismatches.append((rel, source['prediction_raw'], ref['raw'],
                                       source['prediction_normalized'], ref['normalized']))
                a_edit, r_edit = edit_distance(gt, ar['normalized']), edit_distance(gt, ref['normalized'])
                a_cat, r_cat = category(gt, ar['normalized']), category(gt, ref['normalized'])
                results.append({
                    'image_path': rel,
                    'split': source['scenario'],
                    'ground_truth': gt,
                    'V17_raw': source['prediction_raw'],
                    'V17_normalizada': source['prediction_normalized'],
                    'AR_raw': ar['raw'],
                    'AR_normalizada': ar['normalized'],
                    'REF_raw': ref['raw'],
                    'REF_normalizada': ref['normalized'],
                    'reproducido_V17': reproduced,
                    'AR_exact_match': int(a_edit == 0),
                    'REF_exact_match': int(r_edit == 0),
                    'AR_edit_distance': a_edit,
                    'REF_edit_distance': r_edit,
                    'AR_tipo_error': a_cat,
                    'REF_tipo_error': r_cat,
                    'AR_subtipo_insercion': insertion_subtype(gt, ar['normalized']),
                    'REF_subtipo_insercion': insertion_subtype(gt, ref['normalized']),
                    'AR_EOS_1based': ar['eos_position_1based'] or '',
                    'REF_EOS_1based': ref['eos_position_1based'] or '',
                    'AR_n_logits': ar['n_logits'],
                    'REF_n_logits': ref['n_logits'],
                    'cambio_raw': int(ar['raw'] != ref['raw']),
                    'cambio_normalizado': int(ar['normalized'] != ref['normalized']),
                    'efecto_refinamiento_EM': ('corrige' if r_edit == 0 and a_edit != 0 else
                                              'perjudica' if a_edit == 0 and r_edit != 0 else
                                              'igual'),
                    'efecto_refinamiento_CER': ('mejora' if r_edit < a_edit else
                                               'empeora' if r_edit > a_edit else 'igual'),
                })
                if i <= 5 or i % 50 == 0 or i == len(rows):
                    print(f'[{i:03d}/{len(rows)}] {source["scenario"]:<5} '
                          f'GT={gt:<10} AR={ar["normalized"]:<12} '
                          f'REF={ref["normalized"]:<12} V17={"OK" if reproduced else "NO"}',
                          flush=True)
    finally:
        core.refine_iters = 1

    if mismatches:
        sample = '\n'.join(str(x) for x in mismatches[:8])
        raise RuntimeError(f'No se reprodujeron {len(mismatches)}/{len(rows)} predicciones V17. '
                           f'No se exportan metricas comparativas para evitar conclusiones no equivalentes. '
                           f'Primeros casos:\n{sample}')
    if compute_metrics(results, 'REF')['correctas'] != 517 or compute_metrics(results, 'REF')['ediciones'] != 327:
        raise RuntimeError('Las metricas REF no reproducen la evaluacion final (517 correctas / 327 ediciones)')

    paired = Counter(r['efecto_refinamiento_EM'] for r in results)
    paired_ed = Counter(r['efecto_refinamiento_CER'] for r in results)
    changes = [r for r in results if r['cambio_raw']]
    by_split = []
    for split in ('train', 'val', 'test'):
        sub = [r for r in results if r['split'] == split]
        for stage in ('AR', 'REF'):
            m = compute_metrics(sub, stage)
            by_split.append({'split': split, 'etapa': stage, **{
                k: v for k, v in m.items() if not isinstance(v, dict)
            }})

    before, after = compute_metrics(results, 'AR'), compute_metrics(results, 'REF')
    summary = {
        'paso': 9,
        'modelo': 'PARSeq pretrained oficial',
        'dataset': 'ELPD V17, 733 etiquetas verificadas',
        'total_imagenes': len(rows),
        'refinado_reproducido_V17': len(rows),
        'preprocesado': 'RGB, 32x128 bicubico, ToTensor, Normalize(0.5,0.5)',
        'evaluacion': 'individual por imagen, mismo checkpoint, AR refine_iters=0 vs REF refine_iters=1',
        'AR': before,
        'REF': after,
        'efecto_refinamiento_EM': dict(paired),
        'efecto_refinamiento_CER': dict(paired_ed),
        'n_cambios_raw': len(changes),
        'n_cambios_normalizados': sum(r['cambio_normalizado'] for r in results),
        'EOS_posicion_distinta_con_ambos_presentes': sum(
            bool(r['AR_EOS_1based']) and bool(r['REF_EOS_1based'])
            and r['AR_EOS_1based'] != r['REF_EOS_1based'] for r in results),
        'por_split': by_split,
        'sha256_csv_V17': sha256(args.csv),
        'duracion_segundos': round(time.perf_counter() - started, 2),
        'pesos_modificados': False,
        'GT_modificado': False,
        'advertencias': [
            'Las categorias insercion/omision/sustitucion son puras; errores mixtos se cuentan aparte.',
            'El refinamiento puede activar simbolos descartados en la normalizacion; conservar raw.',
            'Las posiciones EOS entre etapas pueden mantenerse aun con inserciones en AR.',
            'Estos 733 ejemplos incluyen los splits originales train/val/test; no es test externo.',
        ],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    save_csv(args.out / 'comparacion_733.csv', results, list(results[0]))
    save_csv(args.out / 'cambios_refinamiento.csv', changes, list(results[0]))
    save_csv(args.out / 'metricas_por_split.csv', by_split, list(by_split[0]))
    (args.out / 'resumen.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('=== RESUMEN PASO 9 ===', flush=True)
    print('AR:  EM=', before['em_pct'], '% | CER=', before['cer_pct'], '% | inserciones puras=',
          before['inserciones_puras'], flush=True)
    print('REF: EM=', after['em_pct'], '% | CER=', after['cer_pct'], '% | inserciones puras=',
          after['inserciones_puras'], flush=True)
    print('Cambios EM al refinar:', dict(paired), flush=True)
    print('Cambios CER al refinar:', dict(paired_ed), flush=True)
    print('RESULTADO: COMPARACION AR vs REFINAMIENTO GENERADA', flush=True)
    print('SALIDA:', args.out, flush=True)


if __name__ == '__main__':
    main()
