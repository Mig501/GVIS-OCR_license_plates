#!/usr/bin/env python3
"""PARSeq zero-shot sobre UC3M-LP_LPO/test (491 recortes, GT parcial).

Mismo PARSeq checkpoint/preprocesado/batch que el baseline de ELPD.
No entrena ni altera imagenes/manifest/checkpoint.

Modo CPU de solo lectura: --check-only (preflight_parseq_uc3m.py).
Modo GPU por defecto: produce predictions.csv, summary.json, summary.txt.

MUY IMPORTANTE: las metricas parcial_EM y masked_error_rate_proxy NO son
EM/CER completos y NO son directamente comparables con ELPD.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_MANIFEST = DATASETS / 'processed/UC3M-LP_LPO/manifest.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/UC3M-LP_LPO'
DEFAULT_OUT = PROJECT / 'results/baselines/UC3M-LP/parseq_pretrained'
DEFAULT_MODEL_CACHE = DATASETS / 'models/torch_hub'
MODEL_REPO = 'baudm/parseq:main'
MODEL_ENTRY = 'parseq'
CHECKPOINT_NAME = 'parseq-bb5792a6.pt'
EXPECTED_CHECKPOINT_SHA = 'bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00'
EXPECTED_HUBCONF_SHA = '1ecb24721ca4c7b7df3600d71b715a391d65d0f2a6b9bed3d5deaa06189d3c15'

sys.path.insert(0, str(PROJECT))
from src.evaluation.partial_metrics import score_partial, summarize_partial  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path, root: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f'Manifest UC3M-LP no encontrado: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {
            'image_path', 'source_path', 'text_original', 'text_normalized',
            'text_visible', 'known_mask', 'scenario', 'label_status',
            'character_count', 'known_characters', 'anonymized_characters',
        }
        if missing := required - set(reader.fieldnames or []):
            raise ValueError(f'Faltan columnas en manifest: {sorted(missing)}')
        all_rows = list(reader)
    splits = Counter(row['scenario'].strip() for row in all_rows)
    if len(all_rows) != 2547 or splits != {'train': 2056, 'test': 491}:
        raise ValueError(f'Splits UC3M distintos a los auditados: {len(all_rows)}, {dict(splits)}')
    test_rows = [r for r in all_rows if r['scenario'].strip() == 'test']
    seen: set[str] = set()
    for row in test_rows:
        rel = row['image_path'].strip()
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 2 or p.parts[0] != 'images' or not p.name.startswith('test__') or p.suffix.lower() != '.png':
            raise ValueError(f'Ruta de imagen test no permitida: {rel}')
        if rel in seen:
            raise ValueError(f'Imagen duplicada en test: {rel}')
        seen.add(rel)
        if not (root / rel).is_file():
            raise FileNotFoundError(f'No existe el PNG: {root / rel}')
        if row['label_status'] != 'partial_masked':
            raise ValueError(f'GT no marcado parcial: {rel}')
        if '/test/' not in row['source_path'].replace('\\', '/'):
            raise ValueError(f'La foto de origen no pertenece a test: {rel}')
        gt, mask = row['text_normalized'].strip(), row['known_mask'].strip()
        rec = score_partial(gt, mask, '')
        if row['text_original'].split('-', 1)[-1].upper() != gt:
            raise ValueError(f'GT distinto a text_original: {rel}')
        if row['text_visible'].strip() != gt.replace('*', ''):
            raise ValueError(f'text_visible incorrecto: {rel}')
        if (int(row['character_count']), int(row['known_characters']), int(row['anonymized_characters'])) != (
            rec['gt_length'], rec['known_characters'], rec['masked_characters']
        ):
            raise ValueError(f'Contadores GT/mascara incorrectos: {rel}')
    return test_rows, dict(splits)


def cached_model_files() -> tuple[Path, Path, Path]:
    torch_home = Path(os.environ.setdefault('TORCH_HOME', str(DEFAULT_MODEL_CACHE)))
    repo = torch_home / 'hub/baudm_parseq_main'
    return torch_home, torch_home / 'hub/checkpoints' / CHECKPOINT_NAME, repo / 'hubconf.py'


def verify_parseq_cache() -> dict[str, str]:
    torch_home, ckpt, hubconf = cached_model_files()
    if not ckpt.is_file():
        raise FileNotFoundError(f'No se encuentra el checkpoint usado en ELPD: {ckpt}')
    if not hubconf.is_file():
        raise FileNotFoundError(f'No se encuentra el codigo cacheado de PARSeq: {hubconf}')
    ckpt_sha = sha256_file(ckpt)
    hubconf_sha = sha256_file(hubconf)
    if ckpt_sha != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError(f'El checkpoint no coincide con PARSeq ELPD: {ckpt_sha}')
    if hubconf_sha != EXPECTED_HUBCONF_SHA:
        raise RuntimeError(f'El hubconf no coincide con PARSeq ELPD: {hubconf_sha}')
    return {
        'torch_home': str(torch_home),
        'repo_dir': str(hubconf.parent),
        'checkpoint_path': str(ckpt),
        'checkpoint_sha256': ckpt_sha,
        'source_hubconf_sha256': hubconf_sha,
    }


def write_results(out_dir: Path, records: list[dict], summary: dict, report: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (out_dir / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--check-only', action='store_true', help='Validar dataset y cache del modelo, sin GPU')
    args = ap.parse_args(argv)
    if args.batch_size < 1:
        ap.error('--batch-size debe ser positivo')

    print('=== GVIS: PARSEQ ZERO-SHOT / UC3M-LP ===', flush=True)
    rows, splits = load_manifest(args.manifest, args.image_root)
    manifest_sha = sha256_file(args.manifest)
    print(f'Manifest: {args.manifest}', flush=True)
    print(f'Filas: {sum(splits.values())}; Train={splits["train"]}; Test={len(rows)}', flush=True)
    print('Solo se evaluara TEST (491 PNG).', flush=True)
    print('SHA256 manifest:', manifest_sha, flush=True)
    info = verify_parseq_cache()
    print('Cache PARSeq:', info['torch_home'], flush=True)
    print('SHA256 checkpoint:', info['checkpoint_sha256'], flush=True)
    print('SHA256 hubconf:', info['source_hubconf_sha256'], flush=True)
    print('Protocolo: partial_EM estricto + distancia de edicion con wildcard (*).', flush=True)
    print('Las metricas parciales NO son comparables con EM/CER completos de ELPD.', flush=True)

    if args.check_only:
        print('ESTADO: PREFLIGHT CORRECTO; GPU E INFERENCIA PENDIENTES', flush=True)
        return 0

    from PIL import Image
    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: lanzar con GVIS: Ejecutar en GPU')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    os.environ['TORCH_HOME'] = info['torch_home']
    print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Cargando PARSeq desde la misma copia local cacheada que en ELPD...', flush=True)
    t_load = time.perf_counter()
    model = torch.hub.load(info['repo_dir'], MODEL_ENTRY, pretrained=True, source='local', trust_repo=True)
    model.eval().to(device)
    load_seconds = time.perf_counter() - t_load
    img_size = tuple(model.hparams.img_size)
    transform = T.Compose([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    print(f'Imagen de entrada: {img_size}; batch size: {args.batch_size}', flush=True)

    with Image.open(args.image_root / rows[0]['image_path']) as image:
        example = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(2):
            probabilities = model(example).softmax(-1)
            model.tokenizer.decode(probabilities)
    torch.cuda.synchronize()

    pred_rows: list[dict] = []
    total_forward_ms = 0.0
    start_wall = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            subset = rows[offset:offset + args.batch_size]
            tensors = []
            for row in subset:
                with Image.open(args.image_root / row['image_path']) as image:
                    tensors.append(transform(image.convert('RGB')))
            batch = torch.stack(tensors).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = model(batch).softmax(-1)
            labels, confidences = model.tokenizer.decode(probs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if len(labels) != len(subset):
                raise RuntimeError(f'Predicciones incompletas en batch {offset}')
            total_forward_ms += elapsed_ms
            for row, pred_raw, confidence in zip(subset, labels, confidences):
                partial = score_partial(row['text_normalized'].strip(), row['known_mask'].strip(), pred_raw)
                pred_rows.append({
                    'image_path': row['image_path'],
                    'source_path': row['source_path'],
                    'annotation_path': row.get('annotation_path', ''),
                    'view_id': row.get('view_id', ''),
                    'scenario': row['scenario'],
                    'label_status': row['label_status'],
                    'ground_truth_masked': partial['ground_truth_masked'],
                    'known_mask': partial['known_mask'],
                    'ground_truth_visible': row['text_visible'],
                    'prediction_raw': pred_raw,
                    'prediction_normalized': partial['prediction_normalized'],
                    'gt_length': partial['gt_length'],
                    'pred_length': partial['pred_length'],
                    'known_characters': partial['known_characters'],
                    'masked_characters': partial['masked_characters'],
                    'length_match': partial['length_match'],
                    'partial_exact_match': partial['partial_exact_match'],
                    'known_errors_if_aligned': partial['known_errors_if_aligned'],
                    'masked_edit_distance': partial['masked_edit_distance'],
                    'confidence': round(float(confidence.prod().item()), 6),
                    'batch_forward_ms': round(elapsed_ms, 3),
                })
            print(f'Inferencia: {len(pred_rows)}/{len(rows)}', flush=True)
    wall_seconds = time.perf_counter() - start_wall
    if len(pred_rows) != 491:
        raise RuntimeError(f'Esperadas 491 predicciones; obtenidas {len(pred_rows)}')

    metrics = summarize_partial(
        (r['ground_truth_masked'], r['known_mask'], r['prediction_normalized'])
        for r in pred_rows
    )
    summary = {
        'dataset': 'UC3M-LP',
        'dataset_partition': 'test',
        'dataset_size': len(pred_rows),
        'manifest_sha256': manifest_sha,
        'model': 'PARSeq',
        'inference_mode': 'zero-shot; pretrained; autoregressive + 1 refinement (official defaults)',
        'source': MODEL_REPO,
        'source_hubconf_sha256': info['source_hubconf_sha256'],
        'checkpoint_filename': CHECKPOINT_NAME,
        'checkpoint_sha256': info['checkpoint_sha256'],
        'finetuning': False,
        'normalization': 'uppercase ASCII A-Z/0-9 on predictions; masked GT preserves *',
        'metric_protocol': (
            'partial_EM requires equal length and matching all known positions. '
            'masked_error_rate_proxy uses wildcard Levenshtein with substitution cost 0 '
            'at *, insertion/deletion cost 1; denominator known GT chars. '
            'Not comparable to full EM/CER.'
        ),
        **metrics,
        'gpu_forward_ms_total': round(total_forward_ms, 3),
        'gpu_forward_ms_per_image': round(total_forward_ms / len(pred_rows), 3),
        'inference_wall_seconds_including_image_loading': round(wall_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'batch_size': args.batch_size,
        'image_size_hw': img_size,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'num_parameters': sum(p.numel() for p in model.parameters()),
    }
    report = [
        '=== GVIS: PARSEQ ZERO-SHOT / UC3M-LP TEST ===',
        f'Recortes evaluados: {len(pred_rows)}/491',
        f'EM parcial estricto: {metrics["partial_EM_percent"]:.2f}% '
        f'({metrics["partial_exact_matches"]}/{len(pred_rows)})',
        'CER proxy con mascara (NO CER completo): '
        f'{metrics["masked_error_rate_proxy_percent"]:.2f}% '
        f'({metrics["masked_edit_errors"]}/{metrics["known_gt_characters"]} caracteres conocidos)',
        f'Predicciones de longitud correcta: {metrics["equal_length_predictions"]}/{len(pred_rows)}',
        'Tasa de error sobre posiciones conocidas (solo longitudes correctas): '
        + (f'{100*metrics["aligned_known_error_rate"]:.2f}%' if metrics['aligned_known_error_rate'] is not None else 'N/D'),
        f'Latencia GPU por imagen (batch {args.batch_size}): {total_forward_ms/len(pred_rows):.2f} ms',
        f'Tiempo de inferencia incluyendo carga/preparacion de imagenes: {wall_seconds:.2f} s',
        f'Checkpoint: {CHECKPOINT_NAME}',
        f'SHA256 checkpoint: {info["checkpoint_sha256"]}',
        f'SHA256 manifest: {manifest_sha}',
        'ATENCION: Las metricas con mascara NO son equivalentes a EM/CER completos de ELPD.',
        'RESULTADO: BASELINE PARCIAL FINALIZADO (sin fine-tuning)',
        f'Predicciones: {args.out / "predictions.csv"}',
    ]
    write_results(args.out, pred_rows, summary, report)
    print('\n'.join(report), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
