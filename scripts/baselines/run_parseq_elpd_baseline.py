#!/usr/bin/env python3
"""Zero-shot PARSeq baseline on the 80 human-audited ELPD crops.

Run on a SLURM GPU node in the OCR_Matriculas Apptainer environment.
Read-only with respect to input images/manifests. Downloads the official
PARSeq source/weights into a persistent model cache if not already present.

Outputs predictions.csv, summary.json and summary.txt under results/baselines.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_MANIFEST = PROJECT / 'results/audit/ELPD/final_eval/verified_test_manifest.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_MODEL_CACHE = DATASETS / 'models/torch_hub'
DEFAULT_OUT = PROJECT / 'results/baselines/ELPD/parseq_pretrained'
MODEL_REPO = 'baudm/parseq:main'
MODEL_ENTRY = 'parseq'
CHECKPOINT_NAME = 'parseq-bb5792a6.pt'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_plate(text: str) -> str:
    """One uniform evaluation rule for predictions and ground truth."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insertion, deletion, substitution = 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_manifest(path: Path, root: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'No existe el manifest: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_path', 'text_normalized', 'gt_review_status', 'strict_gt_eligible'}
        absent = required - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f'Faltan columnas del manifest: {sorted(absent)}')
        rows = list(reader)
    if len(rows) != 80:
        raise ValueError(f'Esperadas 80 filas auditadas; encontradas {len(rows)}')
    if Counter(r['gt_review_status'] for r in rows) != {'verified': 76, 'corrected': 4}:
        raise ValueError('Los estados de GT no son 76 verified + 4 corrected')
    found: set[str] = set()
    for i, row in enumerate(rows, start=2):
        rel = row['image_path'].strip()
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts or not rel.startswith('images/'):
            raise ValueError(f'Ruta de imagen no permitida en fila {i}: {rel}')
        if rel in found:
            raise ValueError(f'Ruta de imagen duplicada: {rel}')
        found.add(rel)
        if not (root / rel).is_file():
            raise FileNotFoundError(f'Falta imagen ELPD: {root / rel}')
        if row['strict_gt_eligible'].strip().lower() != 'yes':
            raise ValueError(f'GT no apto en fila {i}: {rel}')
        gt = row['text_normalized'].strip()
        if not gt or normalize_plate(gt) != gt:
            raise ValueError(f'GT no normalizado en fila {i}: {gt!r}')
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('--batch-size debe ser positivo')

    print('=== GVIS: PARSEQ ZERO-SHOT / ELPD ===', flush=True)
    rows = load_manifest(args.manifest, args.image_root)
    print(f'Dataset: ELPD estricto, {len(rows)} recortes (76 verified + 4 corrected)', flush=True)
    print('SHA256 manifest:', sha256_file(args.manifest), flush=True)
    print('Modelo: PARSeq preentrenado oficial (sin fine-tuning)', flush=True)
    print('Repositorio:', MODEL_REPO, flush=True)
    print('Normalizacion: mayusculas A-Z, numeros 0-9; otros caracteres eliminados', flush=True)

    missing = [m for m in ('nltk', 'yaml') if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError('Dependencias extra no disponibles: ' + ', '.join(missing) +
                           '. Instalar con uv pip install --python /opt/venv/bin/python nltk pyyaml')

    from PIL import Image
    import torch
    from torchvision import transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no visible: ejecutar mediante GVIS: Ejecutar en GPU')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), flush=True)

    # Ensure model source and checkpoint survive beyond a compute allocation.
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_MODEL_CACHE))
    torch_home = Path(os.environ['TORCH_HOME'])
    torch_home.mkdir(parents=True, exist_ok=True)
    print('Cache de modelos:', torch_home, flush=True)
    print('Cargando codigo y pesos oficiales. Primera ejecucion puede requerir descarga...', flush=True)
    started_load = time.perf_counter()
    model = torch.hub.load(MODEL_REPO, MODEL_ENTRY, pretrained=True, trust_repo=True)
    model.eval().to(device)
    load_s = time.perf_counter() - started_load
    print(f'PARSeq cargado en {load_s:.2f} s', flush=True)

    # Same non-augmented transform as SceneTextDataModule.get_transform() in the official repo.
    img_size = tuple(model.hparams.img_size)
    transform = T.Compose([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    print('Tamano de entrada (alto, ancho):', img_size, flush=True)
    print('Batch size:', args.batch_size, flush=True)

    # Warm up on a real input. Time only model+greedy decoding, not preprocessing/disk I/O.
    with Image.open(args.image_root / rows[0]['image_path']) as image:
        first = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(2):
            sample_probs = model(first).softmax(-1)
            model.tokenizer.decode(sample_probs)
    torch.cuda.synchronize()

    predictions: list[dict[str, object]] = []
    batch_latencies: list[float] = []
    total_forward_ms = 0.0
    tic = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch_size):
            chunk = rows[begin:begin + args.batch_size]
            imgs = []
            for row in chunk:
                with Image.open(args.image_root / row['image_path']) as image:
                    imgs.append(transform(image.convert('RGB')))
            batch = torch.stack(imgs).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(batch)
            probabilities = logits.softmax(-1)
            labels, confidences = model.tokenizer.decode(probabilities)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if len(labels) != len(chunk):
                raise RuntimeError('Numero inesperado de predicciones en un batch')
            batch_latencies.append(elapsed_ms)
            total_forward_ms += elapsed_ms
            for row, pred_raw, confidence in zip(chunk, labels, confidences):
                gt = row['text_normalized'].strip()
                pred = normalize_plate(pred_raw)
                dist = edit_distance(pred, gt)
                predictions.append({
                    'image_path': row['image_path'],
                    'gt_review_status': row['gt_review_status'],
                    'ground_truth': gt,
                    'prediction_raw': pred_raw,
                    'prediction_normalized': pred,
                    'exact_match': int(pred == gt),
                    'edit_distance': dist,
                    'gt_length': len(gt),
                    'pred_length': len(pred),
                    'confidence': round(float(confidence.prod().item()), 6),
                    'batch_forward_ms': round(elapsed_ms, 3),
                })
            print(f'Inferencia: {len(predictions)}/{len(rows)}', flush=True)
    wall_s = time.perf_counter() - tic

    n = len(predictions)
    if n != 80:
        raise RuntimeError(f'Inferencia incompleta: {n}/80')
    correct = sum(int(p['exact_match']) for p in predictions)
    total_errors = sum(int(p['edit_distance']) for p in predictions)
    gt_chars = sum(int(p['gt_length']) for p in predictions)
    summary = {
        'dataset': 'ELPD/Kaggle audited strict test',
        'dataset_size': n,
        'gt_verified': 76,
        'gt_corrected': 4,
        'manifest_sha256': sha256_file(args.manifest),
        'model': 'PARSeq',
        'inference_mode': 'zero-shot; pretrained; autoregressive + 1 refinement (official defaults)',
        'source': MODEL_REPO,
        'checkpoint_filename': CHECKPOINT_NAME,
        'checkpoint_sha256': None,
        'source_hubconf_sha256': None,
        'finetuning': False,
        'normalization': 'uppercase and retain ASCII A-Z0-9',
        'EM': correct / n,
        'EM_percent': 100 * correct / n,
        'CER': total_errors / gt_chars,
        'CER_percent': 100 * total_errors / gt_chars,
        'exact_matches': correct,
        'levenshtein_errors': total_errors,
        'gt_characters': gt_chars,
        'gpu_forward_ms_total': round(total_forward_ms, 3),
        'gpu_forward_ms_per_image': round(total_forward_ms / n, 3),
        'gpu_forward_batch_ms_mean': round(statistics.mean(batch_latencies), 3),
        'inference_wall_seconds_including_image_loading': round(wall_s, 3),
        'checkpoint_load_seconds': round(load_s, 3),
        'batch_size': args.batch_size,
        'image_size_hw': img_size,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'num_parameters': sum(x.numel() for x in model.parameters()),
    }
    ckpt_path = torch_home / 'hub' / 'checkpoints' / CHECKPOINT_NAME
    if ckpt_path.is_file():
        summary['checkpoint_sha256'] = sha256_file(ckpt_path)
    hubconf = torch_home / 'hub' / 'baudm_parseq_main' / 'hubconf.py'
    if hubconf.is_file():
        summary['source_hubconf_sha256'] = sha256_file(hubconf)

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(predictions[0].keys())
    with (args.out / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = [
        '=== GVIS: PARSEQ ZERO-SHOT / ELPD ===',
        f'Recortes evaluados: {n}/80',
        f'Aciertos exactos: {correct}/{n}',
        f'EM: {100*correct/n:.2f}%',
        f'CER: {100*total_errors/gt_chars:.2f}% ({total_errors}/{gt_chars} caracteres GT)',
        f'Latencia media GPU por imagen (batch {args.batch_size}): {total_forward_ms/n:.2f} ms',
        f'Tiempo inferencia incluyendo preparacion de imagenes: {wall_s:.2f} s',
        f'Checkpoint: {CHECKPOINT_NAME}',
        f'SHA256 checkpoint: {summary["checkpoint_sha256"] or "No encontrado en cache"}',
        f'SHA256 manifest: {summary["manifest_sha256"]}',
        'RESULTADO: BASELINE FINALIZADO (sin fine-tuning)',
        f'Predicciones: {args.out / "predictions.csv"}',
        f'Resumen: {args.out / "summary.json"}',
    ]
    (args.out / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
