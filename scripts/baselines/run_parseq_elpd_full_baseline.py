#!/usr/bin/env python3
"""PARSeq zero-shot on all 735 ELPD images (candidate GT, audited GT where available).

Inputs are read-only. Unlike the original 80-image strict-test baseline, this is
an expanded descriptive evaluation: most GT labels come from filenames and have
not been visually audited. The audited 80-image manifest is merged by image_path
by default to preserve the previous manual corrections. Strict results are never
rewritten; outputs go to a separate directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
DEFAULT_MANIFEST = PROJECT / 'results/audit/ELPD/eval_manifests/external_eval_manifest.csv'
DEFAULT_STRICT_MANIFEST = PROJECT / 'results/audit/ELPD/final_eval/verified_test_manifest.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_MODEL_CACHE = DATASETS / 'models/torch_hub'
DEFAULT_OUT = PROJECT / 'results/baselines/ELPD/parseq_pretrained_full_735'
MODEL_REPO = 'baudm/parseq:main'
MODEL_ENTRY = 'parseq'
CHECKPOINT_NAME = 'parseq-bb5792a6.pt'
EXPECTED_IMAGES = 735


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_plate(text: str) -> str:
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


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'No existe el manifest: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Faltan columnas en {path}: {sorted(missing)}')
        return list(reader)


def validate_image_and_gt(row: dict[str, str], root: Path, index: int) -> None:
    rel = row['image_path'].strip()
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or not rel.startswith('images/'):
        raise ValueError(f'Ruta no permitida en fila {index}: {rel}')
    if not (root / rel).is_file():
        raise FileNotFoundError(f'Falta imagen ELPD: {root / rel}')
    gt = row['text_normalized'].strip()
    if not gt or normalize_plate(gt) != gt:
        raise ValueError(f'GT no normalizado en fila {index}: {gt!r}')


def load_manifest(path: Path, root: Path, audited_manifest: Path | None):
    """Load all candidate rows; safely overlay audited labels by image_path."""
    rows = read_csv(
        path,
        {'image_path', 'text_normalized', 'scenario', 'label_review_status', 'strict_gt_eligible'},
    )
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f'Se esperan {EXPECTED_IMAGES} filas; encontradas {len(rows)}')
    by_image: dict[str, dict[str, str]] = {}
    for i, row in enumerate(rows, start=2):
        validate_image_and_gt(row, root, i)
        rel = row['image_path'].strip()
        if rel in by_image:
            raise ValueError(f'Ruta de imagen duplicada: {rel}')
        if row['scenario'].strip() not in {'train', 'val', 'test'}:
            raise ValueError(f'Split original desconocido en fila {i}: {row["scenario"]}')
        row['gt_candidate_original'] = row['text_normalized'].strip()
        row['gt_source'] = (
            'external_verified' if row['label_review_status'].strip() == 'verified'
            else 'filename_candidate'
        )
        by_image[rel] = row

    audited_count = 0
    changed_count = 0
    audited_sha = None
    if audited_manifest is not None:
        audited = read_csv(
            audited_manifest,
            {'image_path', 'text_normalized', 'gt_review_status', 'strict_gt_eligible'},
        )
        if len(audited) != 80:
            raise ValueError(f'El test estricto debe contener 80 filas; contiene {len(audited)}')
        statuses = Counter(r['gt_review_status'].strip() for r in audited)
        if statuses != Counter({'verified': 76, 'corrected': 4}):
            raise ValueError(f'Estados inesperados en test estricto: {dict(statuses)}')
        seen_audited: set[str] = set()
        for a in audited:
            rel = a['image_path'].strip()
            if rel in seen_audited:
                raise ValueError(f'Imagen repetida en test estricto: {rel}')
            seen_audited.add(rel)
            if rel not in by_image:
                raise ValueError(f'Imagen auditada no presente en manifiesto completo: {rel}')
            if a['strict_gt_eligible'].strip().lower() != 'yes':
                raise ValueError(f'GT estricto no apto: {rel}')
            gt = a['text_normalized'].strip()
            if not gt or normalize_plate(gt) != gt:
                raise ValueError(f'GT estricto inválido: {rel}: {gt!r}')
            dest = by_image[rel]
            if dest['text_normalized'].strip() != gt:
                changed_count += 1
            dest['text_normalized'] = gt
            dest['gt_source'] = 'strict_' + a['gt_review_status'].strip()
            audited_count += 1
        audited_sha = sha256_file(audited_manifest)

    details = {
        'dataset_size': len(rows),
        'original_split_counts': dict(Counter(r['scenario'] for r in rows)),
        'gt_source_counts': dict(Counter(r['gt_source'] for r in rows)),
        'audited_manifest_path': str(audited_manifest) if audited_manifest is not None else None,
        'audited_manifest_sha256': audited_sha,
        'audited_rows_merged': audited_count,
        'candidate_gt_changed_by_audit': changed_count,
    }
    return rows, details


def metric_summary(predictions: list[dict[str, object]]) -> dict[str, int | float | None]:
    n = len(predictions)
    if not n:
        return {'images': 0, 'exact_matches': 0, 'EM_percent': None, 'CER_percent': None}
    exact = sum(int(p['exact_match']) for p in predictions)
    edits = sum(int(p['edit_distance']) for p in predictions)
    gt_chars = sum(int(p['gt_length']) for p in predictions)
    return {
        'images': n,
        'exact_matches': exact,
        'EM_percent': round(100 * exact / n, 4),
        'levenshtein_errors': edits,
        'gt_characters': gt_chars,
        'CER_percent': round(100 * edits / gt_chars, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--verified-manifest', type=Path, default=DEFAULT_STRICT_MANIFEST,
                    help='Manifest del test estricto de 80 imágenes para incorporar sus GT auditados')
    ap.add_argument('--without-verified-merge', action='store_true',
                    help='No usar la auditoría previa (no recomendado)')
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--check-manifest-only', action='store_true',
                    help='Validar los 735 registros y la fusión de GT sin cargar GPU')
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('--batch-size debe ser positivo')
    audited_path = None if args.without_verified_merge else args.verified_manifest

    print('=== GVIS: PARSEQ ZERO-SHOT / ELPD COMPLETO ===', flush=True)
    rows, details = load_manifest(args.manifest, args.image_root, audited_path)
    print(f'Dataset: ELPD completo, {len(rows)} imágenes, etiquetas candidatas', flush=True)
    print('Splits originales:', details['original_split_counts'], flush=True)
    print('Origen de GT:', details['gt_source_counts'], flush=True)
    print('GT auditados incorporados:', details['audited_rows_merged'], flush=True)
    print('Etiquetas cambiadas tras la auditoría:', details['candidate_gt_changed_by_audit'], flush=True)
    print('SHA256 manifiesto completo:', sha256_file(args.manifest), flush=True)
    if details['audited_manifest_sha256']:
        print('SHA256 manifiesto estricto:', details['audited_manifest_sha256'], flush=True)
    if args.check_manifest_only:
        print('RESULTADO: MANIFIESTOS VALIDADOS, SIN INFERENCIA', flush=True)
        return 0

    output_files = ('predictions.csv', 'summary.json', 'summary.txt')
    existing = [name for name in output_files if (args.out / name).exists()]
    if existing:
        raise FileExistsError(
            f'Para preservar una ejecución previa, no se sobrescriben estos archivos en {args.out}: {existing}'
        )
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

    os.environ.setdefault('TORCH_HOME', str(DEFAULT_MODEL_CACHE))
    torch_home = Path(os.environ['TORCH_HOME'])
    torch_home.mkdir(parents=True, exist_ok=True)
    print('Cache de modelos:', torch_home, flush=True)
    started_load = time.perf_counter()
    model = torch.hub.load(MODEL_REPO, MODEL_ENTRY, pretrained=True, trust_repo=True)
    model.eval().to(device)
    load_s = time.perf_counter() - started_load
    print(f'PARSeq cargado en {load_s:.2f} s', flush=True)

    # Same non-augmented transform as the official SceneTextDataModule.
    img_size = tuple(model.hparams.img_size)
    transform = T.Compose([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    print('Tamano de entrada (alto, ancho):', img_size, flush=True)
    print('Batch size:', args.batch_size, flush=True)

    # Warm up and time only GPU model+greedy decoding, not image loading or preprocessing.
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
                    'scenario': row['scenario'],
                    'gt_source': row['gt_source'],
                    'gt_candidate_original': row['gt_candidate_original'],
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
    if len(predictions) != EXPECTED_IMAGES:
        raise RuntimeError(f'Inferencia incompleta: {len(predictions)}/{EXPECTED_IMAGES}')

    whole = metric_summary(predictions)
    group_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    group_by_gt_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for p in predictions:
        group_by_split[str(p['scenario'])].append(p)
        group_by_gt_source[str(p['gt_source'])].append(p)

    summary = {
        'dataset': 'ELPD/Kaggle full dataset (candidate labels; audited labels overlaid)',
        'dataset_size': len(predictions),
        'label_warning': 'Most labels are candidates from filenames, not visually verified. Entire dataset is not a held-out test.',
        'manifest_sha256': sha256_file(args.manifest),
        'gt_audit': details,
        'metrics_by_original_split': {k: metric_summary(v) for k, v in sorted(group_by_split.items())},
        'metrics_by_gt_source': {k: metric_summary(v) for k, v in sorted(group_by_gt_source.items())},
        'model': 'PARSeq',
        'inference_mode': 'zero-shot; pretrained; autoregressive + 1 refinement (official defaults)',
        'source': MODEL_REPO,
        'checkpoint_filename': CHECKPOINT_NAME,
        'checkpoint_sha256': None,
        'source_hubconf_sha256': None,
        'finetuning': False,
        'normalization': 'uppercase and retain ASCII A-Z0-9',
        'EM': whole['exact_matches'] / whole['images'],
        'EM_percent': whole['EM_percent'],
        'CER': whole['levenshtein_errors'] / whole['gt_characters'],
        'CER_percent': whole['CER_percent'],
        'exact_matches': whole['exact_matches'],
        'levenshtein_errors': whole['levenshtein_errors'],
        'gt_characters': whole['gt_characters'],
        'gpu_forward_ms_total': round(total_forward_ms, 3),
        'gpu_forward_ms_per_image': round(total_forward_ms / len(predictions), 3),
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
        '=== GVIS: PARSEQ ZERO-SHOT / ELPD COMPLETO ===',
        f'Recortes evaluados: {len(predictions)}/{EXPECTED_IMAGES}',
        f'Aciertos exactos (GT disponible): {whole["exact_matches"]}/{len(predictions)}',
        f'EM (GT disponible): {whole["EM_percent"]:.2f}%',
        f'CER (GT disponible): {whole["CER_percent"]:.2f}% ({whole["levenshtein_errors"]}/{whole["gt_characters"]} caracteres GT)',
        f'GT auditados fusionados: {details["audited_rows_merged"]}',
        f'GT candidatos modificados por auditoría: {details["candidate_gt_changed_by_audit"]}',
        'AVISO: gran parte de las etiquetas procede del nombre de archivo y no esta validada visualmente.',
        f'Latencia media GPU por imagen (batch {args.batch_size}): {total_forward_ms / len(predictions):.2f} ms',
        f'Tiempo inferencia incluyendo preparacion de imagenes: {wall_s:.2f} s',
        f'Checkpoint: {CHECKPOINT_NAME}',
        f'SHA256 checkpoint: {summary["checkpoint_sha256"] or "No encontrado en cache"}',
        f'SHA256 manifest: {summary["manifest_sha256"]}',
        'RESULTADO: BASELINE COMPLETA FINALIZADA (sin fine-tuning)',
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
