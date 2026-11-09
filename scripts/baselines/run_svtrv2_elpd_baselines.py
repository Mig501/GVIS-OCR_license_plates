#!/usr/bin/env python3
"""Zero-shot SVTRv2 Tiny + Small baselines over the same 80 audited ELPD images.

Requires OpenOCR pinned checkout, official checkpoints + their matching config.yml,
common src/evaluation/ocr_metrics.py, and the existing OCR_Matriculas CUDA env.
Read-only on original datasets and checkpoints. Saves 3 output files per model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
OPENOCR = PROJECT / 'third_party/OpenOCR'
WEIGHTS = DATASETS / 'weights/SVTRv2'
MANIFEST = PROJECT / 'results/audit/ELPD/final_eval/verified_test_manifest.csv'
IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
OUTPUT = PROJECT / 'results/baselines/ELPD'
EXPECTED_MANIFEST_SHA = 'c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c'
EXPECTED_REPO_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_CHECKPOINT_SHA = {
    'tiny': '684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6',
    'small': '4c4e21c8eabd7a38519d9f10d6ffa286d470a07b1a8976109fe2b995a9414970',
}


def read_git_commit(repo: Path) -> str:
    """Read a checkout's HEAD without requiring the git executable (Apptainer)."""
    git_dir = repo / '.git'
    if git_dir.is_file():  # git worktree: .git holds 'gitdir: ...'
        pointer = git_dir.read_text(encoding='utf-8').strip()
        if not pointer.startswith('gitdir: '):
            raise RuntimeError(f'Invalid Git metadata: {git_dir}')
        git_dir = Path(pointer.removeprefix('gitdir: ').strip())
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    if not git_dir.is_dir():
        raise FileNotFoundError(f'Missing .git metadata: {git_dir}')
    head = (git_dir / 'HEAD').read_text(encoding='utf-8').strip()
    if head.startswith('ref: '):
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            commit = ref_path.read_text(encoding='utf-8').strip()
        else:
            packed_refs = git_dir / 'packed-refs'
            commit = None
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding='utf-8').splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        commit = parts[0]
                        break
            if commit is None:
                raise RuntimeError(f'Cannot resolve Git reference {ref!r} in {git_dir}')
    else:
        commit = head  # detached HEAD
    if len(commit) != 40 or any(c not in '0123456789abcdefABCDEF' for c in commit):
        raise RuntimeError(f'Invalid checkout commit in {git_dir}: {commit!r}')
    return commit.lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f'No existe el manifest ELPD: {MANIFEST}')
    digest = sha256_file(MANIFEST)
    if digest != EXPECTED_MANIFEST_SHA:
        raise ValueError(f'Manifest diferente al de PARSeq: {digest}')
    with MANIFEST.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        required = {'image_path', 'text_normalized', 'gt_review_status', 'strict_gt_eligible'}
        if not required.issubset(set(rd.fieldnames or [])):
            raise ValueError(f'Faltan columnas: {required - set(rd.fieldnames or [])}')
        rows = list(rd)
    if len(rows) != 80 or Counter(r['gt_review_status'] for r in rows) != {'verified': 76, 'corrected': 4}:
        raise ValueError('ELPD: no se encontraron 80 GT (76 verified, 4 corrected)')
    used = set()
    for row in rows:
        path = row['image_path'].strip()
        p = Path(path)
        if p.is_absolute() or '..' in p.parts or not path.startswith('images/'):
            raise ValueError(f'Ruta ELPD inválida: {path}')
        if path in used:
            raise ValueError(f'Ruta duplicada: {path}')
        used.add(path)
        if not (IMAGE_ROOT / path).is_file():
            raise FileNotFoundError(f'Falta imagen: {IMAGE_ROOT / path}')
        if row['strict_gt_eligible'].strip().lower() != 'yes':
            raise ValueError(f'GT no apto: {path}')
        gt = row['text_normalized'].strip()
        if not gt or not gt.isascii() or not gt.isalnum() or gt != gt.upper():
            raise ValueError(f'GT sin normalizar: {gt!r}')
    return rows


def load_official_checkpoint(torch, path: Path, expected_sha: str) -> dict:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f'SHA256 del checkpoint no coincide: {path} / {actual}')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        method = 'weights_only=True'
    except Exception as exc:
        # Only allow full PyTorch serialization if the bytes match the audited
        # exact official Google Drive file, not an arbitrary local checkpoint.
        print(f'Checkpoint oficial no cargable como weights_only: {type(exc).__name__}', flush=True)
        print('Usando carga pickle exclusivamente porque SHA256 coincide con el checkpoint fijado.', flush=True)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        method = 'weights_only=False (SHA256 pinned)'
    if not isinstance(payload, dict):
        raise TypeError(f'Checkpoint inesperado: {type(payload).__name__}')
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict) or not state:
        raise TypeError('Checkpoint sin state_dict tensorial')
    print('Carga checkpoint:', method, '| tensores:', len(state), flush=True)
    return state


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('No hay predicciones para guardar')
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def run_variant(variant: str, rows: list[dict[str, str]], batch_size: int, torch,
                OpenRecognizer, Config, score_complete, summarize_complete):
    cfg_path = WEIGHTS / variant / 'config.yml'
    weights_path = WEIGHTS / variant / 'best.pth'
    if not cfg_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f'Falta config.yml o best.pth de {variant}')
    cfg = Config(str(cfg_path)).cfg
    arch = cfg.get('Architecture', {})
    decoder = arch.get('Decoder', {})
    pp = cfg.get('PostProcess', {})
    if (arch.get('algorithm') != 'BGPD'
        or arch.get('Encoder', {}).get('name') != 'SVTRv2LNConvTwo33'
        or decoder.get('name') != 'GTCDecoder'
        or decoder.get('infer_gtc') is not True
        or pp.get('name') != 'GTCLabelDecode'):
        raise ValueError(f'Arquitectura oficial inesperada en {cfg_path}')
    expected_dims = [64, 128, 256] if variant == 'tiny' else [96, 192, 384]
    if arch['Encoder'].get('dims') != expected_dims:
        raise ValueError(f'Encoder dims inesperadas: {arch["Encoder"].get("dims")}')

    # Config has relative character dictionary path. It must resolve within
    # the exact OpenOCR checkout, never relative to the user's shell CWD.
    dic_path = cfg['Global']['character_dict_path']
    if not Path(dic_path).is_absolute():
        dic_path = str((OPENOCR / dic_path).resolve())
    if not Path(dic_path).is_file():
        raise FileNotFoundError(f'Falta diccionario oficial: {dic_path}')
    cfg['Global']['character_dict_path'] = dic_path

    # OpenOCR's GTCLabelDecode returns [GTC, CTC] when only_gtc=False.
    # The OpenRecognizer wrapper requires one result per image, so use only
    # the *official* GTC decoding branch, without changing model weights.
    cfg['PostProcess']['only_gtc'] = True
    cfg['Global']['pretrained_model'] = None
    cfg['Global']['checkpoints'] = None
    cfg['Global']['device'] = 'gpu'

    print(f'\n=== SVTRv2-{variant.upper()} / ELPD ZERO-SHOT ===', flush=True)
    print('Config oficial:', cfg_path, flush=True)
    print('SHA256 config:', sha256_file(cfg_path), flush=True)
    print('Decoder: GTC infer_gtc=True; salida: solo GTC', flush=True)
    print('Diccionario:', dic_path, flush=True)
    print('Construyendo arquitectura...', flush=True)
    build_t0 = time.perf_counter()
    recognizer = OpenRecognizer(config=cfg, mode='server', backend='torch', use_gpu='true')
    state = load_official_checkpoint(torch, weights_path, EXPECTED_CHECKPOINT_SHA[variant])
    recognizer.model.load_state_dict(state, strict=True)
    del state
    recognizer.model.eval()
    load_s = time.perf_counter() - build_t0
    num_params = sum(p.numel() for p in recognizer.model.parameters())
    print(f'Checkpoint cargado STRICT=True; parámetros: {num_params:,}; {load_s:.2f}s', flush=True)

    out_dir = OUTPUT / f'svtrv2_{variant}'
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'elpd_svtrv2_{variant}_', dir=out_dir) as d:
        stage = Path(d)
        all_dir = stage / 'all'
        warm_dir = stage / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        for i, row in enumerate(rows):
            real = IMAGE_ROOT / row['image_path']
            name = f'{i:04d}{real.suffix.lower()}'
            (all_dir / name).symlink_to(real)
            if i < batch_size:
                (warm_dir / name).symlink_to(real)
        print('Calentamiento GPU (batch):', batch_size, flush=True)
        with torch.inference_mode():
            for _ in range(2):
                warm = recognizer(img_path=str(warm_dir), batch_num=batch_size)
                if len(warm) != min(len(rows), batch_size):
                    raise RuntimeError(f'Número incorrecto de salidas en calentamiento: {len(warm)}')
            torch.cuda.synchronize()

        # Capture actual GPU forward time with explicit CUDA synchronization;
        # the upstream 'elapse' field does not synchronize CUDA and is not a
        # reliable GPU-forward measurement.
        forward_ms = []
        original_forward = recognizer.model.forward
        def forward_timed(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - t0) * 1000)
            return result
        recognizer.model.forward = forward_timed
        print(f'Inferencia: {len(rows)} imágenes, batch={batch_size}', flush=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            results = recognizer(img_path=str(all_dir), batch_num=batch_size)
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        recognizer.model.forward = original_forward

    if len(results) != len(rows):
        raise RuntimeError(f'Inferencia incompleta: {len(results)}/{len(rows)}')
    pred_rows = []
    pairs = []
    for row, result in zip(rows, results):
        raw = str(result['text'])
        gt = row['text_normalized'].strip()
        scored = score_complete(gt, raw)
        pairs.append((gt, raw))
        pred_rows.append({
            'image_path': row['image_path'],
            'gt_review_status': row['gt_review_status'],
            'ground_truth': gt,
            'prediction_raw': raw,
            'prediction_normalized': scored['prediction_normalized'],
            'exact_match': scored['exact_match'],
            'edit_distance': scored['edit_distance'],
            'gt_length': scored['gt_length'],
            'pred_length': scored['pred_length'],
            'confidence': round(float(result['score']), 6),
        })
    metrics = summarize_complete(pairs)
    summary = {
        'dataset': 'ELPD/Kaggle audited strict test',
        'dataset_size': len(rows),
        'gt_verified': 76,
        'gt_corrected': 4,
        'manifest_sha256': EXPECTED_MANIFEST_SHA,
        'model': f'SVTRv2-{variant.capitalize()}',
        'inference_mode': 'zero-shot; pretrained; BGPD + GTCDecoder(infer_gtc=True); GTCLabelDecode',
        'source': 'Topdu/OpenOCR',
        'source_commit': EXPECTED_REPO_COMMIT,
        'checkpoint_filename': 'best.pth',
        'checkpoint_sha256': EXPECTED_CHECKPOINT_SHA[variant],
        'config_sha256': sha256_file(cfg_path),
        'finetuning': False,
        'normalization': 'uppercase and retain ASCII A-Z0-9',
        **metrics,
        'batch_size': batch_size,
        'number_forward_batches': len(forward_ms),
        'gpu_forward_ms_total': round(sum(forward_ms), 3),
        'gpu_forward_ms_per_image': round(sum(forward_ms)/len(rows), 3),
        'gpu_forward_batch_ms_mean': round(sum(forward_ms)/len(forward_ms), 3) if forward_ms else None,
        'inference_wall_seconds_including_image_loading': round(wall_s, 3),
        'checkpoint_load_seconds': round(load_s, 3),
        'num_parameters': num_params,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'latency_note': 'GPU-forward only; includes model inference but not postprocess; compare under matched protocol',
    }
    write_csv(out_dir / 'predictions.csv', pred_rows)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = (
        f'=== GVIS: SVTRv2-{variant.upper()} ZERO-SHOT / ELPD ===\n'
        f'Recortes evaluados: {len(rows)}/80\n'
        f'Aciertos exactos: {metrics["exact_matches"]}/80\n'
        f'EM: {metrics["EM_percent"]:.2f}%\n'
        f'CER: {metrics["CER_percent"]:.2f}% ({metrics["levenshtein_errors"]}/{metrics["gt_characters"]} caracteres GT)\n'
        f'Latencia GPU-forward por imagen (batch {batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms\n'
        f'Wall inferencia incluyendo imágenes: {wall_s:.2f} s\n'
        f'Parámetros: {num_params}\n'
        f'Checkpoint SHA256: {EXPECTED_CHECKPOINT_SHA[variant]}\n'
        f'Manifest SHA256: {EXPECTED_MANIFEST_SHA}\n'
        f'Código OpenOCR commit: {EXPECTED_REPO_COMMIT}\n'
        f'GTCDecoder infer_gtc=True; normalización A-Z0-9; sin fine-tuning\n'
        f'ESTADO: BASELINE FINALIZADO\n'
    )
    (out_dir / 'summary.txt').write_text(report, encoding='utf-8')
    print(report, flush=True)
    print('Resultados:', out_dir, flush=True)
    del recognizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['tiny', 'small', 'both'], default='both')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size debe ser positivo')
    if not OPENOCR.is_dir():
        raise FileNotFoundError(f'OpenOCR no clonado en {OPENOCR}')
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_REPO_COMMIT:
        raise RuntimeError(f'Commit de OpenOCR no coincide: {commit}')
    rows = load_manifest()
    sys.path.insert(0, str(PROJECT))
    from src.evaluation.ocr_metrics import score_complete, summarize_complete
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: utilizar GVIS: Ejecutar en GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    try:
        from tools.infer_rec import OpenRecognizer
        from tools.engine.config import Config
    except Exception as exc:
        raise RuntimeError('Error importando OpenOCR. Verificar dependencias (p. ej. opencv-python-headless, PyYAML).') from exc
    print('=== BASELINES SVTRv2 / ELPD ===', flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('OpenOCR commit:', commit, flush=True)
    print('80 GT ELPD, SHA256:', EXPECTED_MANIFEST_SHA, flush=True)
    variants = ('tiny','small') if args.variant=='both' else (args.variant,)
    for variant in variants:
        run_variant(variant, rows, args.batch_size, torch, OpenRecognizer, Config, score_complete, summarize_complete)
    print('=== SVTRv2: TODOS LOS BASELINES FINALIZADOS ===', flush=True)


if __name__ == '__main__':
    main()
