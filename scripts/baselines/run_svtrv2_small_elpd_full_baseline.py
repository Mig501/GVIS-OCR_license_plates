#!/usr/bin/env python3
"""SVTRv2-Small zero-shot en las 735 imágenes de ELPD.

Reutiliza el mismo manifiesto + overlay de 80 GT auditados que PARSeq full.
Conserva arquitectura, checkpoint y decodificación de la baseline ELPD 80.
No altera imágenes, checkpoints ni resultados anteriores. Almacena por separado.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

# Esta dependencia ya existe para la baseline completa de PARSeq, en esta carpeta.
from run_parseq_elpd_full_baseline import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_STRICT_MANIFEST,
    EXPECTED_IMAGES,
    load_manifest,
    metric_summary,
    sha256_file,
)

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
OPENOCR = PROJECT / 'third_party/OpenOCR'
CONFIG = DATASETS / 'weights/SVTRv2/small/config.yml'
CHECKPOINT = DATASETS / 'weights/SVTRv2/small/best.pth'
OUTPUT = PROJECT / 'results/baselines/ELPD/svtrv2_small_full_735'
EXPECTED_OPENOCR_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_CHECKPOINT_SHA256 = '4c4e21c8eabd7a38519d9f10d6ffa286d470a07b1a8976109fe2b995a9414970'


def read_git_commit(repo: Path) -> str:
    """Resuelve HEAD desde .git aun sin el ejecutable git dentro de Apptainer."""
    git_dir = repo / '.git'
    if git_dir.is_file():
        pointer = git_dir.read_text(encoding='utf-8').strip()
        if not pointer.startswith('gitdir: '):
            raise ValueError(f'.git no reconocible: {git_dir}')
        git_dir = Path(pointer[len('gitdir: '):].strip())
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    if not git_dir.is_dir():
        raise FileNotFoundError(f'No existe .git en OpenOCR: {git_dir}')
    head = (git_dir / 'HEAD').read_text(encoding='utf-8').strip()
    if head.startswith('ref: '):
        ref = head[5:]
        ref_file = git_dir / ref
        if ref_file.is_file():
            commit = ref_file.read_text(encoding='utf-8').strip()
        else:
            commit = None
            packed = git_dir / 'packed-refs'
            if packed.is_file():
                for line in packed.read_text(encoding='utf-8').splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        commit = parts[0]
                        break
            if commit is None:
                raise RuntimeError(f'No se ha podido resolver HEAD: {ref}')
    else:
        commit = head
    if len(commit) != 40 or any(c not in '0123456789abcdefABCDEF' for c in commit):
        raise ValueError(f'Hash git inválido: {commit}')
    return commit.lower()


def validate_assets() -> dict:
    if not OPENOCR.is_dir():
        raise FileNotFoundError(f'Falta el repositorio OpenOCR: {OPENOCR}')
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_OPENOCR_COMMIT:
        raise ValueError(f'Commit OpenOCR distinto al original: {commit}')
    for path in (CONFIG, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f'Falta archivo del modelo: {path}')
    ckpt_sha = sha256_file(CHECKPOINT)
    if ckpt_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint distinto al original: {ckpt_sha}')
    return {'openocr_commit': commit,
            'checkpoint_sha256': ckpt_sha,
            'config_sha256': sha256_file(CONFIG)}


def load_config_and_validate(Config):
    cfg = Config(str(CONFIG)).cfg
    arch = cfg.get('Architecture', {})
    decoder = arch.get('Decoder', {})
    pp = cfg.get('PostProcess', {})
    if not (
        arch.get('algorithm') == 'BGPD'
        and arch.get('Encoder', {}).get('name') == 'SVTRv2LNConvTwo33'
        and arch.get('Encoder', {}).get('dims') == [96, 192, 384]
        and decoder.get('name') == 'GTCDecoder'
        and decoder.get('infer_gtc') is True
        and pp.get('name') == 'GTCLabelDecode'
    ):
        raise ValueError(f'Arquitectura SVTRv2-Small inesperada: {CONFIG}')
    dic = Path(cfg['Global']['character_dict_path'])
    if not dic.is_absolute():
        dic = (OPENOCR / dic).resolve()
    if not dic.is_file():
        raise FileNotFoundError(f'Diccionario oficial no encontrado: {dic}')
    cfg['Global']['character_dict_path'] = str(dic)
    cfg['PostProcess']['only_gtc'] = True  # Igual que baseline original ELPD 80
    cfg['Global']['pretrained_model'] = None
    cfg['Global']['checkpoints'] = None
    cfg['Global']['device'] = 'gpu'
    return cfg, dic


def load_checkpoint_strict(torch):
    # validated SHA256 before deserializing
    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        mode = 'weights_only=True'
    except Exception as exc:
        print(f'Checkpoint oficial no cargable con weights_only: {type(exc).__name__}', flush=True)
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        mode = 'weights_only=False (checkpoint SHA256 verificado)'
    if not isinstance(payload, dict):
        raise TypeError(f'Formato de checkpoint inesperado: {type(payload)}')
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict) or not state:
        raise TypeError('Checkpoint sin state_dict válido')
    print(f'Cargando pesos: {mode} ({len(state)} tensores)', flush=True)
    return state


def save_csv(path: Path, rows: list[dict]):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('--batch-size debe ser positivo')

    print('=== GVIS: SVTRv2-SMALL ZERO-SHOT / ELPD COMPLETO ===', flush=True)
    rows, details = load_manifest(DEFAULT_MANIFEST, DEFAULT_IMAGE_ROOT, DEFAULT_STRICT_MANIFEST)
    print(f'Dataset: ELPD completo, {len(rows)} imágenes (etiquetas candidatas + GT auditados)', flush=True)
    print('Splits originales:', details['original_split_counts'], flush=True)
    print('Origen de GT:', details['gt_source_counts'], flush=True)
    print('GT auditados incorporados:', details['audited_rows_merged'], flush=True)
    print('GT corregidos por auditoría:', details['candidate_gt_changed_by_audit'], flush=True)
    print('SHA256 manifest completo:', sha256_file(DEFAULT_MANIFEST), flush=True)
    print('SHA256 manifest estricto:', details['audited_manifest_sha256'], flush=True)
    assets = validate_assets()
    print('OpenOCR commit:', assets['openocr_commit'], flush=True)
    print('Checkpoint SHA256:', assets['checkpoint_sha256'], flush=True)
    print('Config SHA256:', assets['config_sha256'], flush=True)

    output_names = ('predictions.csv', 'summary.json', 'summary.txt')
    if any((OUTPUT / file).exists() for file in output_names):
        raise FileExistsError(f'Ya existen resultados en {OUTPUT}; no se sobrescriben')

    sys.path.insert(0, str(PROJECT))
    from src.evaluation.ocr_metrics import score_complete
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: ejecutar GVIS: Ejecutar en GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.infer_rec import OpenRecognizer
    from tools.engine.config import Config

    cfg, dictionary = load_config_and_validate(Config)
    print('Diccionario oficial:', dictionary, flush=True)
    print('Decodificación: GTCDecoder(infer_gtc=True) + GTCLabelDecode(only_gtc=True)', flush=True)
    print('GPU:', torch.cuda.get_device_name(0), '| batch-size:', args.batch_size, flush=True)
    load_start = time.perf_counter()
    recognizer = OpenRecognizer(config=cfg, mode='server', backend='torch', use_gpu='true')
    state = load_checkpoint_strict(torch)
    recognizer.model.load_state_dict(state, strict=True)
    del state
    recognizer.model.eval()
    load_s = time.perf_counter() - load_start
    num_params = sum(p.numel() for p in recognizer.model.parameters())
    if num_params != 12518159:
        raise RuntimeError(f'Número de parámetros SVTRv2-Small inesperado: {num_params}')
    print('Pesos cargados STRICT=True | parámetros:', num_params, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    forward_ms = []
    with tempfile.TemporaryDirectory(prefix='stage_svtrv2_small_', dir=OUTPUT) as tmp:
        stage = Path(tmp)
        all_dir, warm_dir = stage / 'all', stage / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        for i, row in enumerate(rows):
            source = DEFAULT_IMAGE_ROOT / row['image_path']
            name = f'{i:04d}{source.suffix.lower()}'
            (all_dir / name).symlink_to(source)
            if i < args.batch_size:
                (warm_dir / name).symlink_to(source)
        with torch.inference_mode():
            for _ in range(2):
                result = recognizer(img_path=str(warm_dir), batch_num=args.batch_size)
                if len(result) != min(len(rows), args.batch_size):
                    raise RuntimeError(f'Número de predicciones inesperado en warm-up: {len(result)}')
            torch.cuda.synchronize()

        original_forward = recognizer.model.forward
        def forward_timed(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = original_forward(*a, **kw)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - t0) * 1000)
            return result
        recognizer.model.forward = forward_timed
        print(f'Inferencia: {len(rows)} imágenes, batch={args.batch_size}', flush=True)
        begin = time.perf_counter()
        try:
            with torch.inference_mode():
                results = recognizer(img_path=str(all_dir), batch_num=args.batch_size)
                torch.cuda.synchronize()
        finally:
            recognizer.model.forward = original_forward
        wall_s = time.perf_counter() - begin

    if len(results) != EXPECTED_IMAGES:
        raise RuntimeError(f'Inferencia incompleta: {len(results)}/{EXPECTED_IMAGES}')
    if not forward_ms:
        raise RuntimeError('No se registraron llamadas GPU-forward')

    predictions = []
    groups_split = defaultdict(list)
    groups_source = defaultdict(list)
    for row, result in zip(rows, results):
        raw = str(result['text'])
        gt = row['text_normalized'].strip()
        scored = score_complete(gt, raw)
        entry = {
            'image_path': row['image_path'],
            'scenario': row['scenario'],
            'gt_source': row['gt_source'],
            'gt_candidate_original': row['gt_candidate_original'],
            'ground_truth': gt,
            'prediction_raw': raw,
            'prediction_normalized': scored['prediction_normalized'],
            'exact_match': scored['exact_match'],
            'edit_distance': scored['edit_distance'],
            'gt_length': scored['gt_length'],
            'pred_length': scored['pred_length'],
            'confidence': round(float(result['score']), 6),
        }
        predictions.append(entry)
        groups_split[str(row['scenario'])].append(entry)
        groups_source[str(row['gt_source'])].append(entry)

    whole = metric_summary(predictions)
    summary = {
        'dataset': 'ELPD/Kaggle full dataset (candidate labels; audited labels overlaid)',
        'dataset_size': len(rows),
        'label_warning': 'Most labels are candidates from filenames, not visually verified; dataset includes original train/val/test.',
        'manifest_sha256': sha256_file(DEFAULT_MANIFEST),
        'gt_audit': details,
        'metrics_by_original_split': {k: metric_summary(v) for k, v in sorted(groups_split.items())},
        'metrics_by_gt_source': {k: metric_summary(v) for k, v in sorted(groups_source.items())},
        'model': 'SVTRv2-Small',
        'inference_mode': 'zero-shot; pretrained; BGPD + GTCDecoder(infer_gtc=True); GTCLabelDecode(only_gtc=True)',
        'source': 'Topdu/OpenOCR',
        'source_commit': assets['openocr_commit'],
        'checkpoint_filename': CHECKPOINT.name,
        'checkpoint_sha256': assets['checkpoint_sha256'],
        'config_sha256': assets['config_sha256'],
        'finetuning': False,
        'normalization': 'uppercase and retain ASCII A-Z0-9',
        'EM': whole['exact_matches'] / whole['images'],
        'EM_percent': whole['EM_percent'],
        'CER': whole['levenshtein_errors'] / whole['gt_characters'],
        'CER_percent': whole['CER_percent'],
        'exact_matches': whole['exact_matches'],
        'levenshtein_errors': whole['levenshtein_errors'],
        'gt_characters': whole['gt_characters'],
        'gpu_forward_ms_total': round(sum(forward_ms), 3),
        'gpu_forward_ms_per_image': round(sum(forward_ms) / len(rows), 3),
        'gpu_forward_batch_ms_mean': round(statistics.mean(forward_ms), 3),
        'inference_wall_seconds_including_image_loading': round(wall_s, 3),
        'checkpoint_load_seconds': round(load_s, 3),
        'batch_size': args.batch_size,
        'number_forward_batches': len(forward_ms),
        'num_parameters': num_params,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'latency_note': 'GPU-forward only; excludes image decoding and postprocess',
    }

    save_csv(OUTPUT / 'predictions.csv', predictions)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = [
        '=== GVIS: SVTRv2-SMALL ZERO-SHOT / ELPD COMPLETO ===',
        f'Recortes evaluados: {len(rows)}/{EXPECTED_IMAGES}',
        f'Aciertos exactos (GT disponible): {whole["exact_matches"]}/{len(rows)}',
        f'EM (GT disponible): {whole["EM_percent"]:.2f}%',
        f'CER (GT disponible): {whole["CER_percent"]:.2f}% ({whole["levenshtein_errors"]}/{whole["gt_characters"]} caracteres GT)',
        f'GT auditados incorporados: {details["audited_rows_merged"]}',
        f'Latencia GPU-forward por imagen (batch {args.batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms',
        f'Checkpoint SHA256: {assets["checkpoint_sha256"]}',
        f'Manifest SHA256: {summary["manifest_sha256"]}',
        'AVISO: la mayoría de etiquetas son candidatas y no están verificadas visualmente.',
        'RESULTADO: BASELINE COMPLETA FINALIZADA (sin fine-tuning)',
        f'Predicciones: {OUTPUT / "predictions.csv"}',
        f'Resumen: {OUTPUT / "summary.json"}',
    ]
    (OUTPUT / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
