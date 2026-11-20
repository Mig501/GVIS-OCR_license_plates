#!/usr/bin/env python3
"""MDiff4STR-Small zero-shot sobre las 735 imágenes de ELPD.

Mantiene los pesos oficiales y los parámetros del decoder de la baseline ELPD
estricta de 80 imágenes. Reutiliza el cargador del manifiesto completo de PARSeq
para aplicar las mismas 80 etiquetas auditadas (incluidas cuatro correcciones).

La mayoría de etiquetas siguen siendo candidatas, derivadas de filenames.
No modifica los datos, los checkpoints ni las salidas de baselines anteriores.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

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
WEIGHTS = DATASETS / 'weights/MDiff4STR/small'
CONFIG = WEIGHTS / 'config.yml'
CHECKPOINT = WEIGHTS / 'best.pth'
OUTPUT = PROJECT / 'results/baselines/ELPD/mdiff4str_small_full_735'
EXPECTED_OPENOCR_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_CHECKPOINT_SHA256 = 'ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937'
EXPECTED_FULL_MANIFEST_SHA256 = '5854abd860c89a17e35ef225627b17725c60d23088197f280e34eb5272097f13'
EXPECTED_STRICT_MANIFEST_SHA256 = 'c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c'
MODEL_SOURCE = 'https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_small'


def read_git_commit(repo: Path) -> str:
    """Resuelve HEAD sin necesitar el ejecutable git dentro del contenedor."""
    git_dir = repo / '.git'
    if git_dir.is_file():
        pointer = git_dir.read_text(encoding='utf-8').strip()
        if not pointer.startswith('gitdir: '):
            raise RuntimeError(f'.git no reconocible: {git_dir}')
        git_dir = Path(pointer[len('gitdir: '):].strip())
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    if not git_dir.is_dir():
        raise FileNotFoundError(f'Falta .git del repositorio OpenOCR: {git_dir}')
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
                raise RuntimeError(f'No se pudo resolver HEAD: {ref}')
    else:
        commit = head
    if len(commit) != 40 or any(c not in '0123456789abcdefABCDEF' for c in commit):
        raise RuntimeError(f'Commit no válido: {commit!r}')
    return commit.lower()


def load_rows_and_details():
    """Usa la misma validación y superposición de GT auditados que PARSeq."""
    actual_full_sha = sha256_file(DEFAULT_MANIFEST)
    actual_strict_sha = sha256_file(DEFAULT_STRICT_MANIFEST)
    if actual_full_sha != EXPECTED_FULL_MANIFEST_SHA256:
        raise ValueError(f'Manifest completo distinto del preflight 92453: {actual_full_sha}')
    if actual_strict_sha != EXPECTED_STRICT_MANIFEST_SHA256:
        raise ValueError(f'Manifest estricto distinto de la baseline original: {actual_strict_sha}')
    rows, details = load_manifest(DEFAULT_MANIFEST, DEFAULT_IMAGE_ROOT, DEFAULT_STRICT_MANIFEST)
    if len(rows) != EXPECTED_IMAGES or details['audited_rows_merged'] != 80:
        raise ValueError('No hay 735 imágenes con 80 etiquetas auditadas superpuestas')
    if details['candidate_gt_changed_by_audit'] != 4:
        raise ValueError('No se han aplicado exactamente las cuatro correcciones auditadas')
    return rows, details


def validate_assets() -> dict[str, str]:
    """Preflight CPU: fija commit, pesos, configuración y opciones del decoder."""
    import yaml

    if not OPENOCR.is_dir():
        raise FileNotFoundError(f'OpenOCR no está disponible: {OPENOCR}')
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_OPENOCR_COMMIT:
        raise ValueError(f'Commit OpenOCR inesperado: {commit}')
    decoder_source = OPENOCR / 'openrec/modeling/decoders/mdiff_decoder.py'
    if not decoder_source.is_file():
        raise FileNotFoundError(f'Este checkout no incluye MDiffDecoder: {decoder_source}')
    if not CONFIG.is_file():
        raise FileNotFoundError(f'Falta config.yml de MDiff4STR: {CONFIG}')
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f'Falta el checkpoint MDiff4STR: {CHECKPOINT}')
    ckpt_sha = sha256_file(CHECKPOINT)
    if ckpt_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint SHA256 distinto del oficial: {ckpt_sha}')
    with CONFIG.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    arch = cfg.get('Architecture') or {}
    enc = arch.get('Encoder') or {}
    dec = arch.get('Decoder') or {}
    pp = cfg.get('PostProcess') or {}
    expected = {
        'Architecture.algorithm': (arch.get('algorithm'), 'NRTR'),
        'Encoder.name': (enc.get('name'), 'SVTRv2LNConvTwo33'),
        'Decoder.name': (dec.get('name'), 'MDiffDecoder'),
        'Decoder.sampler_step': (dec.get('sampler_step'), 3),
        'Decoder.sample_k': (dec.get('sample_k'), 3),
        'Decoder.semi_autoregressive_decoding': (dec.get('semi_autoregressive_decoding'), True),
        'PostProcess.name': (pp.get('name'), 'ARLabelDecode'),
    }
    for name, (actual, wanted) in expected.items():
        if type(actual) is not type(wanted) or actual != wanted:
            raise ValueError(f'{name} inesperado: {actual!r}; esperado {wanted!r}')
    for flag in ('parallel_decoding', 'autoregressive_decoding', 'low_confidence_decoding',
                 'random_mask_decoding', 'cloze_mask_decoding'):
        if dec.get(flag) is not False:
            raise ValueError(f'Decoder.{flag} debe permanecer en False')
    dict_relative = './tools/utils/EN_symbol_dict.txt'
    if (cfg.get('Global', {}).get('character_dict_path') != dict_relative
            or pp.get('character_dict_path') != dict_relative):
        raise ValueError('El diccionario configurado no coincide con la baseline original')
    dictionary = OPENOCR / dict_relative
    if not dictionary.is_file():
        raise FileNotFoundError(f'No existe el diccionario: {dictionary}')
    return {
        'openocr_commit': commit,
        'checkpoint_sha256': ckpt_sha,
        'config_sha256': sha256_file(CONFIG),
        'dictionary': str(dictionary.resolve()),
    }


def load_checkpoint_strict(torch, model):
    """Deserialización clásica únicamente del checkpoint con SHA256 fijado."""
    try:
        blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        mode = 'weights_only=True'
    except Exception as exc:
        print(f'Checkpoint oficial incompatible con weights_only: {type(exc).__name__}', flush=True)
        blob = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        mode = 'weights_only=False (SHA256 fijado y verificado)'
    if not isinstance(blob, dict):
        raise TypeError(f'Checkpoint inesperado: {type(blob)}')
    state = blob.get('state_dict', blob)
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint sin state_dict válido')
    model.load_state_dict(state, strict=True)
    print('Pesos cargados STRICT=True | método:', mode, flush=True)
    del state, blob


def save_csv(path: Path, rows: list[dict]):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size debe ser positivo')

    print('=== GVIS: MDIFF4STR-SMALL ZERO-SHOT / ELPD COMPLETO ===', flush=True)
    rows, details = load_rows_and_details()
    print(f'Dataset: ELPD completo, {len(rows)} imágenes (GT candidato + auditoría)', flush=True)
    print('Splits originales:', details['original_split_counts'], flush=True)
    print('Origen de GT:', details['gt_source_counts'], flush=True)
    print('GT auditados incorporados:', details['audited_rows_merged'], flush=True)
    print('GT corregidos por auditoría:', details['candidate_gt_changed_by_audit'], flush=True)
    print('SHA256 manifest completo:', EXPECTED_FULL_MANIFEST_SHA256, flush=True)
    print('SHA256 manifest estricto:', EXPECTED_STRICT_MANIFEST_SHA256, flush=True)
    assets = validate_assets()
    print('OpenOCR commit:', assets['openocr_commit'], flush=True)
    print('Checkpoint SHA256:', assets['checkpoint_sha256'], flush=True)
    print('Config SHA256:', assets['config_sha256'], flush=True)
    print('Decoder: MDiffDecoder; 3 pasos; sample_k=3; semi_autoregressive=True', flush=True)

    if any((OUTPUT / name).exists() for name in ('predictions.csv', 'summary.json', 'summary.txt')):
        raise FileExistsError(f'Ya existen resultados en {OUTPUT}; no se sobrescriben')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No se ha detectado GPU CUDA: utiliza GVIS: Ejecutar en GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    sys.path.insert(0, str(PROJECT))
    from src.evaluation.ocr_metrics import score_complete
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    runtime_cfg = Config(str(CONFIG)).cfg
    runtime_cfg['Global']['character_dict_path'] = assets['dictionary']
    runtime_cfg['PostProcess']['character_dict_path'] = assets['dictionary']
    runtime_cfg['Global']['pretrained_model'] = None
    runtime_cfg['Global']['checkpoints'] = None
    runtime_cfg['Global']['device'] = 'gpu'
    print('GPU:', torch.cuda.get_device_name(0), '| batch-size:', args.batch_size, flush=True)
    load_start = time.perf_counter()
    recognizer = OpenRecognizer(config=runtime_cfg, mode='server', backend='torch', use_gpu='true')
    load_checkpoint_strict(torch, recognizer.model)
    recognizer.model.eval()
    load_seconds = time.perf_counter() - load_start
    num_params = sum(p.numel() for p in recognizer.model.parameters())
    if num_params != 18_891_072:
        raise RuntimeError(f'Número de parámetros MDiff4STR-Small inesperado: {num_params}')
    print('Parámetros:', num_params, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    forward_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix='stage_mdiff4str_small_', dir=OUTPUT) as temp:
        stage = Path(temp)
        all_dir = stage / 'all'
        warm_dir = stage / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        for index, row in enumerate(rows):
            source = DEFAULT_IMAGE_ROOT / row['image_path']
            dest_name = f'{index:04d}{source.suffix.lower()}'
            (all_dir / dest_name).symlink_to(source)
            if index < args.batch_size:
                (warm_dir / dest_name).symlink_to(source)
        print(f'Calentamiento: dos batches GPU, batch={args.batch_size}', flush=True)
        with torch.inference_mode():
            for _ in range(2):
                warm = recognizer(img_path=str(warm_dir), batch_num=args.batch_size)
                if len(warm) != min(len(rows), args.batch_size):
                    raise RuntimeError(f'Warm-up incompleto: {len(warm)} predicciones')
            torch.cuda.synchronize()
        # Mismo reinicio de RNG que en la baseline estricta original.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        original_forward = recognizer.model.forward

        def timed_forward(*forward_args, **forward_kwargs):
            torch.cuda.synchronize()
            tic = time.perf_counter()
            output = original_forward(*forward_args, **forward_kwargs)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - tic) * 1000)
            return output

        recognizer.model.forward = timed_forward
        print(f'Inferencia: {len(rows)} imágenes, batch={args.batch_size}', flush=True)
        infer_start = time.perf_counter()
        try:
            with torch.inference_mode():
                results = recognizer(img_path=str(all_dir), batch_num=args.batch_size)
                torch.cuda.synchronize()
        finally:
            recognizer.model.forward = original_forward
        infer_seconds = time.perf_counter() - infer_start

    if len(results) != EXPECTED_IMAGES:
        raise RuntimeError(f'Inferencia incompleta: {len(results)}/{EXPECTED_IMAGES}')
    expected_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    if len(forward_ms) != expected_batches:
        raise RuntimeError(f'Número de forwards inesperado: {len(forward_ms)} frente a {expected_batches}')

    # Relacionar cada salida con su imagen mediante el índice del archivo
    # temporal, no por el orden devuelto por OpenOCR.
    predictions_by_index = {}
    for result in results:
        result_file = result.get('file')
        if result_file is None:
            raise RuntimeError('OpenOCR no ha proporcionado la ruta de imagen')
        try:
            index = int(Path(result_file).stem)
        except ValueError as exc:
            raise RuntimeError(f'Nombre de salida no identificable: {result_file}') from exc
        if index in predictions_by_index or not (0 <= index < len(rows)):
            raise RuntimeError(f'Índice de imagen duplicado o fuera de rango: {index}')
        predictions_by_index[index] = result
    if len(predictions_by_index) != len(rows):
        raise RuntimeError('No existen predicciones para todas las imágenes del manifest')

    predictions = []
    by_split = defaultdict(list)
    by_source = defaultdict(list)
    for index, row in enumerate(rows):
        result = predictions_by_index[index]
        raw = result['text']
        if not isinstance(raw, str):
            raise TypeError(f'Predicción no textual en {row["image_path"]}: {type(raw)}')
        gt = row['text_normalized'].strip()
        score = score_complete(gt, raw)
        confidence = result.get('score')
        entry = {
            'image_path': row['image_path'],
            'scenario': row['scenario'],
            'gt_source': row['gt_source'],
            'gt_candidate_original': row['gt_candidate_original'],
            'ground_truth': gt,
            'prediction_raw': raw,
            'prediction_normalized': score['prediction_normalized'],
            'exact_match': score['exact_match'],
            'edit_distance': score['edit_distance'],
            'gt_length': score['gt_length'],
            'pred_length': score['pred_length'],
            'confidence': round(float(confidence), 6) if confidence is not None else '',
        }
        predictions.append(entry)
        by_split[row['scenario']].append(entry)
        by_source[row['gt_source']].append(entry)

    metrics = metric_summary(predictions)
    summary = {
        'dataset': 'ELPD/Kaggle full dataset (candidate labels; audited labels overlaid)',
        'dataset_size': len(rows),
        'label_warning': 'Most labels are filename candidates, not visually verified; train/val/test original splits included.',
        'manifest_sha256': EXPECTED_FULL_MANIFEST_SHA256,
        'gt_audit': details,
        'metrics_by_original_split': {k: metric_summary(v) for k, v in sorted(by_split.items())},
        'metrics_by_gt_source': {k: metric_summary(v) for k, v in sorted(by_source.items())},
        'model': 'MDiff4STR-Small',
        'source': MODEL_SOURCE,
        'source_commit_openocr': assets['openocr_commit'],
        'checkpoint_filename': CHECKPOINT.name,
        'checkpoint_sha256': assets['checkpoint_sha256'],
        'config_sha256': assets['config_sha256'],
        'finetuning': False,
        'normalization': 'uppercase and retain ASCII A-Z0-9',
        'inference_mode': 'MDiffDecoder; 3 steps; sample_k=3; semi-autoregressive; official configuration',
        'sampler_step': 3,
        'sample_k': 3,
        'seed': 42,
        'EM': metrics['exact_matches'] / len(rows),
        'EM_percent': metrics['EM_percent'],
        'CER': metrics['levenshtein_errors'] / metrics['gt_characters'],
        'CER_percent': metrics['CER_percent'],
        'exact_matches': metrics['exact_matches'],
        'levenshtein_errors': metrics['levenshtein_errors'],
        'gt_characters': metrics['gt_characters'],
        'batch_size': args.batch_size,
        'forward_batches': len(forward_ms),
        'gpu_forward_ms_total': round(sum(forward_ms), 3),
        'gpu_forward_ms_per_image': round(sum(forward_ms) / len(rows), 3),
        'gpu_forward_batch_ms_mean': round(statistics.mean(forward_ms), 3),
        'inference_wall_seconds_including_image_loading': round(infer_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'num_parameters': num_params,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'latency_note': 'GPU forward includes 3 denoising steps; excludes image loading and postprocessing; not an end-to-end timing.',
    }
    save_csv(OUTPUT / 'predictions.csv', predictions)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = [
        '=== GVIS: MDIFF4STR-SMALL ZERO-SHOT / ELPD COMPLETO ===',
        f'Recortes evaluados: {len(rows)}/{EXPECTED_IMAGES}',
        f'Aciertos exactos (GT disponible): {metrics["exact_matches"]}/{len(rows)}',
        f'EM (GT disponible): {metrics["EM_percent"]:.2f}%',
        f'CER (GT disponible): {metrics["CER_percent"]:.2f}% ({metrics["levenshtein_errors"]}/{metrics["gt_characters"]} caracteres GT)',
        f'GT auditados incorporados: {details["audited_rows_merged"]}',
        f'Pasos difusión: 3 | sample_k=3 | semi-autoregressive=True',
        f'Latencia GPU-forward por imagen (batch {args.batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms',
        f'Parámetros: {num_params}',
        f'Checkpoint SHA256: {assets["checkpoint_sha256"]}',
        f'Manifest SHA256: {EXPECTED_FULL_MANIFEST_SHA256}',
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
