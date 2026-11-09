#!/usr/bin/env python3
"""Baseline zero-shot MDiff4STR-Small sobre 491 matrículas TEST de UC3M-LP.

Reutiliza el mismo checkpoint, configuracion, decodificacion, batch y protocolo
GPU-forward que run_mdiff4str_elpd_baseline.py, pero evalua etiquetas parciales
con src.evaluation.partial_metrics (asteriscos = posiciones anonimizadas).

--check-only: verificaciones de solo lectura sobre CPU sin importar OpenOCR.
Sin opciones: inferencia en A100, resultados en results/baselines/UC3M-LP/mdiff4str_small/.

No usa caracteres ocultos como GT, no recalcula etiquetas, no entrena.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
OPENOCR = PROJECT / 'third_party/OpenOCR'
MANIFEST = DATASETS / 'processed/UC3M-LP_LPO/manifest.csv'
IMAGE_ROOT = DATASETS / 'processed/UC3M-LP_LPO'
OUT = PROJECT / 'results/baselines/UC3M-LP/mdiff4str_small'
EXPECTED_MANIFEST_SHA = '97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194'

# Ambos modulos viven junto a este script en scripts/baselines/ y ya se usan
# en los baselines ELPD/UC3M. Se comparten los validadores para evitar deriva.
import run_mdiff4str_elpd_baseline as mdiff  # noqa: E402
from run_parseq_uc3m_baseline import load_manifest  # noqa: E402
sys.path.insert(0, str(PROJECT))
from src.evaluation.partial_metrics import score_partial, summarize_partial  # noqa: E402


def preflight():
    print('=== GVIS: MDiff4STR-SMALL / UC3M-LP ===', flush=True)
    manifest_sha = mdiff.sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(f'Manifest UC3M-LP distinto del utilizado por PARSeq/SVTRv2: {manifest_sha}')

    rows, splits = load_manifest(MANIFEST, IMAGE_ROOT)
    if len(rows) != 491:
        raise RuntimeError(f'Esperadas 491 matriculas de test; hay {len(rows)}')
    # Valida commit OpenOCR, arquitectura MDiffDecoder, configuracion, diccionario
    # y SHA256 del checkpoint oficial; no importa el modelo ni usa GPU.
    cfg, commit, config_sha = mdiff.validate_config_and_sources()
    print(f'Manifest: {MANIFEST}', flush=True)
    print(f'Filas: {sum(splits.values())}; Train={splits["train"]}; Test={len(rows)}', flush=True)
    print('Solo se evaluara TEST (491 PNG).', flush=True)
    print(f'SHA256 manifest: {manifest_sha}', flush=True)
    print(f'OpenOCR commit: {commit}', flush=True)
    print(f'Checkpoint: {mdiff.CKPT}', flush=True)
    print(f'SHA256 checkpoint: {mdiff.EXPECTED_CHECKPOINT_SHA256}', flush=True)
    print(f'SHA256 config: {config_sha}', flush=True)
    print('Decoder: MDiffDecoder; sampler_step=3; sample_k=3; semi_autoregressive=True', flush=True)
    print('Metricas: partial_EM estricto + masked_error_rate_proxy (NO CER completo).', flush=True)
    return rows, manifest_sha, commit, config_sha, cfg


def _records_from_results(rows: list[dict[str, str]], raw_results: list[dict]):
    """Relaciona cada prediccion con su PNG usando el indice del symlink temporal."""
    indexed: dict[int, dict] = {}
    for result in raw_results:
        if not isinstance(result, dict) or 'file' not in result or 'text' not in result:
            raise RuntimeError(f'Formato inesperado de salida OpenOCR: {result!r}')
        file_name = Path(str(result['file'])).name
        if not file_name.endswith('.png') or not file_name[:-4].isdigit():
            raise RuntimeError(f'No se puede identificar la salida: {file_name!r}')
        index = int(file_name[:-4])
        if not 0 <= index < len(rows) or index in indexed:
            raise RuntimeError(f'Salida duplicada o indice fuera de rango: {index}')
        indexed[index] = result
    if len(indexed) != len(rows):
        raise RuntimeError(f'Predicciones incompletas: {len(indexed)}/{len(rows)}')

    records = []
    metric_pairs = []
    for index, row in enumerate(rows):
        result = indexed[index]
        raw = result['text']
        if not isinstance(raw, str):
            raise TypeError(f'Prediccion no textual en {index}: {type(raw).__name__}')
        gt = row['text_normalized'].strip()
        mask = row['known_mask'].strip()
        scored = score_partial(gt, mask, raw)
        metric_pairs.append((gt, mask, raw))
        confidence = result.get('score')
        records.append({
            'image_path': row['image_path'],
            'source_path': row['source_path'],
            'annotation_path': row.get('annotation_path', ''),
            'view_id': row.get('view_id', ''),
            'scenario': row['scenario'],
            'label_status': row['label_status'],
            'ground_truth_masked': scored['ground_truth_masked'],
            'known_mask': scored['known_mask'],
            'ground_truth_visible': row['text_visible'],
            'prediction_raw': raw,
            'prediction_normalized': scored['prediction_normalized'],
            'gt_length': scored['gt_length'],
            'pred_length': scored['pred_length'],
            'known_characters': scored['known_characters'],
            'masked_characters': scored['masked_characters'],
            'length_match': scored['length_match'],
            'partial_exact_match': scored['partial_exact_match'],
            'known_errors_if_aligned': scored['known_errors_if_aligned'],
            'masked_edit_distance': scored['masked_edit_distance'],
            'confidence': round(float(confidence), 6) if confidence is not None else '',
        })
    return records, summarize_partial(metric_pairs)


def _write_outputs(predictions: list[dict], summary: dict, report: list[str]):
    if len(predictions) != 491:
        raise RuntimeError(f'Se intentan guardar {len(predictions)} resultados en vez de 491')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    (OUT / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    (OUT / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only', action='store_true', help='Comprobar datos/pesos/config sin GPU')
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > 491:
        ap.error('--batch-size debe estar entre 1 y 491')

    rows, manifest_sha, commit, config_sha, _ = preflight()
    if args.check_only:
        print('ESTADO: PREFLIGHT CORRECTO; GPU E INFERENCIA PENDIENTES', flush=True)
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No hay GPU CUDA visible. Usar GVIS: Ejecutar en GPU.')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.infer_rec import OpenRecognizer
    from tools.engine.config import Config

    runtime_cfg = Config(str(mdiff.CONFIG)).cfg
    official_dict = str((OPENOCR / 'tools/utils/EN_symbol_dict.txt').resolve())
    runtime_cfg['Global']['character_dict_path'] = official_dict
    runtime_cfg['PostProcess']['character_dict_path'] = official_dict
    runtime_cfg['Global']['pretrained_model'] = None
    runtime_cfg['Global']['checkpoints'] = None
    runtime_cfg['Global']['device'] = 'gpu'
    print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Cargando modelo y checkpoint MDiff4STR-Small...', flush=True)
    load_start = time.perf_counter()
    recognizer = OpenRecognizer(config=runtime_cfg, mode='server', backend='torch', use_gpu='true')
    mdiff.load_weights_strict(torch, recognizer.model)
    recognizer.model.eval()
    load_seconds = time.perf_counter() - load_start
    parameters = sum(p.numel() for p in recognizer.model.parameters())
    print(f'Checkpoint cargado STRICT=True; parametros={parameters:,}; carga={load_seconds:.2f}s', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    forward_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix='uc3m_mdiff_', dir=OUT) as tmp:
        tmp_root = Path(tmp)
        all_dir = tmp_root / 'images'
        warm_dir = tmp_root / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        for idx, row in enumerate(rows):
            source = IMAGE_ROOT / row['image_path']
            filename = f'{idx:04d}.png'
            (all_dir / filename).symlink_to(source)
            if idx < args.batch_size:
                (warm_dir / filename).symlink_to(source)

        with torch.inference_mode():
            print(f'Calentamiento GPU: 2 batches de {min(args.batch_size, len(rows))} imagenes', flush=True)
            for _ in range(2):
                warm = recognizer(img_path=str(warm_dir), batch_num=args.batch_size)
                if len(warm) != min(args.batch_size, len(rows)):
                    raise RuntimeError('Warmup incompleto')
            torch.cuda.synchronize()

        # Mantener el protocolo de tiempos del baseline MDiff4STR/ELPD.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        original_forward = recognizer.model.forward

        def timed_forward(*a, **kw):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = original_forward(*a, **kw)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - start) * 1000)
            return output

        recognizer.model.forward = timed_forward
        try:
            print(f'Inferencia UC3M-LP: {len(rows)} imagenes de test; batch={args.batch_size}', flush=True)
            wall_start = time.perf_counter()
            with torch.inference_mode():
                raw_results = recognizer(img_path=str(all_dir), batch_num=args.batch_size)
                torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
        finally:
            recognizer.model.forward = original_forward

    if not forward_ms:
        raise RuntimeError('No se han registrado tiempos de forward GPU')
    records, metrics = _records_from_results(rows, raw_results)
    summary = {
        'dataset': 'UC3M-LP',
        'dataset_partition': 'test',
        'dataset_size': len(rows),
        'manifest_sha256': manifest_sha,
        'model': 'MDiff4STR-Small',
        'source': mdiff.MODEL_SOURCE,
        'source_commit_openocr': commit,
        'checkpoint_filename': mdiff.CKPT.name,
        'checkpoint_sha256': mdiff.EXPECTED_CHECKPOINT_SHA256,
        'config_sha256': config_sha,
        'finetuning': False,
        'normalization': 'uppercase A-Z/0-9 predictions; GT masked retains *',
        'inference_mode': 'MDiffDecoder; sampler_step=3; sample_k=3; semi_autoregressive=True; official config',
        'sampler_step': 3,
        'sample_k': 3,
        'seed': 42,
        'metric_protocol': (
            'partial_EM: same length and all known positions correct. '
            'masked_error_rate_proxy: Levenshtein with * as zero-cost substitution; '
            'indels cost 1; divide by known GT characters. Not CER/EM complete.'
        ),
        **metrics,
        'batch_size': args.batch_size,
        'gpu_forward_ms_total': round(sum(forward_ms), 3),
        'gpu_forward_ms_per_image': round(sum(forward_ms) / len(rows), 3),
        'gpu_forward_batch_ms_mean': round(sum(forward_ms) / len(forward_ms), 3),
        'number_forward_calls': len(forward_ms),
        'inference_wall_seconds_including_image_loading': round(wall_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'num_parameters': parameters,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'latency_note': 'GPU forward (3 diffusion steps), excludes preprocessing/postprocessing; warmup=2',
    }
    report = [
        '=== GVIS: MDIFF4STR-SMALL ZERO-SHOT / UC3M-LP TEST ===',
        f'Recortes evaluados: {len(rows)}/491',
        f'EM parcial estricto: {metrics["partial_EM_percent"]:.2f}% '
        f'({metrics["partial_exact_matches"]}/{len(rows)})',
        f'Tasa de error con mascara (proxy, NO CER completo): {metrics["masked_error_rate_proxy_percent"]:.2f}% '
        f'({metrics["masked_edit_errors"]}/{metrics["known_gt_characters"]} conocidos)',
        f'Longitudes correctas: {metrics["equal_length_predictions"]}/{len(rows)}',
        'Error en posiciones conocidas (solo predicciones de longitud correcta): '
        + (f'{100 * metrics["aligned_known_error_rate"]:.2f}%'
           if metrics['aligned_known_error_rate'] is not None else 'N/D'),
        f'Latencia GPU-forward por imagen (batch {args.batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms',
        f'Tiempo total incluyendo carga de imagenes: {wall_seconds:.2f} s',
        f'Parametros: {parameters:,}',
        f'Checkpoint SHA256: {mdiff.EXPECTED_CHECKPOINT_SHA256}',
        f'SHA256 manifest: {manifest_sha}',
        f'OpenOCR commit: {commit}',
        'ATENCION: metricas parciales no comparables con EM/CER completos de ELPD.',
        'ESTADO: BASELINE PARCIAL FINALIZADO (sin fine-tuning)',
        f'Predicciones: {OUT / "predictions.csv"}',
    ]
    _write_outputs(records, summary, report)
    print('\n'.join(report), flush=True)
    print('Resultados:', OUT, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
