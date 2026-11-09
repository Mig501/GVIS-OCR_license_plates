#!/usr/bin/env python3
"""Baselines zero-shot de SVTRv2-Tiny/Small sobre UC3M-LP/test (491 PNG).

Uso:
  python scripts/baselines/run_svtrv2_uc3m_baselines.py --check-only  # CPU
  python scripts/baselines/run_svtrv2_uc3m_baselines.py               # GPU

Se usan los mismos checkpoints/configuraciones oficiales de SVTRv2 que en ELPD.
Se preservan las etiquetas parcialmente anonimizadas y se aplican las mismas
metricas parciales que en PARSeq sobre UC3M-LP. No altera datos ni pesos.
"""
from __future__ import annotations

import argparse
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
MANIFEST = DATASETS / 'processed/UC3M-LP_LPO/manifest.csv'
IMAGE_ROOT = DATASETS / 'processed/UC3M-LP_LPO'
OUTPUT_ROOT = PROJECT / 'results/baselines/UC3M-LP'
EXPECTED_MANIFEST_SHA = '97a6e041a996f2865796780f44b41ef4318b1c25066685e8293deb5df1939194'
EXPECTED_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_CHECKPOINT_SHA = {
    'tiny': '684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6',
    'small': '4c4e21c8eabd7a38519d9f10d6ffa286d470a07b1a8976109fe2b995a9414970',
}

# Importamos exactamente la auditoria de manifiesto utilizada en PARSeq/UC3M.
# El script de PARSeq y las metricas parciales ya se encuentran en el proyecto.
sys.path.insert(0, str(PROJECT))
from src.evaluation.partial_metrics import score_partial, summarize_partial  # noqa: E402
from run_parseq_uc3m_baseline import load_manifest  # noqa: E402


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_git_commit(repo: Path) -> str:
    """Lee HEAD sin el ejecutable git (no disponible dentro de Apptainer)."""
    git_dir = repo / '.git'
    if git_dir.is_file():
        pointer = git_dir.read_text(encoding='utf-8').strip()
        if not pointer.startswith('gitdir: '):
            raise RuntimeError(f'Metadatos Git inesperados: {git_dir}')
        git_dir = Path(pointer[8:].strip())
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    if not git_dir.is_dir():
        raise FileNotFoundError(f'No existe .git en {repo}')
    head = (git_dir / 'HEAD').read_text(encoding='utf-8').strip()
    if head.startswith('ref: '):
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            head = ref_path.read_text(encoding='utf-8').strip()
        else:
            packed_refs = git_dir / 'packed-refs'
            head = ''
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding='utf-8').splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        head = parts[0]
                        break
    if len(head) != 40 or any(c not in '0123456789abcdefABCDEF' for c in head):
        raise RuntimeError(f'No se pudo determinar commit Git en {repo}')
    return head.lower()


def prepare_config(variant: str) -> tuple[dict, Path, Path, str]:
    import yaml

    config_path = WEIGHTS / variant / 'config.yml'
    ckpt_path = WEIGHTS / variant / 'best.pth'
    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    arch = config.get('Architecture') or {}
    encoder = arch.get('Encoder') or {}
    decoder = arch.get('Decoder') or {}
    postprocess = config.get('PostProcess') or {}
    expected_dims = [64, 128, 256] if variant == 'tiny' else [96, 192, 384]
    if (arch.get('algorithm') != 'BGPD'
            or encoder.get('name') != 'SVTRv2LNConvTwo33'
            or encoder.get('dims') != expected_dims
            or encoder.get('depths') != [3, 6, 3]
            or decoder.get('name') != 'GTCDecoder'
            or decoder.get('infer_gtc') is not True
            or postprocess.get('name') != 'GTCLabelDecode'):
        raise ValueError(f'Config oficial de SVTRv2-{variant} no coincide con el checkpoint esperado')
    dictionary = config['Global']['character_dict_path']
    if not Path(dictionary).is_absolute():
        dictionary = str((OPENOCR / dictionary).resolve())
    if not Path(dictionary).is_file():
        raise FileNotFoundError(f'Diccionario OpenOCR no encontrado: {dictionary}')
    actual_ckpt_sha = sha256_file(ckpt_path)
    if actual_ckpt_sha != EXPECTED_CHECKPOINT_SHA[variant]:
        raise RuntimeError(f'Checkpoint {variant} distinto del utilizado en ELPD: {actual_ckpt_sha}')
    # Cambio exclusivo de ruta y selección de rama de decodificación oficial;
    # no se modifican arquitectura, pesos, ni parámetros del modelo.
    config['Global']['character_dict_path'] = dictionary
    if 'gtc_label_decode' in config['PostProcess']:
        nested = config['PostProcess']['gtc_label_decode']
        if isinstance(nested, dict) and 'character_dict_path' in nested:
            nested['character_dict_path'] = dictionary
    config['PostProcess']['only_gtc'] = True
    config['Global']['pretrained_model'] = None
    config['Global']['checkpoints'] = None
    config['Global']['device'] = 'gpu'
    return config, config_path, ckpt_path, actual_ckpt_sha


def preflight(variants: tuple[str, ...]) -> tuple[list[dict[str, str]], str]:
    print('=== GVIS: SVTRv2 / UC3M-LP ===', flush=True)
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f'Commit OpenOCR inesperado: {commit}')
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise RuntimeError(f'Manifest UC3M diferente al de PARSeq: {manifest_sha}')
    rows, splits = load_manifest(MANIFEST, IMAGE_ROOT)
    print(f'Manifest: {MANIFEST}', flush=True)
    print(f'Filas: {sum(splits.values())}; Train={splits["train"]}; Test={len(rows)}', flush=True)
    print('Solo se evaluara TEST (491 PNG).', flush=True)
    print(f'SHA256 manifest: {manifest_sha}', flush=True)
    print(f'OpenOCR commit: {commit}', flush=True)
    for variant in variants:
        config, cfg_path, ckpt_path, ckpt_sha = prepare_config(variant)
        print(f'{variant.upper()} config SHA256: {sha256_file(cfg_path)}', flush=True)
        print(f'{variant.upper()} checkpoint: {ckpt_path}', flush=True)
        print(f'{variant.upper()} checkpoint SHA256: {ckpt_sha}', flush=True)
        print('Decoder: GTCDecoder, infer_gtc=True, PostProcess=GTCLabelDecode', flush=True)
    print('Metricas: partial_EM estricto y masked_error_rate_proxy (NO CER convencional).', flush=True)
    return rows, manifest_sha


def load_official_checkpoint(torch, path: Path, expected_sha: str) -> dict:
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f'Hash del checkpoint ha cambiado: {path}')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        how = 'weights_only=True'
    except Exception as exc:
        # Solo se admite pickle para los archivos con SHA-256 oficial verificado.
        print(f'weights_only=True no pudo abrir el checkpoint: {type(exc).__name__}', flush=True)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        how = 'weights_only=False (SHA256 fijado)'
    if not isinstance(payload, dict):
        raise TypeError('Checkpoint no es un diccionario')
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint sin state_dict')
    print(f'Checkpoint cargado: {how}; claves={len(state)}', flush=True)
    return state


def save_results(folder: Path, records: list[dict], summary: dict, report: list[str]) -> None:
    if not records:
        raise ValueError('No se pueden guardar 0 predicciones')
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (folder / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')


def run_variant(variant: str, rows: list[dict[str, str]], manifest_sha: str,
                batch_size: int, torch, OpenRecognizer, Config) -> None:
    config, cfg_path, ckpt_path, ckpt_sha = prepare_config(variant)
    print(f'\n=== SVTRv2-{variant.upper()} ZERO-SHOT / UC3M-LP TEST ===', flush=True)
    t_load = time.perf_counter()
    recognizer = OpenRecognizer(config=config, mode='server', backend='torch', use_gpu='true')
    state = load_official_checkpoint(torch, ckpt_path, ckpt_sha)
    recognizer.model.load_state_dict(state, strict=True)
    del state
    recognizer.model.eval()
    load_seconds = time.perf_counter() - t_load
    params = sum(p.numel() for p in recognizer.model.parameters())
    print(f'Pesos cargados STRICT=True; parametros={params}; carga={load_seconds:.2f}s', flush=True)

    out_dir = OUTPUT_ROOT / f'svtrv2_{variant}'
    out_dir.mkdir(parents=True, exist_ok=True)
    # Las rutas con indice preservan el orden del manifiesto incluso si OpenOCR
    # enumera los archivos por nombre. Son symlinks temporales, no copias.
    with tempfile.TemporaryDirectory(prefix=f'uc3m_svtrv2_{variant}_', dir=out_dir) as tmp:
        stage = Path(tmp)
        all_dir = stage / 'all'
        warm_dir = stage / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        for i, row in enumerate(rows):
            real = IMAGE_ROOT / row['image_path']
            filename = f'{i:04d}.png'
            (all_dir / filename).symlink_to(real)
            if i < batch_size:
                (warm_dir / filename).symlink_to(real)

        print(f'Calentamiento GPU: {min(batch_size, len(rows))} imagenes (2 repeticiones)', flush=True)
        with torch.inference_mode():
            for _ in range(2):
                warm = recognizer(img_path=str(warm_dir), batch_num=batch_size)
                if len(warm) != min(len(rows), batch_size):
                    raise RuntimeError(f'Warmup incompleto: {len(warm)}')
            torch.cuda.synchronize()

        forward_ms: list[float] = []
        original_forward = recognizer.model.forward

        def timed_forward(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
            forward_ms.append(1000.0 * (time.perf_counter() - start))
            return output

        recognizer.model.forward = timed_forward
        print(f'Inferencia: {len(rows)} imagenes, batch={batch_size}', flush=True)
        start_wall = time.perf_counter()
        try:
            with torch.inference_mode():
                predictions = recognizer(img_path=str(all_dir), batch_num=batch_size)
                torch.cuda.synchronize()
        finally:
            recognizer.model.forward = original_forward
        wall_seconds = time.perf_counter() - start_wall

    if len(predictions) != len(rows):
        raise RuntimeError(f'Salidas incompletas: {len(predictions)}/{len(rows)}')
    if not forward_ms:
        raise RuntimeError('No se pudieron medir tiempos de inferencia GPU')
    # OpenOCR debe devolver las imagenes en el mismo orden que los indices
    # usados en los enlaces temporales; fallo explicito si se desordena.
    for index, result in enumerate(predictions):
        actual_file = Path(str(result.get('file', ''))).name
        expected_file = f'{index:04d}.png'
        if actual_file != expected_file:
            raise RuntimeError(
                f'Orden de resultados de OpenOCR inesperado: {actual_file!r} vs {expected_file!r}'
            )

    records = []
    metric_pairs = []
    for row, result in zip(rows, predictions):
        raw = str(result['text'])
        scored = score_partial(row['text_normalized'].strip(), row['known_mask'].strip(), raw)
        metric_pairs.append((scored['ground_truth_masked'], scored['known_mask'], raw))
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
            'confidence': round(float(result['score']), 6),
        })
    metrics = summarize_partial(metric_pairs)
    summary = {
        'dataset': 'UC3M-LP',
        'dataset_partition': 'test',
        'dataset_size': len(rows),
        'manifest_sha256': manifest_sha,
        'model': f'SVTRv2-{variant.capitalize()}',
        'inference_mode': 'zero-shot; pretrained; BGPD + GTCDecoder(infer_gtc=True); GTCLabelDecode(only_gtc=True)',
        'source': 'Topdu/OpenOCR',
        'source_commit': EXPECTED_COMMIT,
        'checkpoint_filename': 'best.pth',
        'checkpoint_sha256': ckpt_sha,
        'config_sha256': sha256_file(cfg_path),
        'finetuning': False,
        'normalization': 'uppercase A-Z/0-9 predictions; masked GT preserves *',
        'metric_protocol': (
            'partial_EM: equal length and all known positions correct. '
            'masked_error_rate_proxy: wildcard Levenshtein distance / known GT chars; '
            'not comparable to complete CER/EM on ELPD.'
        ),
        **metrics,
        'batch_size': batch_size,
        'gpu_forward_ms_total': round(sum(forward_ms), 3),
        'gpu_forward_ms_per_image': round(sum(forward_ms) / len(rows), 3),
        'gpu_forward_batch_ms_mean': round(sum(forward_ms) / len(forward_ms), 3),
        'number_forward_calls': len(forward_ms),
        'inference_wall_seconds_including_image_loading': round(wall_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'num_parameters': params,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'latency_note': 'GPU-forward only; excludes preprocess/postprocess; warmup=2; common timed protocol per architecture',
    }
    report = [
        f'=== GVIS: SVTRv2-{variant.upper()} ZERO-SHOT / UC3M-LP TEST ===',
        f'Recortes evaluados: {len(rows)}/491',
        f'EM parcial estricto: {metrics["partial_EM_percent"]:.2f}% '
        f'({metrics["partial_exact_matches"]}/{len(rows)})',
        f'CER proxy con mascara (NO CER completo): {metrics["masked_error_rate_proxy_percent"]:.2f}% '
        f'({metrics["masked_edit_errors"]}/{metrics["known_gt_characters"]} conocidos)',
        f'Longitudes correctas: {metrics["equal_length_predictions"]}/{len(rows)}',
        'Tasa de error posicional en longitud correcta: '
        + (f'{100 * metrics["aligned_known_error_rate"]:.2f}%'
           if metrics['aligned_known_error_rate'] is not None else 'N/D'),
        f'Latencia GPU-forward por imagen (batch {batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms',
        f'Tiempo total incluyendo imagenes: {wall_seconds:.2f}s',
        f'Parametros modelo: {params}',
        f'SHA256 checkpoint: {ckpt_sha}',
        f'SHA256 manifest: {manifest_sha}',
        f'OpenOCR commit: {EXPECTED_COMMIT}',
        'ATENCION: las metricas parciales no son CER/EM completos.',
        'ESTADO: BASELINE PARCIAL FINALIZADO (sin fine-tuning)',
        f'Predicciones: {out_dir / "predictions.csv"}',
    ]
    save_results(out_dir, records, summary, report)
    print('\n'.join(report), flush=True)
    del recognizer
    gc.collect()
    torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Auditar datos y modelos sin usar GPU')
    parser.add_argument('--variant', choices=('tiny', 'small', 'both'), default='both')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('batch-size debe ser mayor que cero')
    variants = ('tiny', 'small') if args.variant == 'both' else (args.variant,)
    rows, manifest_sha = preflight(variants)
    if args.check_only:
        print('ESTADO: PREFLIGHT CORRECTO; GPU E INFERENCIA PENDIENTES', flush=True)
        return 0

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; utilizar GVIS: Ejecutar en GPU')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.infer_rec import OpenRecognizer
    from tools.engine.config import Config
    print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__, flush=True)
    for variant in variants:
        run_variant(variant, rows, manifest_sha, args.batch_size, torch, OpenRecognizer, Config)
    print('=== SVTRv2: BASELINES UC3M-LP COMPLETADOS ===', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
