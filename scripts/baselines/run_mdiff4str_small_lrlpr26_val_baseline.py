#!/usr/bin/env python3
"""MDiff4STR-Small zero-shot en 10 000 imágenes HR de validación de LRLPR-26.

Reutiliza el checkpoint, configuración y parámetros del decodificador empleados
para MDiff4STR-Small en ELPD; usa exactamente la misma selección de imágenes,
normalización y métricas que las demás baselines LRLPR-26.

La validación es interna; los GT originales son `annotation_unverified`.
Solo lee imágenes, manifiesto y pesos. Nunca sobrescribe resultados anteriores.
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

from run_parseq_lrlpr26_val_baseline import (
    DEFAULT_MANIFEST, DEFAULT_IMAGES, EXPECTED_COUNT,
    load_and_check, metrics, normalize_plate, edit_distance, sha256_file,
)

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
OPENOCR = PROJECT / 'third_party/OpenOCR'
WEIGHTS = DATASETS / 'weights/MDiff4STR/small'
CONFIG = WEIGHTS / 'config.yml'
CHECKPOINT = WEIGHTS / 'best.pth'
DEFAULT_OUT = PROJECT / 'results/baselines/LRLPR-26/mdiff4str_small_val10k'
EXPECTED_OPENOCR_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_CHECKPOINT_SHA256 = 'ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937'
EXPECTED_PARAMS = 18_891_072


def read_git_commit(repo: Path) -> str:
    """Resuelve HEAD sin depender del ejecutable git dentro del contenedor."""
    git_dir = repo / '.git'
    if git_dir.is_file():
        pointer = git_dir.read_text(encoding='utf-8').strip()
        if not pointer.startswith('gitdir: '):
            raise ValueError(f'.git desconocido: {git_dir}')
        git_dir = Path(pointer[len('gitdir: '):].strip())
        if not git_dir.is_absolute():
            git_dir = (repo / git_dir).resolve()
    if not git_dir.is_dir():
        raise FileNotFoundError(f'Falta .git de OpenOCR: {git_dir}')
    head = (git_dir / 'HEAD').read_text(encoding='utf-8').strip()
    if head.startswith('ref: '):
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            commit = ref_path.read_text(encoding='utf-8').strip()
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
                raise RuntimeError(f'No se puede resolver HEAD: {ref}')
    else:
        commit = head
    if len(commit) != 40 or any(c not in '0123456789abcdefABCDEF' for c in commit):
        raise ValueError(f'Commit OpenOCR no valido: {commit!r}')
    return commit.lower()


def validate_assets() -> dict[str, str]:
    """Comprueba los archivos y las opciones oficiales usadas en la baseline ELPD."""
    import yaml

    if not OPENOCR.is_dir():
        raise FileNotFoundError(f'OpenOCR no disponible: {OPENOCR}')
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_OPENOCR_COMMIT:
        raise ValueError(f'OpenOCR ha cambiado desde ELPD: {commit}')
    decoder_path = OPENOCR / 'openrec/modeling/decoders/mdiff_decoder.py'
    if not decoder_path.is_file():
        raise FileNotFoundError(f'Falta MDiffDecoder: {decoder_path}')
    for path in (CONFIG, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f'Falta archivo del modelo: {path}')
    ckpt_sha = sha256_file(CHECKPOINT)
    if ckpt_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint MDiff4STR-Small diferente del usado en ELPD: {ckpt_sha}')

    with CONFIG.open('r', encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    arch = cfg.get('Architecture') or {}
    encoder = arch.get('Encoder') or {}
    decoder = arch.get('Decoder') or {}
    post = cfg.get('PostProcess') or {}
    checks = {
        'Architecture.algorithm': (arch.get('algorithm'), 'NRTR'),
        'Encoder.name': (encoder.get('name'), 'SVTRv2LNConvTwo33'),
        'Decoder.name': (decoder.get('name'), 'MDiffDecoder'),
        'Decoder.sampler_step': (decoder.get('sampler_step'), 3),
        'Decoder.sample_k': (decoder.get('sample_k'), 3),
        'Decoder.semi_autoregressive_decoding': (decoder.get('semi_autoregressive_decoding'), True),
        'PostProcess.name': (post.get('name'), 'ARLabelDecode'),
    }
    for key, (actual, expected) in checks.items():
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'Configuracion {key} inesperada: {actual!r}, esperaba {expected!r}')
    for flag in ('parallel_decoding', 'autoregressive_decoding',
                 'low_confidence_decoding', 'random_mask_decoding',
                 'cloze_mask_decoding'):
        if decoder.get(flag) is not False:
            raise ValueError(f'Decoder.{flag} debe ser False, como en ELPD')
    relative_dict = './tools/utils/EN_symbol_dict.txt'
    if (cfg.get('Global', {}).get('character_dict_path') != relative_dict
            or post.get('character_dict_path') != relative_dict):
        raise ValueError('El diccionario de caracteres no coincide con ELPD')
    dictionary = OPENOCR / relative_dict
    if not dictionary.is_file():
        raise FileNotFoundError(f'Diccionario inexistente: {dictionary}')
    return {
        'source_commit': commit,
        'checkpoint_sha256': ckpt_sha,
        'config_sha256': sha256_file(CONFIG),
        'dictionary': str(dictionary.resolve()),
    }


def load_checkpoint_strict(torch, model) -> None:
    """Únicamente permite cargar el checkpoint oficial tras validar su SHA256."""
    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        how = 'weights_only=True'
    except Exception as exc:
        print(f'[AVISO] Carga de pesos oficial con fallback: {type(exc).__name__}', flush=True)
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        how = 'weights_only=False (SHA256 verificado)'
    if not isinstance(payload, dict):
        raise TypeError(f'Estructura de checkpoint inesperada: {type(payload)}')
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint sin state_dict valido')
    model.load_state_dict(state, strict=True)
    print(f'[OK] Pesos cargados STRICT=True ({how})', flush=True)
    del state, payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGES)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--expected-count', type=int, default=EXPECTED_COUNT)
    parser.add_argument('--check-manifest-only', action='store_true',
                        help='Validar rutas y GT antes de cargar el modelo; no utiliza GPU')
    args = parser.parse_args()
    if args.batch_size < 1 or args.expected_count < 1:
        parser.error('--batch-size y --expected-count deben ser positivos')

    print('=== MDIFF4STR-SMALL ZERO-SHOT / LRLPR-26 HR VAL ===', flush=True)
    rows, groups, n_tracks = load_and_check(args.manifest, args.image_root, args.expected_count)
    manifest_sha = sha256_file(args.manifest)
    print(f'[OK] Imagenes: {len(rows)} | tracks: {n_tracks}', flush=True)
    for (scenario, layout), n in sorted(groups.items()):
        print(f'[OK] {scenario}/{layout}: {n} HR', flush=True)
    print('[OK] SHA256 manifest:', manifest_sha, flush=True)
    print('[AVISO] GT annotation_unverified; validacion interna, no test externo', flush=True)
    if args.check_manifest_only:
        print('RESULTADO: PREFLIGHT FINALIZADO (SIN INFERENCIA)', flush=True)
        return 0

    output_files = ('predictions.csv', 'summary.json', 'summary.txt')
    existing = [name for name in output_files if (args.out / name).exists()]
    if existing:
        raise FileExistsError(f'No se sobrescriben resultados en {args.out}: {existing}')
    assets = validate_assets()
    print('[OK] OpenOCR commit:', assets['source_commit'], flush=True)
    print('[OK] Checkpoint SHA256:', assets['checkpoint_sha256'], flush=True)
    print('[OK] Config SHA256:', assets['config_sha256'], flush=True)
    print('[OK] Decoder: MDiffDecoder; sampler_step=3; sample_k=3; semi-autoregressive=True', flush=True)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no visible: usar GVIS: Ejecutar en GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print('[OK] GPU:', torch.cuda.get_device_name(0), flush=True)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    runtime = Config(str(CONFIG)).cfg
    runtime['Global']['character_dict_path'] = assets['dictionary']
    runtime['PostProcess']['character_dict_path'] = assets['dictionary']
    runtime['Global']['pretrained_model'] = None
    runtime['Global']['checkpoints'] = None
    runtime['Global']['device'] = 'gpu'

    load_started = time.perf_counter()
    recognizer = OpenRecognizer(config=runtime, mode='server', backend='torch', use_gpu='true')
    load_checkpoint_strict(torch, recognizer.model)
    recognizer.model.eval()
    load_seconds = time.perf_counter() - load_started
    n_params = sum(p.numel() for p in recognizer.model.parameters())
    if n_params != EXPECTED_PARAMS:
        raise RuntimeError(f'Numero de parametros inesperado: {n_params} != {EXPECTED_PARAMS}')
    print(f'[OK] Modelo cargado; parametros: {n_params:,}', flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    forward_times: list[float] = []
    with tempfile.TemporaryDirectory(prefix='stage_mdiff_lrlpr26_', dir=args.out) as temp:
        stage = Path(temp)
        all_dir, warm_dir = stage / 'all', stage / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        # 5 digitos permiten indexar 10 000 filas sin conflictos de orden;
        # se conserva la extension original (PNG para A; JPG para B).
        for index, row in enumerate(rows):
            source: Path = row['_absolute_path']
            name = f'{index:05d}{source.suffix.lower()}'
            (all_dir / name).symlink_to(source)
            if index < args.batch_size:
                (warm_dir / name).symlink_to(source)

        print('[OK] Warm-up: 2 pasadas sin medir', flush=True)
        with torch.inference_mode():
            for _ in range(2):
                warm = recognizer(img_path=str(warm_dir), batch_num=args.batch_size)
                if len(warm) != min(len(rows), args.batch_size):
                    raise RuntimeError('Warm-up MDiff4STR incompleto')
            torch.cuda.synchronize()

        # Mismo seed que la baseline MDiff4STR de ELPD tras el warm-up.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        original_forward = recognizer.model.forward

        def timed_forward(*a, **kw):
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = original_forward(*a, **kw)
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - started) * 1000.0)
            return outputs

        recognizer.model.forward = timed_forward
        print(f'[OK] Iniciando inferencia: {len(rows)} recortes; batch={args.batch_size}', flush=True)
        inference_started = time.perf_counter()
        try:
            with torch.inference_mode():
                results = recognizer(img_path=str(all_dir), batch_num=args.batch_size)
                torch.cuda.synchronize()
        finally:
            recognizer.model.forward = original_forward
        inference_seconds = time.perf_counter() - inference_started

    if len(results) != len(rows):
        raise RuntimeError(f'Inferencia incompleta: {len(results)}/{len(rows)}')
    expected_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    if len(forward_times) != expected_batches:
        raise RuntimeError(f'GPU forwards: {len(forward_times)}; se esperaban {expected_batches}')

    # OpenOCR puede devolver resultados en un orden distinto al manifiesto.
    # Vinculamos cada prediccion por el indice codificado en el nombre temporal.
    predictions_by_index: dict[int, dict] = {}
    for item in results:
        if not isinstance(item, dict) or 'file' not in item or 'text' not in item:
            raise RuntimeError(f'Resultado OpenOCR sin file/text: {item!r}')
        try:
            idx = int(Path(str(item['file'])).stem)
        except ValueError as exc:
            raise RuntimeError(f'No se pudo relacionar imagen y prediccion: {item["file"]}') from exc
        if idx in predictions_by_index or not (0 <= idx < len(rows)):
            raise RuntimeError(f'Indice duplicado o fuera de rango: {idx}')
        predictions_by_index[idx] = item
    if len(predictions_by_index) != len(rows):
        raise RuntimeError('Faltan predicciones al asociar el indice del manifiesto')

    predictions = []
    by_group = defaultdict(list)
    for idx, row in enumerate(rows):
        item = predictions_by_index[idx]
        raw = item['text']
        if not isinstance(raw, str):
            raise TypeError(f'Prediccion no textual en fila {idx}: {raw!r}')
        pred = normalize_plate(raw)
        gt = row['_gt']
        confidence = item.get('score')
        entry = {
            'image_path': row['image_path'],
            'source_path': row['source_path'],
            'track_id': row['track_id'],
            'scenario': row['_scenario'],
            'plate_layout': row['_layout'],
            'ground_truth': gt,
            'label_status': row['label_status'],
            'prediction_raw': raw,
            'prediction_normalized': pred,
            'exact_match': int(pred == gt),
            'edit_distance': edit_distance(pred, gt),
            'gt_length': len(gt),
            'pred_length': len(pred),
            'confidence': round(float(confidence), 6) if confidence is not None else '',
            'batch_forward_ms': '',
        }
        predictions.append(entry)
        by_group[(row['_scenario'], row['_layout'])].append(entry)

    overall = metrics(predictions)
    group_metrics = {
        f'{scenario}/{layout}': metrics(group)
        for (scenario, layout), group in sorted(by_group.items())
    }
    summary = {
        'dataset': 'LRLPR-26',
        'original_source_split': 'train',
        'evaluation_split': 'val grouped by text/track',
        'annotation_status': 'annotation_unverified',
        'model': 'MDiff4STR-Small',
        'method': 'zero-shot; official MDiffDecoder; no fine-tuning',
        'source': 'OpenOCR',
        'source_commit': assets['source_commit'],
        'checkpoint_filename': str(CHECKPOINT),
        'checkpoint_sha256': assets['checkpoint_sha256'],
        'config_sha256': assets['config_sha256'],
        'manifest_sha256': manifest_sha,
        'n_tracks': n_tracks,
        'metrics': overall,
        'by_scenario_layout': group_metrics,
        'gpu_forward_ms_total': round(sum(forward_times), 3),
        'gpu_forward_ms_per_image': round(sum(forward_times) / len(rows), 4),
        'gpu_forward_batch_ms_mean': round(statistics.mean(forward_times), 3),
        'inference_wall_seconds_with_image_loading': round(inference_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'batch_size': args.batch_size,
        'number_forward_batches': len(forward_times),
        'num_parameters': n_params,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'diffusion_sampler_step': 3,
        'diffusion_sample_k': 3,
        'diffusion_semiautoregressive': True,
        'note': 'Validacion interna con GT no auditado visualmente; forward GPU incluye 3 pasos de difusion.',
    }
    columns = ['image_path', 'source_path', 'track_id', 'scenario', 'plate_layout',
               'ground_truth', 'label_status', 'prediction_raw', 'prediction_normalized',
               'exact_match', 'edit_distance', 'gt_length', 'pred_length', 'confidence',
               'batch_forward_ms']
    with (args.out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predictions)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = [
        '=== MDIFF4STR-SMALL ZERO-SHOT / LRLPR-26 HR VAL ===',
        f'Recortes evaluados: {len(rows)}/{args.expected_count}',
        f'Tracks: {n_tracks}',
        f'Aciertos exactos: {overall["correct"]}/{len(rows)}',
        f'EM: {overall["EM_percent"]:.2f}%',
        f'CER: {overall["CER_percent"]:.2f}%',
        f'Latencia GPU-forward por imagen (batch {args.batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms',
        f'Parametros: {n_params}',
        f'Checkpoint SHA256: {assets["checkpoint_sha256"]}',
        f'Manifest SHA256: {manifest_sha}',
        'AVISO: GT annotation_unverified; resultados provisionales de validacion interna.',
        'RESULTADO: BASELINE FINALIZADA (SIN FINE-TUNING)',
        f'Predicciones: {args.out / "predictions.csv"}',
        f'Resumen: {args.out / "summary.json"}',
    ]
    (args.out / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
