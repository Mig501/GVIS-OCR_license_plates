#!/usr/bin/env python3
"""Baseline zero-shot MDiff4STR-Small (checkpoint oficial) en 80 recortes ELPD.

- Utiliza pesos y config oficiales de topdu/MDiff4STR/mdiff4str_small.
- Conserva la inferencia original: tres pasos, semi-autoregressive decoding.
- Evalua con las mismas reglas EM/CER que PARSeq y SVTRv2.
- --check-only audita datos, modelo y config sin requerir GPU ni cargar pesos en RAM.
- No modifica ni el dataset ni los checkpoints. No descarga modelos en la ejecucion.

Guardado: scripts/baselines/run_mdiff4str_elpd_baseline.py
Salida: results/baselines/ELPD/mdiff4str_small/
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
import tempfile
import time

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
OPENOCR = PROJECT / 'third_party/OpenOCR'
WEIGHTS = DATASETS / 'weights/MDiff4STR/small'
CKPT = WEIGHTS / 'best.pth'
CONFIG = WEIGHTS / 'config.yml'
MANIFEST = PROJECT / 'results/audit/ELPD/final_eval/verified_test_manifest.csv'
IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
OUTPUT = PROJECT / 'results/baselines/ELPD/mdiff4str_small'
EXPECTED_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_MANIFEST_SHA256 = 'c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c'
# SHA-256 anunciado por el repositorio oficial topdu/MDiff4STR en Hugging Face.
EXPECTED_CHECKPOINT_SHA256 = 'ebc96f11e5739eae98491f7c8c140157ed12b35dafa220511ef9d399cc20a937'
MODEL_SOURCE = 'https://huggingface.co/topdu/MDiff4STR/tree/main/mdiff4str_small'


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_manifest() -> list[dict[str, str]]:
    sha = sha256_file(MANIFEST)
    if sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f'Manifest no coincide con PARSeq: {sha}')
    with MANIFEST.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        needed = {'image_path', 'text_normalized', 'gt_review_status', 'strict_gt_eligible'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError(f'Faltan columnas: {needed - set(reader.fieldnames or [])}')
        rows = list(reader)
    if len(rows) != 80 or Counter(r['gt_review_status'] for r in rows) != {'verified': 76, 'corrected': 4}:
        raise ValueError('Deben ser 80 filas: 76 verified y 4 corrected')
    seen = set()
    for r in rows:
        relative = r['image_path'].strip()
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or not relative.startswith('images/'):
            raise ValueError(f'Ruta insegura: {relative}')
        if relative in seen:
            raise ValueError(f'Ruta repetida: {relative}')
        seen.add(relative)
        if not (IMAGE_ROOT / p).is_file():
            raise FileNotFoundError(IMAGE_ROOT / p)
        if r['strict_gt_eligible'].strip().lower() != 'yes':
            raise ValueError(f'Etiqueta no verificada: {relative}')
        gt = r['text_normalized'].strip()
        if not gt or not gt.isascii() or not gt.isalnum() or gt != gt.upper():
            raise ValueError(f'Etiqueta inválida: {relative}: {gt!r}')
    return rows


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


def validate_config_and_sources() -> tuple[dict, str, str]:
    import yaml

    if not OPENOCR.is_dir():
        raise FileNotFoundError(f'OpenOCR no encontrado en {OPENOCR}')
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f'OpenOCR en commit diferente; esperado {EXPECTED_COMMIT}, actual {commit}')
    mdiff_source = OPENOCR / 'openrec/modeling/decoders/mdiff_decoder.py'
    if not mdiff_source.is_file():
        raise FileNotFoundError(f'Este checkout no incluye MDiffDecoder: {mdiff_source}')

    with CONFIG.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    a = cfg.get('Architecture') or {}
    d = a.get('Decoder') or {}
    pp = cfg.get('PostProcess') or {}
    enc = a.get('Encoder') or {}
    required = {
        'Architecture.algorithm': (a.get('algorithm'), 'NRTR'),
        'Architecture.Encoder.name': (enc.get('name'), 'SVTRv2LNConvTwo33'),
        'Architecture.Decoder.name': (d.get('name'), 'MDiffDecoder'),
        'Architecture.Decoder.sampler_step': (d.get('sampler_step'), 3),
        'Architecture.Decoder.sample_k': (d.get('sample_k'), 3),
        'Architecture.Decoder.semi_autoregressive_decoding': (d.get('semi_autoregressive_decoding'), True),
        'PostProcess.name': (pp.get('name'), 'ARLabelDecode'),
    }
    for name, (current, expected) in required.items():
        if current != expected or type(current) is not type(expected):
            raise ValueError(f'Config oficial inesperada: {name}={current!r}; esperado {expected!r}')
    for flag in ('parallel_decoding', 'autoregressive_decoding', 'low_confidence_decoding',
                 'random_mask_decoding', 'cloze_mask_decoding'):
        if d.get(flag) is not False:
            raise ValueError(f'Config {flag} distinta de False')

    expected_dict = './tools/utils/EN_symbol_dict.txt'
    if cfg.get('Global', {}).get('character_dict_path') != expected_dict or pp.get('character_dict_path') != expected_dict:
        raise ValueError('Ruta del diccionario oficial inesperada')
    dict_file = OPENOCR / expected_dict
    if not dict_file.is_file():
        raise FileNotFoundError(f'Falta diccionario oficial: {dict_file}')
    actual_checkpoint_sha = sha256_file(CKPT)
    if actual_checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint MDiff4STR-Small no coincide con SHA-256 oficial: {actual_checkpoint_sha}')
    config_sha = sha256_file(CONFIG)
    return cfg, commit, config_sha


def load_weights_strict(torch, model):
    try:
        blob = torch.load(CKPT, map_location='cpu', weights_only=True)
    except Exception as exc:
        # Se permite pickle solo tras comparar el SHA-256 completo con el oficial.
        print(f'Aviso: weights_only=True no permite abrir este checkpoint ({type(exc).__name__}).', flush=True)
        print('Carga clásica autorizada únicamente para el SHA-256 oficial fijado.', flush=True)
        blob = torch.load(CKPT, map_location='cpu', weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f'Checkpoint no es diccionario: {type(blob).__name__}')
    state = blob.get('state_dict', blob)
    if not isinstance(state, dict) or not state:
        raise TypeError('Checkpoint sin state_dict válido')
    model.load_state_dict(state, strict=True)
    del state, blob


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('No hay predicciones')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Solo verifica manifest, pesos y arquitectura; no GPU')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 80:
        parser.error('--batch-size debe estar entre 1 y 80')

    print('=== GVIS: MDiff4STR-SMALL / ELPD ===', flush=True)
    rows = validate_manifest()
    cfg, commit, config_sha = validate_config_and_sources()
    print('Manifest: 80/80, 76 verified + 4 corrected', flush=True)
    print('SHA256 manifest:', EXPECTED_MANIFEST_SHA256, flush=True)
    print('OpenOCR commit:', commit, flush=True)
    print('Checkpoint SHA256:', EXPECTED_CHECKPOINT_SHA256, flush=True)
    print('Config SHA256:', config_sha, flush=True)
    print('Decoder: MDiffDecoder; 3 pasos; sample_k=3; semi_autoregressive=True', flush=True)
    if args.check_only:
        print('ESTADO: PREFLIGHT CORRECTO; GPU E INFERENCIA PENDIENTES', flush=True)
        return

    sys.path.insert(0, str(PROJECT))
    from src.evaluation.ocr_metrics import score_complete, summarize_complete
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No hay GPU CUDA: ejecutar con GVIS: Ejecutar en GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if hasattr(torch.cuda, 'manual_seed_all'):
        torch.cuda.manual_seed_all(42)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.engine.config import Config
    from tools.infer_rec import OpenRecognizer

    # Usamos la configuración oficial sin alterar los hiperparámetros del decoder.
    runtime_cfg = Config(str(CONFIG)).cfg
    official_dict_abs = str((OPENOCR / 'tools/utils/EN_symbol_dict.txt').resolve())
    runtime_cfg['Global']['character_dict_path'] = official_dict_abs
    runtime_cfg['PostProcess']['character_dict_path'] = official_dict_abs
    runtime_cfg['Global']['pretrained_model'] = None
    runtime_cfg['Global']['checkpoints'] = None
    runtime_cfg['Global']['device'] = 'gpu'

    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Construyendo modelo MDiff4STR-Small...', flush=True)
    started = time.perf_counter()
    recognizer = OpenRecognizer(config=runtime_cfg, mode='server', backend='torch', use_gpu='true')
    load_weights_strict(torch, recognizer.model)
    recognizer.model.eval()
    load_seconds = time.perf_counter() - started
    num_parameters = sum(p.numel() for p in recognizer.model.parameters())
    print(f'Pesos cargados STRICT=True; parametros={num_parameters:,}; tiempo={load_seconds:.2f}s', flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    forward_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix='mdiff_elpd_', dir=str(OUTPUT)) as tmp:
        tmp_path = Path(tmp)
        all_dir = tmp_path / 'images'
        warm_dir = tmp_path / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        for idx, r in enumerate(rows):
            origin = IMAGE_ROOT / r['image_path']
            fname = f'{idx:04d}{origin.suffix.lower()}'
            (all_dir / fname).symlink_to(origin)
            if idx < args.batch_size:
                (warm_dir / fname).symlink_to(origin)
        print('Calentamiento: dos batches GPU de', args.batch_size, 'imagenes', flush=True)
        with torch.inference_mode():
            for _ in range(2):
                warm_results = recognizer(img_path=str(warm_dir), batch_num=args.batch_size)
                if len(warm_results) != min(len(rows), args.batch_size):
                    raise RuntimeError('Warmup incompleto')
            torch.cuda.synchronize()
        # Reinicio RNG tras warmup si el decodificador emplea muestreo aleatorio.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        real_forward = recognizer.model.forward

        def timed_forward(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = real_forward(*a, **kw)
            torch.cuda.synchronize()
            forward_ms.append((time.perf_counter() - t0) * 1000)
            return out

        recognizer.model.forward = timed_forward
        print(f'Inferencia: {len(rows)} imagenes; batch={args.batch_size}', flush=True)
        start_infer = time.perf_counter()
        try:
            with torch.inference_mode():
                results = recognizer(img_path=str(all_dir), batch_num=args.batch_size)
                torch.cuda.synchronize()
        finally:
            recognizer.model.forward = real_forward
        wall_seconds = time.perf_counter() - start_infer

    if len(results) != len(rows):
        raise RuntimeError(f'Salidas incorrectas: {len(results)}/{len(rows)}')
    if len(forward_ms) != (len(rows) + args.batch_size - 1) // args.batch_size:
        raise RuntimeError(f'Número de forwards inesperado: {len(forward_ms)}')

    predictions = []
    pairs = []
    used = set()
    for result in results:
        result_file = result.get('file')
        if result_file is None:
            raise RuntimeError('OpenOCR no ha devuelto ruta de imagen en un resultado')
        try:
            idx = int(Path(result_file).stem)
        except ValueError as exc:
            raise RuntimeError(f'Archivo de resultado no identificable: {result_file}') from exc
        if idx < 0 or idx >= len(rows) or idx in used:
            raise RuntimeError(f'Indice de resultado duplicado/fuera de rango: {idx}')
        used.add(idx)
        r = rows[idx]
        gt = r['text_normalized'].strip()
        raw = result['text']
        if not isinstance(raw, str):
            raise TypeError(f'Texto de predicción no es str en imagen {idx}: {type(raw)}')
        scored = score_complete(gt, raw)
        pairs.append((gt, raw))
        score = result.get('score')
        predictions.append({
            'image_path': r['image_path'],
            'gt_review_status': r['gt_review_status'],
            'ground_truth': gt,
            'prediction_raw': raw,
            'prediction_normalized': scored['prediction_normalized'],
            'exact_match': scored['exact_match'],
            'edit_distance': scored['edit_distance'],
            'gt_length': scored['gt_length'],
            'pred_length': scored['pred_length'],
            'confidence': float(score) if score is not None else '',
            'batch_forward_ms': '',  # El tiempo corresponde al batch, no a cada imagen.
        })
    # Mantener orden del manifest para comparar CSV imagen a imagen.
    predictions.sort(key=lambda r: next(i for i, original in enumerate(rows) if original['image_path'] == r['image_path']))
    metrics = summarize_complete(pairs)
    summary = {
        'dataset': 'ELPD/Kaggle audited strict test',
        'dataset_size': len(rows),
        'gt_verified': 76,
        'gt_corrected': 4,
        'manifest_sha256': EXPECTED_MANIFEST_SHA256,
        'model': 'MDiff4STR-Small',
        'source': MODEL_SOURCE,
        'source_commit_openocr': commit,
        'checkpoint_filename': 'best.pth',
        'checkpoint_sha256': EXPECTED_CHECKPOINT_SHA256,
        'config_sha256': config_sha,
        'finetuning': False,
        'normalization': 'uppercase and retain ASCII A-Z0-9',
        'inference_mode': 'MDiffDecoder; 3 steps; semi-autoregressive; sample_k=3; official config',
        'sampler_step': 3,
        'sample_k': 3,
        'seed': 42,
        **metrics,
        'batch_size': args.batch_size,
        'gpu_forward_ms_total': round(sum(forward_ms), 3),
        'gpu_forward_ms_per_image': round(sum(forward_ms) / len(rows), 3),
        'gpu_forward_batch_ms_mean': round(sum(forward_ms) / len(forward_ms), 3),
        'forward_batches': len(forward_ms),
        'inference_wall_seconds_including_image_loading': round(wall_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'num_parameters': num_parameters,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'python_version': sys.version.split()[0],
        'latency_note': 'GPU forward includes 3 denoising steps; excludes preprocessing/postprocessing; benchmark comparable only under matched protocol',
    }
    write_csv(OUTPUT / 'predictions.csv', predictions)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = (
        '=== GVIS: MDIFF4STR-SMALL ZERO-SHOT / ELPD ===\n'
        f'Recortes evaluados: {len(rows)}/80\n'
        f'Aciertos exactos: {metrics["exact_matches"]}/80\n'
        f'EM: {metrics["EM_percent"]:.2f}%\n'
        f'CER: {metrics["CER_percent"]:.2f}% ({metrics["levenshtein_errors"]}/{metrics["gt_characters"]} GT chars)\n'
        f'Latencia GPU-forward por imagen (batch {args.batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms\n'
        f'Inferencia total (carga imagenes incluida): {wall_seconds:.2f} s\n'
        f'Pasos difusión: 3 | sample_k=3 | semi-autoregressive=True\n'
        f'Parámetros: {num_parameters:,}\n'
        f'Checkpoint SHA256: {EXPECTED_CHECKPOINT_SHA256}\n'
        f'OpenOCR commit: {commit}\n'
        'ESTADO: BASELINE FINALIZADO (sin fine-tuning)\n'
    )
    (OUTPUT / 'summary.txt').write_text(report, encoding='utf-8')
    print(report, flush=True)
    print('Resultados:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
