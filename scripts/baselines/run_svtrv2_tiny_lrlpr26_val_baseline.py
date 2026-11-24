#!/usr/bin/env python3
"""Baseline zero-shot SVTRv2-Tiny, 10 000 recortes HR val de LRLPR-26.

Se reutilizan la configuracion, checkpoint y decodificacion del SVTRv2-Tiny
ya evaluado en ELPD. El split val es interno (train original de LRLPR-26),
con GT no auditado visualmente. No se modifica ni se entrena ningun modelo.

Resultados: results/baselines/LRLPR-26/svtrv2_tiny_val10k/
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
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
CONFIG = DATASETS / 'weights/SVTRv2/tiny/config.yml'
CHECKPOINT = DATASETS / 'weights/SVTRv2/tiny/best.pth'
OUTPUT = PROJECT / 'results/baselines/LRLPR-26/svtrv2_tiny_val10k'
EXPECTED_OPENOCR_COMMIT = '1ccfc6ee6161f7133b192e16af3a9265273997ba'
EXPECTED_CHECKPOINT_SHA256 = '684f23bc3a2b1f724cadfd275e1a185456dd51b5e56c8830780d1998e90f77f6'


def read_git_commit(repo: Path) -> str:
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
        raise ValueError(f'Hash git no valido: {commit}')
    return commit.lower()


def validate_assets() -> dict[str, str]:
    if not OPENOCR.is_dir():
        raise FileNotFoundError(f'Falta OpenOCR: {OPENOCR}')
    commit = read_git_commit(OPENOCR)
    if commit != EXPECTED_OPENOCR_COMMIT:
        raise ValueError(f'Commit de OpenOCR distinto del utilizado en ELPD: {commit}')
    for path in (CONFIG, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f'Falta archivo del modelo: {path}')
    ckpt_sha = sha256_file(CHECKPOINT)
    if ckpt_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint distinto al evaluado anteriormente: {ckpt_sha}')
    return {'source_commit': commit,
            'checkpoint_sha256': ckpt_sha,
            'config_sha256': sha256_file(CONFIG)}


def validated_config(Config):
    cfg = Config(str(CONFIG)).cfg
    arch = cfg.get('Architecture', {})
    decoder = arch.get('Decoder', {})
    pp = cfg.get('PostProcess', {})
    if not (arch.get('algorithm') == 'BGPD'
            and arch.get('Encoder', {}).get('name') == 'SVTRv2LNConvTwo33'
            and arch.get('Encoder', {}).get('dims') == [64, 128, 256]
            and decoder.get('name') == 'GTCDecoder'
            and decoder.get('infer_gtc') is True
            and pp.get('name') == 'GTCLabelDecode'):
        raise ValueError(f'Configuracion inesperada de SVTRv2-Tiny: {CONFIG}')
    dic = Path(cfg['Global']['character_dict_path'])
    if not dic.is_absolute():
        dic = (OPENOCR / dic).resolve()
    if not dic.is_file():
        raise FileNotFoundError(f'Diccionario oficial no encontrado: {dic}')
    cfg['Global']['character_dict_path'] = str(dic)
    cfg['PostProcess']['only_gtc'] = True
    cfg['Global']['pretrained_model'] = None
    cfg['Global']['checkpoints'] = None
    cfg['Global']['device'] = 'gpu'
    return cfg


def checkpoint_state(torch):
    # Se comprueba la suma SHA-256 antes de deserializar.
    try:
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    except Exception as exc:
        print(f'Fallback de carga para checkpoint oficial verificado: {type(exc).__name__}', flush=True)
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError('Checkpoint con estructura no esperada')
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict) or not state:
        raise TypeError('Checkpoint sin state_dict')
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGES)
    ap.add_argument('--out', type=Path, default=OUTPUT)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--expected-count', type=int, default=EXPECTED_COUNT)
    ap.add_argument('--check-manifest-only', action='store_true')
    args = ap.parse_args()
    if args.batch_size < 1 or args.expected_count < 1:
        ap.error('batch-size y expected-count deben ser positivos')

    print('=== SVTRv2-TINY ZERO-SHOT / LRLPR-26 HR VAL ===', flush=True)
    rows, groups, n_tracks = load_and_check(args.manifest, args.image_root, args.expected_count)
    manifest_sha = sha256_file(args.manifest)
    print(f'[OK] Imagenes: {len(rows)} | tracks: {n_tracks}', flush=True)
    for (scenario, layout), n in sorted(groups.items()):
        print(f'[OK] {scenario}/{layout}: {n} HR', flush=True)
    print('[OK] SHA256 manifest:', manifest_sha, flush=True)
    print('[AVISO] GT annotation_unverified, resultados provisionales', flush=True)
    if args.check_manifest_only:
        print('RESULTADO: PREFLIGHT FINALIZADO (SIN INFERENCIA)', flush=True)
        return 0

    required_outputs = ('predictions.csv', 'summary.json', 'summary.txt')
    existent = [p for p in required_outputs if (args.out / p).exists()]
    if existent:
        raise FileExistsError(f'No se sobrescriben resultados en {args.out}: {existent}')
    assets = validate_assets()
    print('[OK] OpenOCR commit:', assets['source_commit'], flush=True)
    print('[OK] Checkpoint SHA256:', assets['checkpoint_sha256'], flush=True)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no visible: usar GVIS: Ejecutar en GPU')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    print('[OK] GPU:', torch.cuda.get_device_name(0), flush=True)
    os.chdir(OPENOCR)
    sys.path.insert(0, str(OPENOCR))
    from tools.infer_rec import OpenRecognizer
    from tools.engine.config import Config

    cfg = validated_config(Config)
    load_started = time.perf_counter()
    recognizer = OpenRecognizer(config=cfg, mode='server', backend='torch', use_gpu='true')
    state = checkpoint_state(torch)
    recognizer.model.load_state_dict(state, strict=True)
    del state
    recognizer.model.eval()
    load_seconds = time.perf_counter() - load_started
    params = sum(p.numel() for p in recognizer.model.parameters())
    print('[OK] Pesos cargados estrictamente; parametros:', params, flush=True)
    print('[OK] Decodificador GTC (infer_gtc=True; only_gtc=True); batch:', args.batch_size, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    forward_times = []
    with tempfile.TemporaryDirectory(prefix='stage_svtrv2_tiny_', dir=args.out) as tmp:
        stage = Path(tmp)
        all_dir, warm_dir = stage / 'all', stage / 'warmup'
        all_dir.mkdir()
        warm_dir.mkdir()
        # 5 digitos: el orden lexicografico coincide exactamente con el manifiesto.
        # 10 000 registros requieren 00000.png ... 09999.png.
        for i, row in enumerate(rows):
            src = row['_absolute_path']
            name = f'{i:05d}{src.suffix.lower()}'
            (all_dir / name).symlink_to(src)
            if i < args.batch_size:
                (warm_dir / name).symlink_to(src)
        with torch.inference_mode():
            for _ in range(2):
                warm = recognizer(img_path=str(warm_dir), batch_num=args.batch_size)
                if len(warm) != min(args.batch_size, len(rows)):
                    raise RuntimeError(f'Warm-up incompleto: {len(warm)}')
            torch.cuda.synchronize()

        original_forward = recognizer.model.forward
        def timed_forward(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = original_forward(*a, **kw)
            torch.cuda.synchronize()
            forward_times.append((time.perf_counter() - t0) * 1000.0)
            return result
        recognizer.model.forward = timed_forward
        begin = time.perf_counter()
        print(f'[OK] Iniciando inferencia de {len(rows)} imagenes...', flush=True)
        try:
            with torch.inference_mode():
                results = recognizer(img_path=str(all_dir), batch_num=args.batch_size)
                torch.cuda.synchronize()
        finally:
            recognizer.model.forward = original_forward
        wall_seconds = time.perf_counter() - begin

    if len(results) != len(rows):
        raise RuntimeError(f'Inferencia incompleta: {len(results)}/{len(rows)}')
    if not forward_times:
        raise RuntimeError('No se registraron tiempos de GPU-forward')

    predictions = []
    by_group = defaultdict(list)
    for row, result in zip(rows, results):
        if not isinstance(result, dict) or 'text' not in result or 'score' not in result:
            raise ValueError(f'Formato de prediccion inesperado: {result!r}')
        raw = str(result['text'])
        pred = normalize_plate(raw)
        gt = row['_gt']
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
            'confidence': round(float(result['score']), 6),
            'batch_forward_ms': '',  # Timing total detallado en summary.json.
        }
        predictions.append(entry)
        by_group[(row['_scenario'], row['_layout'])].append(entry)

    whole = metrics(predictions)
    group_metrics = {f'{sc}/{lo}': metrics(part) for (sc, lo), part in sorted(by_group.items())}
    summary = {
        'dataset': 'LRLPR-26',
        'original_source_split': 'train',
        'evaluation_split': 'val grouped by text/track',
        'annotation_status': 'annotation_unverified',
        'model': 'SVTRv2-Tiny',
        'method': 'zero-shot, no fine-tuning',
        'source': 'OpenOCR',
        'source_commit': assets['source_commit'],
        'checkpoint_filename': str(CHECKPOINT),
        'checkpoint_sha256': assets['checkpoint_sha256'],
        'config_sha256': assets['config_sha256'],
        'manifest_sha256': manifest_sha,
        'n_tracks': n_tracks,
        'metrics': whole,
        'by_scenario_layout': group_metrics,
        'gpu_forward_ms_total': round(sum(forward_times), 3),
        'gpu_forward_ms_per_image': round(sum(forward_times) / len(rows), 4),
        'gpu_forward_batch_ms_mean': round(statistics.mean(forward_times), 3),
        'inference_wall_seconds_with_image_loading': round(wall_seconds, 3),
        'checkpoint_load_seconds': round(load_seconds, 3),
        'batch_size': args.batch_size,
        'number_forward_batches': len(forward_times),
        'num_parameters': params,
        'torch_version': torch.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'note': 'Validacion interna; etiquetas no verificadas visualmente. Tiempo GPU-forward excluye lectura y posprocesado.',
    }
    columns = ['image_path', 'source_path', 'track_id', 'scenario', 'plate_layout',
               'ground_truth', 'label_status', 'prediction_raw', 'prediction_normalized',
               'exact_match', 'edit_distance', 'gt_length', 'pred_length', 'confidence',
               'batch_forward_ms']
    with (args.out / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predictions)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = [
        '=== SVTRv2-TINY ZERO-SHOT / LRLPR-26 HR VAL ===',
        f'Recortes evaluados: {len(rows)}/{args.expected_count}',
        f'Tracks: {n_tracks}',
        f'Aciertos exactos: {whole["correct"]}/{len(rows)}',
        f'EM: {whole["EM_percent"]:.2f}%',
        f'CER: {whole["CER_percent"]:.2f}%',
        f'Latencia GPU-forward por imagen (batch {args.batch_size}): {summary["gpu_forward_ms_per_image"]:.3f} ms',
        f'Checkpoint SHA256: {assets["checkpoint_sha256"]}',
        f'Manifest SHA256: {manifest_sha}',
        'AVISO: etiquetas annotation_unverified; evaluacion provisional de validacion interna.',
        'RESULTADO: BASELINE FINALIZADA (SIN FINE-TUNING)',
        f'Predicciones: {args.out / "predictions.csv"}',
        f'Resumen: {args.out / "summary.json"}',
    ]
    (args.out / 'summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
