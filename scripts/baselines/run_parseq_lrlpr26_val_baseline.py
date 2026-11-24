#!/usr/bin/env python3
"""PARSeq zero-shot sobre las 10 000 imagenes HR de validacion de LRLPR-26.

No entrena, rectifica ni altera el dataset. Lee val_manifest.csv (split agrupado
por matricula), comprueba rutas/etiquetas y produce predicciones y EM/CER.
El GT se conserva como annotation_unverified: las metricas son provisionales.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import time

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_MANIFEST = DATASETS / 'processed/LRLPR-26_LPO/val_manifest.csv'
DEFAULT_IMAGES = DATASETS / 'raw/LRLPR-26/selected_hr/train'
DEFAULT_OUT = PROJECT / 'results/baselines/LRLPR-26/parseq_pretrained_val10k'
EXPECTED_COUNT = 10000
PATTERNS = {'Brazilian': re.compile(r'[A-Z]{3}[0-9]{4}\Z'),
            'Mercosur': re.compile(r'[A-Z]{3}[0-9][A-Z][0-9]{2}\Z')}
FIELDS = ('image_path', 'source_path', 'text_normalized', 'scenario',
          'plate_layout', 'track_id', 'view_id', 'label_status', 'source_split', 'split')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize_plate(s):
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1,
                               previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def load_and_check(manifest, image_root, expected_count):
    manifest = manifest.resolve()
    image_root = image_root.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f'No existe el manifest: {manifest}')
    if not image_root.is_dir():
        raise FileNotFoundError(f'No existe la carpeta HR: {image_root}')
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = sorted(set(FIELDS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f'Columnas ausentes: {missing}')
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f'Filas esperadas: {expected_count}; obtenidas: {len(rows)}')

    unique_images = set()
    by_track = defaultdict(list)
    groups = Counter()
    for idx, row in enumerate(rows, 2):
        gt = row['text_normalized'].strip()
        layout = row['plate_layout'].strip()
        scenario = row['scenario'].strip()
        track = row['track_id'].strip()
        if layout not in PATTERNS or not PATTERNS[layout].fullmatch(gt):
            raise ValueError(f'GT/plantilla invalido en fila {idx}: {layout} {gt!r}')
        if scenario not in ('Scenario-A', 'Scenario-B'):
            raise ValueError(f'Escenario invalido en fila {idx}: {scenario}')
        if row['split'].strip() != 'val' or row['source_split'].strip() != 'train':
            raise ValueError(f'Splits inesperados en fila {idx}: {row["split"]}, {row["source_split"]}')
        if row['label_status'].strip() != 'annotation_unverified':
            raise ValueError(f'Estado GT inesperado en fila {idx}')
        if not track.startswith(scenario + '/' + layout + '/track_'):
            raise ValueError(f'Track incoherente en fila {idx}: {track}')
        source_path = Path(row['source_path'])
        if source_path.is_absolute() or '..' in source_path.parts:
            raise ValueError(f'source_path insegura en fila {idx}: {source_path}')
        # En el conjunto HR original, Scenario-A usa PNG y Scenario-B usa JPG.
        # La ruta debe coincidir con el track, la vista y la extension real.
        extension = '.png' if scenario == 'Scenario-A' else '.jpg'
        expected_source = f'train/{track}/hr-{row["view_id"].strip()}{extension}'
        if source_path.as_posix() != expected_source:
            raise ValueError(
                f'Ruta vs track/vista no coinciden en fila {idx}: {source_path} '
                f'(se esperaba {expected_source})'
            )
        absolute = (manifest.parent / row['image_path'].strip()).resolve()
        expected_image = image_root / source_path.relative_to('train')
        if absolute != expected_image:
            raise ValueError(f'Ruta relativa incorrecta en fila {idx}: {absolute} != {expected_image}')
        if not absolute.is_file():
            raise FileNotFoundError(f'Imagen ausente, fila {idx}: {absolute}')
        if absolute in unique_images:
            raise ValueError(f'Imagen duplicada: {absolute}')
        unique_images.add(absolute)
        row['_absolute_path'] = absolute
        row['_gt'] = gt
        row['_scenario'] = scenario
        row['_layout'] = layout
        by_track[track].append(row)
        groups[(scenario, layout)] += 1
    for track, group in by_track.items():
        if len(group) != 5 or len({r['_gt'] for r in group}) != 1 or {r['view_id'] for r in group} != {'001','002','003','004','005'}:
            raise ValueError(f'Track de 5 vistas o GT incoherente: {track}')
    if len(by_track) * 5 != len(rows):
        raise ValueError('Numero de tracks inesperado')
    return rows, groups, len(by_track)


def metrics(rows):
    count = len(rows)
    matched = sum(int(r['exact_match']) for r in rows)
    edits = sum(int(r['edit_distance']) for r in rows)
    chars = sum(int(r['gt_length']) for r in rows)
    return {'images': count, 'correct': matched, 'errors': count-matched,
            'EM_percent': round(100*matched/count, 4) if count else None,
            'CER_percent': round(100*edits/chars, 4) if chars else None,
            'edit_operations': edits, 'gt_characters': chars}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGES)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--expected-count', type=int, default=EXPECTED_COUNT)
    parser.add_argument('--check-manifest-only', action='store_true', help='Solo validar rutas/GT, sin GPU ni salida')
    args = parser.parse_args()
    if args.batch_size < 1 or args.expected_count < 1:
        parser.error('batch-size y expected-count deben ser positivos')
    rows, groups, n_tracks = load_and_check(args.manifest, args.image_root, args.expected_count)
    manifest_sha = sha256_file(args.manifest)
    print('=== PARSEQ ZERO-SHOT / LRLPR-26 HR VAL ===', flush=True)
    print(f'[OK] Imagenes: {len(rows)} | tracks: {n_tracks}', flush=True)
    for (scenario, layout), n in sorted(groups.items()):
        print(f'[OK] {scenario}/{layout}: {n} HR', flush=True)
    print('[OK] SHA256 manifest:', manifest_sha, flush=True)
    print('[AVISO] GT annotation_unverified, resultados provisionales', flush=True)
    if args.check_manifest_only:
        print('RESULTADO: PREFLIGHT FINALIZADO (SIN INFERENCIA)', flush=True)
        return

    outputs = ('predictions.csv', 'summary.json', 'summary.txt')
    exists = [x for x in outputs if (args.out / x).exists()]
    if exists:
        raise FileExistsError(f'No se sobrescriben resultados existentes en {args.out}: {exists}')
    missing = [name for name in ('nltk', 'yaml') if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Faltan paquetes: {missing}. Preparar entorno usado por la baseline PARSeq ELPD.')

    import torch
    from PIL import Image
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('GPU no disponible; ejecutar con GVIS: Ejecutar en GPU')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    os.environ.setdefault('TORCH_HOME', str(DATASETS / 'models/torch_hub'))
    Path(os.environ['TORCH_HOME']).mkdir(parents=True, exist_ok=True)
    print('[OK] GPU:', torch.cuda.get_device_name(0), flush=True)
    start_load = time.perf_counter()
    model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True)
    model.eval().to(device)
    load_s = time.perf_counter() - start_load
    size = tuple(model.hparams.img_size)
    transform = T.Compose([T.Resize(size, T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize(0.5, 0.5)])

    with Image.open(rows[0]['_absolute_path']) as im:
        first = transform(im.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(2):
            model.tokenizer.decode(model(first).softmax(-1))
    torch.cuda.synchronize()

    results = []
    forward_ms = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch_size):
            batch_rows = rows[i:i + args.batch_size]
            images = []
            for row in batch_rows:
                with Image.open(row['_absolute_path']) as im:
                    images.append(transform(im.convert('RGB')))
            batch = torch.stack(images).to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(batch)
            labels, confidence = model.tokenizer.decode(logits.softmax(-1))
            torch.cuda.synchronize()
            delta_ms = (time.perf_counter() - t0) * 1000
            forward_ms += delta_ms
            if len(labels) != len(batch_rows):
                raise RuntimeError(f'Batch incoherente en indice {i}')
            for row, raw, conf in zip(batch_rows, labels, confidence):
                pred = normalize_plate(raw)
                gt = row['_gt']
                results.append({
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
                    'confidence': round(float(conf.prod().item()), 6),
                    'batch_forward_ms': round(delta_ms, 3),
                })
            if len(results) % 1000 == 0 or len(results) == len(rows):
                print(f'Inferencia: {len(results)}/{len(rows)}', flush=True)
    elapsed = time.perf_counter() - started
    if len(results) != len(rows):
        raise RuntimeError('Inferencia incompleta')

    split_groups = defaultdict(list)
    for r in results:
        split_groups[(r['scenario'], r['plate_layout'])].append(r)
    group_metrics = {f'{sc}/{lo}': metrics(chunk) for (sc,lo),chunk in sorted(split_groups.items())}
    overall = metrics(results)
    summary = {
        'dataset': 'LRLPR-26', 'original_source_split': 'train',
        'evaluation_split': 'val grouped by text/track',
        'annotation_status': 'annotation_unverified',
        'method': 'zero-shot (pretrained PARSeq), no finetuning',
        'model': 'PARSeq', 'checkpoint_source': 'baudm/parseq:main',
        'manifest_sha256': manifest_sha, 'n_tracks': n_tracks,
        'metrics': overall, 'by_scenario_layout': group_metrics,
        'gpu_forward_ms_total': round(forward_ms, 3),
        'gpu_forward_ms_per_image': round(forward_ms / len(rows), 4),
        'inference_wall_seconds_with_image_loading': round(elapsed, 3),
        'checkpoint_load_seconds': round(load_s, 3),
        'batch_size': args.batch_size,
        'note': 'No usar como test independiente; conjunto de validacion interna del dataset original. Etiquetas no auditadas individualmente.'
    }
    args.out.mkdir(parents=True, exist_ok=True)
    columns = ['image_path', 'source_path', 'track_id', 'scenario', 'plate_layout',
               'ground_truth','label_status','prediction_raw','prediction_normalized',
               'exact_match','edit_distance','gt_length','pred_length','confidence','batch_forward_ms']
    with (args.out / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(results)
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['=== PARSEQ LRLPR-26 HR: ZERO-SHOT VAL ===',
             f'Imagenes: {len(rows)} | Tracks: {n_tracks}',
             f'EM: {overall["EM_percent"]:.2f}% ({overall["correct"]}/{len(rows)})',
             f'CER: {overall["CER_percent"]:.2f}% ({overall["edit_operations"]}/{overall["gt_characters"]})',
             f'GPU forward: {forward_ms / len(rows):.3f} ms/imagen',
             f'Tiempo de inferencia incluyendo carga de imagenes: {elapsed:.2f} s',
             'AVISO: etiquetas no auditadas visualmente; validacion interna, no test externo',
             f'Predicciones: {args.out / "predictions.csv"}',
             'RESULTADO: BASELINE FINALIZADA (SIN FINE-TUNING)']
    (args.out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
