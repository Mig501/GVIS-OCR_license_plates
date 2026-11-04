#!/usr/bin/env python3
"""Crea splits train/val de LRLPR-26 HR sin compartir texto de matricula.

La unidad minima es el track (5 imágenes). Los tracks que comparten
text_normalized se asignan íntegramente al mismo split.

Los manifiestos derivados se guardan JUNTO a manifest.csv para conservar
la validez de las rutas relativas image_path y annotation_path.
No mueve ni modifica imágenes, anotaciones ni manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PROJECT_ROOT = Path('/home/miguel.sanchez/ocr_matriculas')
REQUIRED = {
    'image_path', 'text_normalized', 'track_id', 'scenario', 'plate_layout',
    'view_id', 'source_split', 'dataset',
}
EXPECTED_STRATA = {
    ('Scenario-A', 'Brazilian'): 5000,
    ('Scenario-A', 'Mercosur'): 5000,
    ('Scenario-B', 'Brazilian'): 2000,
    ('Scenario-B', 'Mercosur'): 8000,
}


def args_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_ROOT / 'processed/LRLPR-26_LPO/manifest.csv')
    p.add_argument('--reports', type=Path, default=PROJECT_ROOT / 'results/inspection/LRLPR-26')
    p.add_argument('--val-fraction', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expected-images', type=int, default=100000)
    p.add_argument('--expected-tracks', type=int, default=20000)
    p.add_argument('--expected-labels', type=int, default=16946)
    return p.parse_args()


def atomic_csv(path: Path, fields: list[str], data: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_text(value, encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    a = args_parse()
    if not 0 < a.val_fraction < 1:
        raise ValueError('--val-fraction debe estar entre 0 y 1')
    source = a.manifest.resolve()
    if source.name != 'manifest.csv':
        raise ValueError('El archivo de entrada debe llamarse manifest.csv, por seguridad')
    if not source.is_file():
        raise FileNotFoundError(source)

    original_sha = hash_file(source)
    with source.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if missing := (REQUIRED - set(fields)):
            raise ValueError(f'Faltan columnas del manifest: {sorted(missing)}')
        if 'split' in fields:
            raise ValueError('El manifest original ya tiene columna split: revisar antes de derivar')
        rows = list(reader)

    if len(rows) != a.expected_images:
        raise ValueError(f'Imágenes: {len(rows)}; esperadas: {a.expected_images}')

    # Construir identificadores de track y comprobar que las 5 imágenes y la etiqueta
    # se mantienen constantes dentro de cada track.
    tracks: dict[str, list[dict[str, str]]] = defaultdict(list)
    paths: set[str] = set()
    for i, r in enumerate(rows, 2):
        plate = r['text_normalized'].strip()
        track = r['track_id'].strip()
        image = r['image_path'].strip()
        if not plate or not track or not image:
            raise ValueError(f'Campos vacíos en fila {i}')
        if image in paths:
            raise ValueError(f'image_path repetida: {image}')
        paths.add(image)
        if r['source_split'] != 'train' or r['dataset'] != 'LRLPR-26':
            raise ValueError(f'Fila {i}: source_split o dataset inesperados')
        tracks[track].append(r)

    if len(tracks) != a.expected_tracks:
        raise ValueError(f'Tracks: {len(tracks)}; esperados: {a.expected_tracks}')

    plate_tracks: dict[str, set[str]] = defaultdict(set)
    track_stratum: dict[str, tuple[str, str]] = {}
    original_strata: Counter[tuple[str, str]] = Counter()
    for track, samples in tracks.items():
        labels = {x['text_normalized'] for x in samples}
        scenarios = {x['scenario'] for x in samples}
        layouts = {x['plate_layout'] for x in samples}
        views = {x['view_id'] for x in samples}
        if len(samples) != 5 or len(labels) != 1 or len(scenarios) != 1 or len(layouts) != 1 or len(views) != 5:
            raise ValueError(f'Track inválido: {track}: n={len(samples)}, textos={labels}, escenarios={scenarios}, layouts={layouts}, vistas={views}')
        label = next(iter(labels))
        stratum = (next(iter(scenarios)), next(iter(layouts)))
        if stratum not in EXPECTED_STRATA:
            raise ValueError(f'Estrato desconocido: {stratum}')
        original_strata[stratum] += 1
        track_stratum[track] = stratum
        plate_tracks[label].add(track)

    if len(plate_tracks) != a.expected_labels:
        raise ValueError(f'Etiquetas únicas: {len(plate_tracks)}; esperadas: {a.expected_labels}')
    if original_strata != Counter(EXPECTED_STRATA):
        raise ValueError(f'Distribución de tracks inesperada: {original_strata}')

    target = round(len(tracks) * a.val_fraction)
    # Orden pseudoaleatorio estable. Cada grupo de texto se selecciona entero;
    # los grupos que no caben en el cupo se saltan. Con suficientes grupos de
    # un track, el objetivo de 2000 tracks se puede alcanzar exactamente.
    labels_by_seed = sorted(
        plate_tracks,
        key=lambda x: hashlib.sha256(f'{a.seed}:{x}'.encode('utf-8')).digest(),
    )
    val_labels: set[str] = set()
    val_tracks: set[str] = set()
    for label in labels_by_seed:
        group = plate_tracks[label]
        if len(group) <= target - len(val_tracks):
            val_labels.add(label)
            val_tracks.update(group)
        if len(val_tracks) == target:
            break
    if len(val_tracks) != target:
        raise ValueError(f'No se alcanzó el objetivo de validación: {len(val_tracks)}/{target}')

    track_assign = {track: ('val' if track in val_tracks else 'train') for track in tracks}
    split_fields = fields + ['split']
    combined = [{**r, 'split': track_assign[r['track_id']]} for r in rows]
    train_rows = [r for r in combined if r['split'] == 'train']
    val_rows = [r for r in combined if r['split'] == 'val']
    val_strata = Counter(track_stratum[t] for t in val_tracks)

    if len(train_rows) + len(val_rows) != len(rows) or len(val_rows) != target * 5:
        raise RuntimeError('Inconsistencia de tamaño del split')
    if {r['text_normalized'] for r in train_rows} & {r['text_normalized'] for r in val_rows}:
        raise RuntimeError('Fuga detectada: hay textos presentes en train y val')
    if {r['track_id'] for r in train_rows} & {r['track_id'] for r in val_rows}:
        raise RuntimeError('Fuga detectada: hay tracks presentes en train y val')

    base = source.parent  # IMPORTANTE: mismo directorio para preservar rutas relativas
    atomic_csv(base / 'split_manifest.csv', split_fields, combined)
    atomic_csv(base / 'train_manifest.csv', split_fields, train_rows)
    atomic_csv(base / 'val_manifest.csv', split_fields, val_rows)
    groups_report = [
        {
            'text_normalized': label,
            'split': 'val' if label in val_labels else 'train',
            'num_tracks': str(len(trackset)),
            'track_ids': ';'.join(sorted(trackset)),
        }
        for label, trackset in sorted(plate_tracks.items())
    ]
    atomic_csv(a.reports / 'hr_split_groups.csv',
               ['text_normalized', 'split', 'num_tracks', 'track_ids'], groups_report)

    summary = [
        '=== LRLPR-26: SPLITS HR AGRUPADOS POR MATRICULA ===',
        f'Imagenes en manifest original: {len(rows)}',
        f'Tracks totales: {len(tracks)}',
        f'Etiquetas de matricula distintas: {len(plate_tracks)}',
        f'Seed: {a.seed}',
        f'Fraccion de validacion solicitada: {a.val_fraction:.3f}',
        f'Train: {len(train_rows)} imagenes, {len(tracks)-len(val_tracks)} tracks',
        f'Val: {len(val_rows)} imagenes, {len(val_tracks)} tracks',
        'Matriculas compartidas entre train/val: 0',
        'Tracks compartidos entre train/val: 0',
        'Distribucion por escenario/layout (tracks train / val):',
    ]
    for stratum, overall in sorted(original_strata.items()):
        n_val = val_strata[stratum]
        summary.append(f'- {stratum[0]}/{stratum[1]}: {overall-n_val} / {n_val}')
    summary.extend([
        f'SHA256 original: {original_sha}',
        f'Manifiestos derivados: {base}/split_manifest.csv, train_manifest.csv, val_manifest.csv',
        f'Informe de grupos: {a.reports}/hr_split_groups.csv',
        'No se han copiado ni modificado imagenes o anotaciones.',
        'Manifest original intacto: ' + ('SI' if hash_file(source) == original_sha else 'NO'),
        'AVISO: Agrupar por texto evita coincidencias exactas de placa; no demuestra identidad de vehiculo.',
        'AVISO: Las etiquetas originales siguen sin verificacion humana exhaustiva.',
        '=== FINALIZADO ===',
    ])
    if hash_file(source) != original_sha:
        raise RuntimeError('El manifiesto original cambió durante la ejecución')
    atomic_text(a.reports / 'hr_split_summary.txt', '\n'.join(summary) + '\n')
    print('\n'.join(summary), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
