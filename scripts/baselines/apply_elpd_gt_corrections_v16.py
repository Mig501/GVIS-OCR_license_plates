#!/usr/bin/env python3
"""Aplica un parche de GT visualmente revisado a ELPD (735) y recalcula 4 baselines.

NO reentrena ni hace inferencia. NO altera el manifiesto/CSV/summaries originales.
Se evalua una VERSION NUEVA del GT, comparando cada prediccion guardada con su
nuevo GT. Para la referencia inicial se emplea el GT ya fusionado del CSV PARSeq
(80 auditadas), y se valida su coincidencia con los otros tres CSV.

Colocar el archivo en: scripts/baselines/apply_elpd_gt_corrections_v16.py
Colocar el parche en: configs/elpd_gt_corrections_v16.csv
Ejecucion SLURM: CPU, 2 cores, 4 GB, 00:10:00, sin argumentos adicionales.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
RAW_MANIFEST_REL = Path('results/audit/ELPD/eval_manifests/external_eval_manifest.csv')
PATCH_REL = Path('configs/elpd_gt_corrections_v16.csv')
OUTPUT_REL = Path('results/audit/ELPD/gt_corrections_v16')
EXPECTED_MANIFEST_SHA = '5854abd860c89a17e35ef225627b17725c60d23088197f280e34eb5272097f13'
EXPECTED_STRICT_SHA = 'c22ea5efd511ca82a3889ed08c2b79ae44c655bbfbddaa844f12d800169cfc1c'
EXPECTED_IMAGES = 735
EXPECTED_GT_COUNTS = Counter({'strict_verified': 76, 'strict_corrected': 4, 'filename_candidate': 653, 'external_verified': 2})
MODEL_PATHS = {
    'parseq': Path('results/baselines/ELPD/parseq_pretrained_full_735'),
    'svtrv2_tiny': Path('results/baselines/ELPD/svtrv2_tiny_full_735'),
    'svtrv2_small': Path('results/baselines/ELPD/svtrv2_small_full_735'),
    'mdiff4str_small': Path('results/baselines/ELPD/mdiff4str_small_full_735'),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        headers = list(rd.fieldnames or [])
        missing = required - set(headers)
        if missing:
            raise ValueError(f'Faltan columnas de {path}: {sorted(missing)}')
        rows = list(rd)
    return rows, headers


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def normalize(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', s.upper())


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    exact = sum(int(r['exact_match']) for r in rows)
    errors = sum(int(r['edit_distance']) for r in rows)
    chars = sum(int(r['gt_length']) for r in rows)
    return {
        'images': n,
        'exact_matches': exact,
        'incorrect_matches': n - exact,
        'EM_percent': round(100 * exact / n, 4) if n else None,
        'levenshtein_errors': errors,
        'gt_characters': chars,
        'CER_percent': round(100 * errors / chars, 4) if chars else None,
    }


def score_groups(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {group: score(group_rows) for group, group_rows in sorted(groups.items())}


def clean_ip(ip: str) -> str:
    p = Path(ip)
    if p.is_absolute() or '..' in p.parts or not ip.startswith('images/'):
        raise ValueError(f'Ruta de imagen no permitida: {ip!r}')
    return ip


def load_inputs(project: Path, manifest_path: Path, patch_path: Path):
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise ValueError('El SHA-256 del manifiesto ORIGINAL no es el validado por el preflight. No aplicar correcciones.')
    raw_rows, raw_headers = read_csv(manifest_path, {'image_path','text_normalized','scenario','label_review_status'})
    if len(raw_rows) != EXPECTED_IMAGES:
        raise ValueError(f'Se esperaban {EXPECTED_IMAGES} imágenes en el manifiesto original.')
    raw_by_ip = {}
    for r in raw_rows:
        ip = clean_ip(r['image_path'])
        if ip in raw_by_ip:
            raise ValueError(f'Duplicado en manifiesto: {ip}')
        raw_by_ip[ip] = r
    if Counter(r['scenario'] for r in raw_rows) != Counter({'train': 588, 'val': 74, 'test': 73}):
        raise ValueError('Los splits del manifiesto han cambiado.')

    patch_rows, _ = read_csv(patch_path, {'image_path','gt_actual','gt_corregido','justificacion'})
    if not patch_rows:
        raise ValueError('El parche está vacío.')
    patch = {}
    for r in patch_rows:
        ip = clean_ip(r['image_path'].strip())
        old = r['gt_actual'].strip()
        new = r['gt_corregido'].strip()
        if ip in patch:
            raise ValueError(f'Parche repetido: {ip}')
        if ip not in raw_by_ip:
            raise ValueError(f'Parche no encontrado en manifiesto: {ip}')
        if raw_by_ip[ip]['text_normalized'] != old:
            raise ValueError(f'GT de partida distinto para {ip}: {raw_by_ip[ip]["text_normalized"]!r} != {old!r}')
        if not new or new != normalize(new) or old == new:
            raise ValueError(f'GT corregido inválido: {ip}: {old!r} -> {new!r}')
        if not r['justificacion'].strip():
            raise ValueError(f'Falta justificación visual: {ip}')
        patch[ip] = dict(r, image_path=ip, gt_actual=old, gt_corregido=new)

    baseline_rows: dict[str, list[dict[str, str]]] = {}
    baseline_headers: dict[str, list[str]] = {}
    baseline_summaries: dict[str, dict[str, Any]] = {}
    for model, rel in MODEL_PATHS.items():
        directory = project / rel
        preds, headers = read_csv(directory / 'predictions.csv', {'image_path','scenario','gt_source','gt_candidate_original','ground_truth','prediction_raw','prediction_normalized','exact_match','edit_distance','gt_length','pred_length'})
        with (directory / 'summary.json').open('r', encoding='utf-8') as f:
            summ = json.load(f)
        if len(preds) != EXPECTED_IMAGES or len({r['image_path'] for r in preds}) != EXPECTED_IMAGES:
            raise ValueError(f'Predicciones incompletas/duplicadas: {model}')
        if summ['manifest_sha256'] != EXPECTED_MANIFEST_SHA or summ['gt_audit']['audited_manifest_sha256'] != EXPECTED_STRICT_SHA:
            raise ValueError(f'Hashes de manifiesto incompatibles: {model}')
        if summ['dataset_size'] != EXPECTED_IMAGES:
            raise ValueError(f'Resumen con tamaño distinto: {model}')
        if Counter(r['gt_source'] for r in preds) != EXPECTED_GT_COUNTS:
            raise ValueError(f'Orígenes GT incompatibles: {model}')
        for r in preds:
            ip = clean_ip(r['image_path'])
            if ip not in raw_by_ip or raw_by_ip[ip]['text_normalized'] != r['gt_candidate_original']:
                raise ValueError(f'Predicción y manifiesto original no encajan: {model} / {ip}')
            if r['scenario'] != raw_by_ip[ip]['scenario']:
                raise ValueError(f'Split distinto: {model} / {ip}')
            if normalize(r['prediction_raw']) != r['prediction_normalized']:
                raise ValueError(f'Normalización inconsistente: {model} / {ip}')
            gt = r['ground_truth']
            pred = r['prediction_normalized']
            dist = edit_distance(gt, pred)
            if (dist != int(r['edit_distance']) or len(gt) != int(r['gt_length']) or
                len(pred) != int(r['pred_length']) or int(gt == pred) != int(r['exact_match'])):
                raise ValueError(f'Métricas de partida no reproducibles: {model} / {ip}')
        metrics = score(preds)
        if (summ['exact_matches'] != metrics['exact_matches'] or
            summ['levenshtein_errors'] != metrics['levenshtein_errors'] or
            summ['gt_characters'] != metrics['gt_characters']):
            raise ValueError(f'Resumen no coincide con predicciones originales: {model}')
        baseline_rows[model] = preds
        baseline_headers[model] = headers
        baseline_summaries[model] = summ

    # Todos los modelos deben haber utilizado EXACTAMENTE el mismo GT fusionado.
    ref = {r['image_path']: r for r in baseline_rows['parseq']}
    for model, preds in baseline_rows.items():
        for r in preds:
            ip = r['image_path']
            if r['ground_truth'] != ref[ip]['ground_truth'] or r['gt_source'] != ref[ip]['gt_source']:
                raise ValueError(f'El GT fusionado difiere entre modelos: {model}/{ip}')
    for ip, r in patch.items():
        if ref[ip]['gt_source'].startswith('strict_'):
            raise ValueError(f'No se permiten cambios al test estricto auditado: {ip}')
        if ref[ip]['ground_truth'] != r['gt_actual']:
            raise ValueError(f'El GT fusionado no coincide con parche: {ip}')
    return raw_rows, raw_headers, patch, baseline_rows, baseline_headers, baseline_summaries, ref


def generate(project: Path, manifest_path: Path, patch_path: Path, out_dir: Path) -> None:
    print('=== GVIS: APLICAR CORRECCIONES GT ELPD V16 (SIN GPU) ===', flush=True)
    (raw_rows, raw_headers, patch, baseline_rows, baseline_headers,
     baseline_summaries, ref) = load_inputs(project, manifest_path, patch_path)
    print(f'Manifiesto original validado: {len(raw_rows)} imágenes', flush=True)
    print(f'Mismo GT inicial en cuatro modelos: OK; test estricto de 80 protegido', flush=True)
    print(f'Correcciones visuales verificadas: {len(patch)}', flush=True)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'El directorio de salida ya tiene archivos: {out_dir} (no sobrescribir)')

    # El original raw conserva text_original/filename. En esta versión, text_normalized
    # almacena el GT fusionado auditado (80) más las correcciones nuevas.
    merged = []
    for raw in raw_rows:
        r = dict(raw)
        ip = r['image_path']
        r['gt_original_dataset'] = r['text_normalized']
        r['gt_fuente_v16'] = ref[ip]['gt_source']
        r['gt_nota_revision_v16'] = ''
        r['text_normalized'] = ref[ip]['ground_truth']
        if ip in patch:
            r['text_normalized'] = patch[ip]['gt_corregido']
            r['gt_fuente_v16'] = 'manual_corrected_v16'
            r['gt_nota_revision_v16'] = patch[ip]['justificacion'].strip()
            r['label_review_status'] = 'verified'
        merged.append(r)

    manifest_headers = raw_headers + ['gt_original_dataset','gt_fuente_v16','gt_nota_revision_v16']
    manifest_out = out_dir / 'eval_manifest_gt_corrected_v16.csv'
    write_csv(manifest_out, merged, manifest_headers)
    applied_out = out_dir / 'correcciones_aplicadas_v16.csv'
    applied_headers = ['image_path','gt_actual','gt_corregido','justificacion','aplicada']
    write_csv(applied_out, [{**{k:p[k] for k in applied_headers if k!='aplicada'}, 'aplicada':'si'} for p in patch.values()], applied_headers)

    comparison_rows = []
    by_model = {}
    for model, preds in baseline_rows.items():
        updated = []
        changed_count = 0
        for original in preds:
            r = dict(original)
            ip = r['image_path']
            r['gt_anterior'] = r['ground_truth']
            r['correccion_gt_v16'] = 'si' if ip in patch else 'no'
            if ip in patch:
                new_gt = patch[ip]['gt_corregido']
                r['ground_truth'] = new_gt
                r['gt_source'] = 'manual_corrected_v16'
                r['edit_distance'] = edit_distance(new_gt, r['prediction_normalized'])
                r['exact_match'] = int(new_gt == r['prediction_normalized'])
                r['gt_length'] = len(new_gt)
                changed_count += 1
            updated.append(r)
        assert changed_count == len(patch)
        old_metrics = score(preds)
        new_metrics = score(updated)
        new_metrics_split = score_groups(updated, 'scenario')
        new_metrics_source = score_groups(updated, 'gt_source')
        delta_em = round(new_metrics['EM_percent'] - old_metrics['EM_percent'], 4)
        delta_cer = round(new_metrics['CER_percent'] - old_metrics['CER_percent'], 4)
        comparison_rows.append({
            'modelo': baseline_summaries[model].get('model', model),
            'slug': model,
            'imagenes': EXPECTED_IMAGES,
            'correcciones_gt': len(patch),
            'em_original_pct': old_metrics['EM_percent'],
            'em_corregido_pct': new_metrics['EM_percent'],
            'delta_em_pp': delta_em,
            'cer_original_pct': old_metrics['CER_percent'],
            'cer_corregido_pct': new_metrics['CER_percent'],
            'delta_cer_pp': delta_cer,
            'aciertos_antes': old_metrics['exact_matches'],
            'aciertos_despues': new_metrics['exact_matches'],
            'ediciones_antes': old_metrics['levenshtein_errors'],
            'ediciones_despues': new_metrics['levenshtein_errors'],
        })
        model_dir = out_dir / model
        pred_headers = baseline_headers[model] + ['gt_anterior','correccion_gt_v16']
        write_csv(model_dir / 'predictions_gt_corrected_v16.csv', updated, pred_headers)
        updated_summary = {
            'model': baseline_summaries[model]['model'],
            'dataset': f'ELPD 735, candidate GT + 80 strict audited + {len(patch)} manual corrections',
            'gt_revision': 'v16',
            'gt_correction_count': len(patch),
            'original_manifest_sha256': EXPECTED_MANIFEST_SHA,
            'original_audited_strict_sha256': EXPECTED_STRICT_SHA,
            'updated_manifest_sha256': sha256_file(manifest_out),
            'corrections_sha256': sha256_file(patch_path),
            'inference_repeated': False,
            'source_predictions': str(MODEL_PATHS[model] / 'predictions.csv'),
            'source_checkpoint_sha256': baseline_summaries[model]['checkpoint_sha256'],
            'baseline_metrics': old_metrics,
            'recalculated_metrics': new_metrics,
            'metrics_by_original_split': new_metrics_split,
            'metrics_by_gt_source': new_metrics_source,
            'warning': 'La mayoría de etiquetas no se han inspeccionado visualmente; métricas descriptivas, no test cerrado.',
        }
        (model_dir / 'summary_gt_corrected_v16.json').write_text(json.dumps(updated_summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        (model_dir / 'summary_gt_corrected_v16.txt').write_text(
            f'{baseline_summaries[model]["model"]} / ELPD 735 / GT corregido v16\n'
            f'GT revisados: {len(patch)}; EM: {new_metrics["EM_percent"]:.2f}% (antes {old_metrics["EM_percent"]:.2f}%)\n'
            f'CER: {new_metrics["CER_percent"]:.2f}% (antes {old_metrics["CER_percent"]:.2f}%)\n'
            f'No se ha realizado nueva inferencia.\n', encoding='utf-8')
        by_model[model] = {'before':old_metrics, 'after':new_metrics}
        print(f'{model}: EM {old_metrics["EM_percent"]:.2f}% -> {new_metrics["EM_percent"]:.2f}% | '
              f'CER {old_metrics["CER_percent"]:.2f}% -> {new_metrics["CER_percent"]:.2f}%', flush=True)

    write_csv(out_dir / 'comparacion_metricas_v16.csv', comparison_rows, list(comparison_rows[0]))
    report = {
        'version':'v16',
        'correction_count':len(patch),
        'corrected_image_paths':list(patch),
        'total_images':EXPECTED_IMAGES,
        'original_manifest_sha256':EXPECTED_MANIFEST_SHA,
        'corrected_manifest_sha256':sha256_file(manifest_out),
        'corrections_sha256':sha256_file(patch_path),
        'original_gt_sources':dict(EXPECTED_GT_COUNTS),
        'model_metrics':by_model,
        'original_test_strict_80_unchanged':True,
        'original_files_untouched':True,
        'filename_candidate_labels_without_a_correction':653-len(patch),
        'note':'Se mantienen las 80 etiquetas auditadas; revisar el resto antes de publicar métricas definitivas.',
    }
    (out_dir / 'recalculation_report_v16.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (out_dir / 'README.txt').write_text(
        'Correcciones visuales de GT ELPD v16\n'
        'El manifiesto original y los cuatro CSV originales NO se alteraron.\n'
        'eval_manifest_gt_corrected_v16.csv: 735 GT, 80 auditadas fusionadas + parche v16.\n'
        'correcciones_aplicadas_v16.csv: trazabilidad de los cambios.\n'
        'comparacion_metricas_v16.csv: métricas antes/después.\n'
        'Carpetas de modelos: CSV de predicciones re-evaluadas y resumen.\n'
        'No se hizo nueva inferencia. Las etiquetas restantes son provisionales.\n', encoding='utf-8')
    print('Resultado:', out_dir, flush=True)
    print('RESULTADO: GT V16 CREADO Y CUATRO BASELINES RECALCULADAS; ORIGINALES INTACTOS', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=PROJECT)
    ap.add_argument('--manifest', type=Path, default=None)
    ap.add_argument('--corrections', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    project = args.project_root.resolve()
    generate(project, args.manifest or project / RAW_MANIFEST_REL,
             args.corrections or project / PATCH_REL,
             args.out or project / OUTPUT_REL)


if __name__ == '__main__':
    main()
