#!/usr/bin/env python3
"""Aplica un parche de GT visualmente revisado a ELPD (735) y recalcula 4 baselines.

NO reentrena ni hace inferencia. NO altera el manifiesto/CSV/summaries originales.
Se evalua una VERSION NUEVA del GT, comparando cada prediccion guardada con su
nuevo GT. Para la referencia inicial se emplea el GT ya fusionado del CSV PARSeq
(80 auditadas), y se valida su coincidencia con los otros tres CSV.

Colocar el archivo en: scripts/baselines/apply_elpd_gt_corrections_v17.py
Colocar el parche en: configs/elpd_gt_corrections_v17.csv
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
PATCH_REL = Path('configs/elpd_gt_corrections_v17.csv')
OUTPUT_REL = Path('results/audit/ELPD/gt_corrections_v17')
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


EXCLUSIONS_REL = Path('configs/elpd_gt_exclusions_v17.csv')
EXPECTED_CORRECTIONS = 39
EXPECTED_EXCLUSIONS = 2

def read_exclusions(exclusions_path: Path, ref, patch):
    rows, _ = read_csv(exclusions_path, {'image_path','gt_actual','decision','justificacion'})
    exclusions = {}
    for row in rows:
        ip = clean_ip(row['image_path'].strip())
        if ip in exclusions or ip not in ref:
            raise ValueError(f'Exclusión duplicada o desconocida: {ip}')
        if ip in patch or ref[ip]['gt_source'].startswith('strict_'):
            raise ValueError(f'No se permite excluir una imagen corregida o test estricto: {ip}')
        if row['decision'].strip()!='NO_VERIFICABLE' or row['gt_actual']!=ref[ip]['ground_truth'] or not row['justificacion'].strip():
            raise ValueError(f'Exclusión inválida: {ip}')
        exclusions[ip]=row
    if len(exclusions)!=EXPECTED_EXCLUSIONS:
        raise ValueError(f'Esperadas {EXPECTED_EXCLUSIONS} exclusiones, recibidas {len(exclusions)}')
    return exclusions


def generate(project: Path, manifest_path: Path, patch_path: Path, exclusions_path: Path, out_dir: Path) -> None:
    print('=== GVIS: CIERRE MANUAL GT ELPD V17 (SIN GPU) ===', flush=True)
    (raw_rows, raw_headers, patch, baseline_rows, baseline_headers,
     baseline_summaries, ref) = load_inputs(project, manifest_path, patch_path)
    exclusions = read_exclusions(exclusions_path, ref, patch)
    if len(patch)!=EXPECTED_CORRECTIONS:
        raise ValueError(f'Esperadas {EXPECTED_CORRECTIONS} correcciones, recibidas {len(patch)}')
    if set(patch) & set(exclusions):
        raise ValueError('Hay imágenes corregidas y excluidas simultáneamente')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'El directorio de salida ya contiene archivos: {out_dir}')
    print('Manifiesto original: 735; test estricto: 80 protegido',flush=True)
    print(f'Correcciones GT acumuladas: {len(patch)}',flush=True)
    print(f'Sin GT verificable (NO se puntúan en subset validado): {len(exclusions)}',flush=True)

    merged=[]
    for raw in raw_rows:
        r=dict(raw)
        ip=r['image_path']
        r['gt_original_dataset']=r['text_normalized']
        r['gt_fuente_v17']=ref[ip]['gt_source']
        r['gt_nota_revision_v17']=''
        r['text_normalized']=ref[ip]['ground_truth']
        r['gt_validation_status_v17']='no_verificable' if ip in exclusions else ('corregido' if ip in patch else 'confirmado_o_estricto')
        r['include_in_verified_eval_v17']='no' if ip in exclusions else 'si'
        if ip in patch:
            r['text_normalized']=patch[ip]['gt_corregido']
            r['gt_fuente_v17']='manual_corrected_v17'
            r['gt_nota_revision_v17']=patch[ip]['justificacion'].strip()
            r['label_review_status']='verified'
        if ip in exclusions:
            r['gt_nota_revision_v17']=exclusions[ip]['justificacion'].strip()
            r['label_review_status']='unverifiable'
        merged.append(r)
    manifest_headers=raw_headers+['gt_original_dataset','gt_fuente_v17','gt_nota_revision_v17','gt_validation_status_v17','include_in_verified_eval_v17']
    manifest_out=out_dir/'eval_manifest_gt_corrected_v17.csv'
    write_csv(manifest_out,merged,manifest_headers)
    verified_manifest=out_dir/'eval_manifest_verified_v17.csv'
    write_csv(verified_manifest,[r for r in merged if r['include_in_verified_eval_v17']=='si'],manifest_headers)
    assert len(merged)==735 and len(merged)-len(exclusions)==733
    applied_headers=['image_path','gt_actual','gt_corregido','justificacion','aplicada']
    write_csv(out_dir/'correcciones_aplicadas_v17.csv',
              [{**p,'aplicada':'si'} for p in patch.values()],applied_headers)
    write_csv(out_dir/'exclusiones_no_verificables_v17.csv',list(exclusions.values()),
              ['image_path','gt_actual','decision','justificacion'])

    comp=[];summarized={}
    for model,preds in baseline_rows.items():
        updated=[]
        for original in preds:
            r=dict(original); ip=r['image_path']
            r['gt_anterior']=r['ground_truth']
            r['correccion_gt_v17']='si' if ip in patch else 'no'
            r['gt_validation_status_v17']='no_verificable' if ip in exclusions else ('corregido' if ip in patch else 'confirmado_o_estricto')
            r['include_in_verified_eval_v17']='no' if ip in exclusions else 'si'
            if ip in patch:
                new_gt=patch[ip]['gt_corregido']
                r['ground_truth']=new_gt
                r['gt_source']='manual_corrected_v17'
                r['edit_distance']=edit_distance(new_gt,r['prediction_normalized'])
                r['exact_match']=int(new_gt==r['prediction_normalized'])
                r['gt_length']=len(new_gt)
            updated.append(r)
        verified=[r for r in updated if r['include_in_verified_eval_v17']=='si']
        old_metrics=score(preds)
        full=score(updated)
        clean=score(verified)
        assert full['images']==735 and clean['images']==733
        comp.append({
            'modelo':baseline_summaries[model].get('model',model),'slug':model,
            'correcciones_gt':len(patch),'excluidas_no_verificables':len(exclusions),
            'imagenes_full':735,'em_original_pct':old_metrics['EM_percent'],
            'em_full_v17_pct':full['EM_percent'],'cer_original_pct':old_metrics['CER_percent'],
            'cer_full_v17_pct':full['CER_percent'],'aciertos_full_v17':full['exact_matches'],
            'ediciones_full_v17':full['levenshtein_errors'],
            'imagenes_verificadas':clean['images'],'em_verificado_v17_pct':clean['EM_percent'],
            'cer_verificado_v17_pct':clean['CER_percent'],
            'aciertos_verificados_v17':clean['exact_matches'],
            'ediciones_verificadas_v17':clean['levenshtein_errors'],
            'caracteres_gt_verificados':clean['gt_characters']
        })
        sub=out_dir/model
        pred_headers=baseline_headers[model]+['gt_anterior','correccion_gt_v17','gt_validation_status_v17','include_in_verified_eval_v17']
        write_csv(sub/'predictions_gt_corrected_v17.csv',updated,pred_headers)
        write_csv(sub/'predictions_verified_v17.csv',verified,pred_headers)
        report={
            'model':baseline_summaries[model]['model'],
            'dataset':'ELPD: 735 full; 733 validated (2 excluded)',
            'version':'v17','gt_corrections':len(patch),'no_verificables':len(exclusions),
            'original_manifest_sha256':EXPECTED_MANIFEST_SHA,
            'original_audited_strict_sha256':EXPECTED_STRICT_SHA,
            'updated_manifest_sha256':sha256_file(manifest_out),
            'verified_manifest_sha256':sha256_file(verified_manifest),
            'corrections_sha256':sha256_file(patch_path),
            'exclusions_sha256':sha256_file(exclusions_path),
            'source_checkpoint_sha256':baseline_summaries[model]['checkpoint_sha256'],
            'original_metrics_735':old_metrics,
            'full_metrics_v17_735':full,
            'verified_metrics_v17_733':clean,
            'verified_metrics_by_split':score_groups(verified,'scenario'),
            'inference_repeated':False,
            'caveat':'Evaluación descriptiva del conjunto completo original train/val/test; no prueba de generalización externa independiente.'
        }
        (sub/'summary_gt_corrected_v17.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (sub/'summary_gt_corrected_v17.txt').write_text(
            f'{report["model"]} / ELPD V17\n'
            f'735 imágenes: EM {full["EM_percent"]:.2f}%, CER {full["CER_percent"]:.2f}%\n'
            f'733 verificadas: EM {clean["EM_percent"]:.2f}%, CER {clean["CER_percent"]:.2f}%\n'
            f'39 GT corregidos; 2 no verificables excluidos de la evaluación validada.\n'
            f'Sin nueva inferencia.\n',encoding='utf-8')
        summarized[model]={'original':old_metrics,'full_735':full,'verified_733':clean}
        print(f'{model}: FULL EM {full["EM_percent"]:.2f}%, CER {full["CER_percent"]:.2f}% | '
              f'VERIFICADO (733) EM {clean["EM_percent"]:.2f}%, CER {clean["CER_percent"]:.2f}%',flush=True)
    write_csv(out_dir/'comparacion_metricas_v17.csv',comp,list(comp[0]))
    doc={
        'version':'v17','total_images':735,'verified_images':733,'unverifiable_images':2,
        'strict_verified_images':80,'cumulative_corrected_images':39,
        'manual_last_round':{'total':19,'corrected_images':4,'confirmed_images':13,'unverifiable_images':2},
        'corrected_paths':list(patch),'unverifiable_paths':list(exclusions),
        'original_untouched':True,'inference_repeated':False,
        'metrics':summarized,
        'note':'Dos GT no verificables permanecen provisionalmente en el conjunto completo de 735, pero se excluyen del subset validado de 733.'
    }
    (out_dir/'recalculation_report_v17.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out_dir/'README.txt').write_text(
        'ELPD V17 — cierre final de auditoría manual\n'
        '735 imágenes conservadas; 39 correcciones acumuladas; 2 GT no verificables.\n'
        'El manifiesto y las predicciones originales permanecen intactos.\n'
        'eval_manifest_gt_corrected_v17.csv: 735 filas con indicador de exclusión.\n'
        'eval_manifest_verified_v17.csv: 733 filas con GT confirmado.\n'
        'predictions_gt_corrected_v17.csv: 735, preserva referencias de los no verificables como provisionales.\n'
        'predictions_verified_v17.csv: 733, exclusión de dos GT no verificables.\n'
        'comparacion_metricas_v17.csv: versión completa y versión verificada.\n'
        'Los splits originales train/val/test conviven en las dos evaluaciones; no son un test externo.\n'
        'No se realizó inferencia ni entrenamiento.\n',encoding='utf-8')
    print('RESULTADO: GT V17 VALIDADO, 39 CORRECCIONES, 2 EXCLUSIONES, 733 GT VERIFICADOS; ORIGINALES INTACTOS',flush=True)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=PROJECT)
    ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--corrections',type=Path,default=None)
    ap.add_argument('--exclusions',type=Path,default=None)
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    project=args.project_root.resolve()
    generate(project,args.manifest or project/RAW_MANIFEST_REL,
             args.corrections or project/PATCH_REL,
             args.exclusions or project/EXCLUSIONS_REL,
             args.out or project/OUTPUT_REL)


if __name__=='__main__':
    main()
