#!/usr/bin/env python3
"""PASO 12 | PARSeq + ELPD V17 | MHA2 en la cohorte de 31 matriculas.

Ubicacion: scripts/analysis/parseq/analyze_parseq_elpd_mha2_cohort.py
Ejecutar SIN argumentos mediante GVIS: Ejecutar en GPU.

Reutiliza deliberadamente el codigo PROBADO en los pasos 6B y 11. Ambos
scripts deben encontrarse EN ESTA MISMA CARPETA:
  analyze_parseq_elpd_cohort_interventions.py
  analyze_parseq_elpd_mha2_masking.py

Por cada imagen se comparan AR (0 refinamientos) y REF (1 refinamiento):
 - original
 - mha2_mascara_vacia (placebo, debe ser identico a original)
 - mha2_objetivo_bloqueo (mascara dura sobre MHA2)
 - mha2_objetivo_atenuacion (sesgo de -3 en logits de MHA2)
 - pixel_gris (comparacion con paso 6B)
 - pixel_desenfoque (comparacion con paso 6B)

Los 31 casos son 19 inserciones y 12 controles correctos; otros 9 de la
cohorte original no tienen una ROI suficientemente segura y NO se intervienen.
Ninguna condicion MHA2 cambia los pixeles ni los pesos del encoder/decoder.

La cohorte esta enriquecida en errores y sus ROI son preliminares:
las tasas resultantes NO son una estimacion del rendimiento global de ELPD.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

from analyze_parseq_elpd_cohort_interventions import (
    ROI, EXCLUDED, DEFAULT_SELECTION, DEFAULT_IMAGES, DEFAULT_TORCH_HOME,
    load_selection, safe_image, load_img, variants, edit_distance, classify, sha256,
)
from analyze_parseq_elpd_mha2_masking import (
    infer, roi_indices, check_noop, OVERLAP_THRESHOLD, SOFT_PENALTY,
)

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/mha2_cohort_v1'
DEFAULT_REFERENCE = PROJECT / 'results/analysis/PARSeq/ELPD/cohorte_intervenciones_gpu_v1/comparacion_intervenciones.csv'
CONDITIONS = (
    'original', 'mha2_mascara_vacia', 'mha2_objetivo_bloqueo',
    'mha2_objetivo_atenuacion', 'pixel_gris', 'pixel_desenfoque',
)
STAGES = (('AR', 0), ('REF', 1))


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError('No hay registros que escribir')
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


def load_prior_results(path: Path):
    """Una ausencia del CSV de paso 6B se anuncia, pero no detiene el trabajo."""
    if not path.is_file():
        return None
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    data = {(r['id'], r['variante']): r for r in rows}
    if len(data) != 93:
        raise RuntimeError('La referencia 6B no contiene las 93 variantes esperadas')
    return data


def conditions_for(image, roi):
    """Usa identico preprocesamiento visual que el paso 6B."""
    prepared = variants(image, roi)
    return [
        ('original', prepared['original'], 'none', 'original'),
        ('mha2_mascara_vacia', prepared['original'], 'noop', 'placebo'),
        ('mha2_objetivo_bloqueo', prepared['original'], 'hard', 'MHA2'),
        ('mha2_objetivo_atenuacion', prepared['original'], 'soft', 'MHA2'),
        ('pixel_gris', prepared['relleno_gris'], 'none', 'pixel'),
        ('pixel_desenfoque', prepared['desenfoque'], 'none', 'pixel'),
    ]


def summarize(records, selection):
    by_id_condition_stage = {(r['id'], r['condicion'], r['etapa']): r for r in records}
    stats = {}
    changes = []
    for stage, _ in STAGES:
        for group in ('insercion', 'correcta_control'):
            group_rows = [s for s in selection if s['group'] == group]
            for condition in CONDITIONS:
                subset = [by_id_condition_stage[s['id'], condition, stage] for s in group_rows]
                baselines = [by_id_condition_stage[s['id'], 'original', stage] for s in group_rows]
                key = f'{stage}|{group}|{condition}'
                improved = sum(r['ediciones'] < b['ediciones'] for r, b in zip(subset, baselines))
                worsened = sum(r['ediciones'] > b['ediciones'] for r, b in zip(subset, baselines))
                stats[key] = {
                    'n': len(subset),
                    'correctas': sum(r['exact_match'] for r in subset),
                    'suma_ediciones': sum(r['ediciones'] for r in subset),
                    'inserciones_puras': sum(r['categoria'] == 'insercion_pura' for r in subset),
                    'corrige_a_EM': sum(not b['exact_match'] and r['exact_match'] for r, b in zip(subset, baselines)),
                    'perjudica_EM': sum(b['exact_match'] and not r['exact_match'] for r, b in zip(subset, baselines)),
                    'mejora_ediciones': improved,
                    'empeora_ediciones': worsened,
                    'predicciones_cambiadas': sum(r['prediction_norm'] != b['prediction_norm'] for r, b in zip(subset, baselines)),
                }
    for item in selection:
        rid = item['id']
        for stage, _ in STAGES:
            baseline = by_id_condition_stage[rid, 'original', stage]
            for condition in CONDITIONS[1:]:
                r = by_id_condition_stage[rid, condition, stage]
                changes.append({
                    'id': rid, 'grupo': item['group'], 'gt': item['ground_truth'],
                    'etapa': stage, 'condicion': condition,
                    'original': baseline['prediction_norm'],
                    'intervenido': r['prediction_norm'],
                    'em_antes': baseline['exact_match'], 'em_despues': r['exact_match'],
                    'ediciones_antes': baseline['ediciones'], 'ediciones_despues': r['ediciones'],
                    'categoria_antes': baseline['categoria'], 'categoria_despues': r['categoria'],
                    'cambio': int(baseline['prediction_norm'] != r['prediction_norm']),
                    'eos_antes': baseline['eos_1based'], 'eos_despues': r['eos_1based'],
                })
    return stats, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGES)
    parser.add_argument('--reference-6b', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true', help='Validar cohorte, ROI e imagenes sin GPU')
    args = parser.parse_args()

    print('=== PASO 12 | PARSEQ ELPD V17 | MHA2 EN COHORTE ===', flush=True)
    if (args.out / 'resumen.json').exists():
        raise FileExistsError(f'No sobrescribir resultados completos: {args.out}')

    selection = load_selection(args.selection)
    chosen = [s for s in selection if s['id'] in ROI]
    if len(chosen) != 31 or Counter(s['group'] for s in chosen) != {'insercion':19, 'correcta_control':12}:
        raise RuntimeError('La seleccion no corresponde a 19 errores + 12 controles')
    if len(EXCLUDED) != 9 or len(set(s['id'] for s in selection) & set(EXCLUDED)) != 9:
        raise RuntimeError('Exclusiones inconsistentes con la cohorte del paso 6')

    sources = []
    regions = []
    for item in chosen:
        name = item['id']
        label, bbox = ROI[name]
        patches = roi_indices(bbox, OVERLAP_THRESHOLD)
        path = safe_image(args.image_root, item['image_path'])
        image = load_img(path)
        sources.append((item, image, patches, bbox))
        regions.append({
            'id':name,'grupo':item['group'],'imagen':item['image_path'],
            'region':label,'roi_128x32':str(bbox),
            'indices_patches':' '.join(str(i) for i,_ in patches),
            'n_patches':len(patches), 'sha256_imagen':sha256(path),
        })

    print(f'Cohorte: {len(selection)} | ROI: {len(chosen)} | inserciones: 19 | controles: 12', flush=True)
    print(f'Condiciones: {len(CONDITIONS)} por imagen | etapas: AR + REF', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'regiones.csv', regions)
    if args.prepare_only:
        print('PREFLIGHT OK: 31 imagenes y ROI comprobadas. No se ha utilizado GPU.', flush=True)
        print('SALIDA:', args.out, flush=True)
        return

    import torch
    from torchvision import transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError('No hay CUDA disponible: ejecutar con GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core = model.model
    if not core.decode_ar or int(core.refine_iters) != 1:
        raise RuntimeError('Se necesita PARSeq oficial con decode_ar=True y refine_iters=1')
    if tuple(int(v) for v in model.hparams.img_size) != (32,128):
        raise RuntimeError('Resolucion de PARSeq distinta de 32x128')
    patch = core.encoder.patch_embed
    if tuple(int(v) for v in patch.patch_size) != (4,8) or tuple(int(v) for v in patch.grid_size) != (8,16):
        raise RuntimeError('La cuadricula ViT no es 8x16')
    if len(core.decoder.layers) != 1 or not isinstance(core.decoder.layers[0].cross_attn, torch.nn.MultiheadAttention):
        raise RuntimeError('No se reconoce la MHA2 oficial')
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(0.5,0.5)])
    prior = load_prior_results(args.reference_6b)
    if prior is None:
        print('AVISO: no se encontro el CSV del Paso 6B; verificaremos las 31 baselines V17 igualmente.', flush=True)
    else:
        print('Referencia del Paso 6B: 93 variantes encontradas.', flush=True)

    records = []
    t0 = time.perf_counter()
    for ix, (item, image, patches, bbox) in enumerate(sources, 1):
        rid, gt = item['id'], item['ground_truth']
        patch_ids = [i for i,_ in patches]
        by_condition = {}
        for name, mod_image, hook_mode, family in conditions_for(image, bbox):
            tensor = to_tensor(mod_image).unsqueeze(0).cuda()
            by_condition[name] = {}
            for stage, nref in STAGES:
                result = infer(model, core, tensor, patch_ids, hook_mode, nref)
                by_condition[name][stage] = result
                records.append({
                    'id':rid,'grupo':item['group'],'tipo_original':item['type'],
                    'imagen':item['image_path'],'GT':gt,'region':ROI[rid][0],
                    'condicion':name,'familia':family,'etapa':stage,
                    'prediction_raw':result['raw'],'prediction_norm':result['pred'],
                    'exact_match':int(result['pred']==gt),
                    'ediciones':edit_distance(gt,result['pred']),
                    'categoria':classify(gt,result['pred']),
                    'eos_1based':result['eos'] or '',
                    'num_logits':result['num_logits'],
                    'n_patches_objetivo':len(patch_ids),
                    'masa_atencion_media_roi':round(result['attn_mass_target'],8) if result['attn_mass_target'] is not None else '',
                    'masa_atencion_max_roi':round(result['attn_mass_max'],8),
                })

        if by_condition['original']['REF']['pred'] != item['prediction_normalized']:
            raise RuntimeError(f'{rid}: no reproduce prediccion refinada V17: '
                               f'{by_condition["original"]["REF"]["pred"]} vs {item["prediction_normalized"]}')
        for stage, _ in STAGES:
            check_noop(by_condition['original'][stage], by_condition['mha2_mascara_vacia'][stage], rid, stage)
            if prior is not None:
                for candidate, previous in (('original','original'),('pixel_gris','relleno_gris'),('pixel_desenfoque','desenfoque')):
                    row = prior[(rid,previous)]
                    previous_pred = row['AR'] if stage=='AR' else row['refinado']
                    if by_condition[candidate][stage]['pred'] != previous_pred:
                        raise RuntimeError(f'{rid} {stage} {candidate}: NO reproduce Paso 6B: '
                                           f'{by_condition[candidate][stage]["pred"]} vs {previous_pred}')

        print(f'[{ix:02d}/31] {rid} GT={gt} | original={by_condition["original"]["REF"]["pred"]}'
              f' | bloqueo={by_condition["mha2_objetivo_bloqueo"]["REF"]["pred"]}'
              f' | atenuacion={by_condition["mha2_objetivo_atenuacion"]["REF"]["pred"]}', flush=True)

    if len(records) != 31*len(CONDITIONS)*len(STAGES):
        raise RuntimeError(f'Recuento de predicciones inesperado: {len(records)}')
    stats, changes = summarize(records, chosen)
    write_csv(args.out/'comparacion_condiciones.csv', records)
    write_csv(args.out/'cambios_vs_original.csv', changes)
    summary = {
        'paso':12,'modelo':'PARSeq pretrained oficial','dataset':'ELPD V17',
        'cohorte_total':40,'cohorte_con_roi':31,'errores_insercion':19,
        'controles_correctos':12,'no_intervenidas':9,
        'n_predicciones':len(records),
        'n_condiciones':len(CONDITIONS),'etapas':['AR','REF'],
        'reproduccion_v17_ref':31,
        'placebo_mascara_vacia':'OK (raw y probabilidades)',
        'comparacion_con_paso_6b':'OK' if prior is not None else 'No disponible',
        'umbral_patch':OVERLAP_THRESHOLD,'penalizacion_atencion':SOFT_PENALTY,
        'resumen_por_etapa_grupo_condicion':stats,
        'segundos':round(time.perf_counter()-t0,2),
        'checkpoint_modificado':False,'GT_modificado':False,
        'limitaciones':[
            'ROI propuestas; un patch seleccionado puede rozar caracteres.',
            'Las 31 imagenes no son una muestra aleatoria de ELPD.',
            'Los 12 controles son matriculas correctas, no ROI placebo por imagen.',
            'La mascara se aplica sobre MHA2; no se modifican el encoder ni sus tokens.',
            'Los tokens del ViT contienen informacion distribuida de varios patches.',
            'Bloquear la atencion demuestra sensibilidad al acceso visual, no causa unica del fallo.',
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n=== RESUMEN REF (tras 1 refinamiento) ===',flush=True)
    for group in ('insercion','correcta_control'):
        for condition in CONDITIONS:
            s=stats[f'REF|{group}|{condition}']
            print(f'{group:16} {condition:24} '
                  f'correctas={s["correctas"]:2}/{s["n"]} '
                  f'inserciones_puras={s["inserciones_puras"]:2} '
                  f'corrige={s["corrige_a_EM"]:2} perjudica={s["perjudica_EM"]:2}',flush=True)
    print(f'RESULTADO: 31 originales reproducidos; {len(records)} comparaciones AR/REF exportadas',flush=True)
    print('SALIDA:',args.out,flush=True)


if __name__=='__main__':
    main()
