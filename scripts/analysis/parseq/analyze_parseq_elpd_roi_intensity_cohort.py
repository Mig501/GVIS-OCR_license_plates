#!/usr/bin/env python3
"""PASO 16 | PARSeq / ELPD V17 | Intensidad global de intervencion ViT en cohorte.

Ruta en el proyecto:
  scripts/analysis/parseq/analyze_parseq_elpd_roi_intensity_cohort.py

Ejecutar con GVIS: Ejecutar en GPU, SIN argumentos.

Reutiliza los scripts existentes EN LA MISMA CARPETA:
  analyze_parseq_elpd_cohort_interventions.py       (Paso 6B, ROI/seleccion)
  analyze_parseq_elpd_visual_token_ablation.py      (Paso 10, inferencia)
  analyze_parseq_elpd_memory_spatial_swap.py        (Paso 14, mezcla)

Se prueban once valores alpha (0 a 1, paso 0.1) para dos intervenciones
sobre los tokens finales del encoder ViT: ROI binaria y ROI fraccional.
En cada condicion, alpha es EL MISMO para las 31 imagenes.

Para cada imagen se codifica por separado el original y la version con ROI
rellena de gris. Los tokens seleccionados se interpolan entre las dos memorias,
sin modificar parametros del modelo, imagen original, ni ground truth.

Cohorte elegida MANUALMENTE: 19 inserciones y 12 aciertos; 9 ROI excluidas.
Las tasas son DESCRIPTIVAS Y NO estiman el rendimiento en ELPD completo. No se
elige alpha por imagen. Cualquier seleccion de alpha realizada sobre esta misma
cohorte requiere validacion externa con imagenes no utilizadas para elegirlo.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from analyze_parseq_elpd_cohort_interventions import (
    PROJECT, ROI, EXCLUDED, DEFAULT_SELECTION, DEFAULT_IMAGES,
    DEFAULT_TORCH_HOME, load_selection, safe_image, load_img, variants,
    edit_distance, classify, sha256,
)
from analyze_parseq_elpd_visual_token_ablation import (
    patch_overlap_weights, forward_with_memory,
)
from analyze_parseq_elpd_memory_spatial_swap import swap_memory

OUT = PROJECT / 'results/analysis/PARSeq/ELPD/roi_intensity_cohort_v1'
REF_6B = PROJECT / 'results/analysis/PARSeq/ELPD/cohorte_intervenciones_gpu_v1/comparacion_intervenciones.csv'
ALPHAS = tuple(k / 10.0 for k in range(11))
MODES = ('roi_binaria', 'roi_fraccional')
STAGES = (('AR', 0), ('REF', 1))
GRID_H, GRID_W, PH, PW = 8, 16, 4, 8
N_VISUAL = GRID_H * GRID_W
EXPECTED_GROUPS = {'insercion': 19, 'correcta_control': 12}


def write_csv(dest: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f'Sin filas para {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def get_weights(box):
    overlaps = patch_overlap_weights(box, GRID_H, GRID_W, PH, PW)
    frac = [float(overlaps.get(i, 0.0)) for i in range(N_VISUAL)]
    hard = [float(w > 0) for w in frac]
    if not 0 < sum(hard) < N_VISUAL:
        raise RuntimeError(f'ROI sin parches seleccionables: {box}')
    return {'roi_binaria': hard, 'roi_fraccional': frac}


def load_prior_6b(path: Path):
    if not path.is_file():
        print('[AVISO] No se encontro el CSV del Paso 6B; se validara contra V17.', flush=True)
        return None
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    values = {(r['id'], r['variante']): r for r in rows}
    if len(values) != 93:
        raise RuntimeError('Referencia Paso 6B incompleta o duplicada (93 variantes esperadas)')
    return values


def prepare(selection_path: Path, image_root: Path):
    all_rows = load_selection(selection_path)
    selected = [r for r in all_rows if r['id'] in ROI]
    if len(all_rows) != 40 or len(selected) != 31 or len(EXCLUDED) != 9:
        raise RuntimeError('La cohorte no coincide con el Paso 6A/6B')
    if dict(Counter(r['group'] for r in selected)) != EXPECTED_GROUPS:
        raise RuntimeError('Se esperaban 19 inserciones y 12 controles')
    prepared, regions = [], []
    for r in selected:
        label, bbox = ROI[r['id']]
        p = safe_image(image_root, r['image_path'])
        img = load_img(p)
        gray = variants(img, bbox)['relleno_gris']
        masks = get_weights(bbox)
        prepared.append((r, img, gray, masks))
        regions.append({
            'id': r['id'], 'grupo': r['group'], 'GT': r['ground_truth'],
            'imagen': r['image_path'], 'sha256_imagen': sha256(p),
            'tipo_insercion': r['type'], 'nombre_roi': label,
            'roi_128x32': str(bbox),
            'n_patches_binarios': int(sum(masks['roi_binaria'])),
            'suma_solapamientos_fraccionales': round(sum(masks['roi_fraccional']), 5),
        })
    return prepared, regions


def row_for(item, mode: str, alpha: float, stage: str, prediction: dict, n_tokens: int, weight_sum: float):
    gt = item['ground_truth']
    pred = prediction['normalizada']
    return {
        'id': item['id'], 'grupo': item['group'], 'tipo_original': item['type'],
        'GT': gt, 'imagen': item['image_path'], 'condicion': mode,
        'alpha': round(alpha, 1), 'etapa': stage,
        'prediction_raw': prediction['raw'], 'prediction_norm': pred,
        'exact_match': int(gt == pred), 'ediciones': edit_distance(gt, pred),
        'categoria': classify(gt, pred),
        'eos_1based': prediction['eos_1based'] or '',
        'num_logits': prediction['n_logits'],
        'n_patches_intervenidos': n_tokens,
        'intensidad_total_equivalente': round(weight_sum, 6),
    }


def calculate_aggregates(rows: list[dict], selected: list[dict]):
    """Metricas de ALPHA FIJO por modo y etapa, comparadas por ID con original."""
    lookup = {(r['id'], r['condicion'], r['alpha'], r['etapa']): r for r in rows}
    if len(lookup) != len(rows):
        raise RuntimeError('Se duplicaron claves (id, condicion, alpha, etapa)')
    n_expected = 31 * len(STAGES) * (len(MODES) * len(ALPHAS) + 2)
    if len(rows) != n_expected:
        raise RuntimeError(f'Se esperaban {n_expected} filas, hay {len(rows)}')
    stat_rows, cases_rows = [], []
    for stage, _ in STAGES:
        for mode in MODES:
            for alpha in ALPHAS:
                current = []
                original = []
                for item in selected:
                    key = item['id']
                    cur = lookup[key, mode, alpha, stage]
                    ref = lookup[key, 'original', 0.0, stage]
                    current.append(cur)
                    original.append(ref)
                    cases_rows.append({
                        'id': key, 'grupo': item['group'], 'GT': item['ground_truth'],
                        'etapa': stage, 'modo': mode, 'alpha': alpha,
                        'original': ref['prediction_norm'], 'intervenido': cur['prediction_norm'],
                        'correcta_antes': ref['exact_match'], 'correcta_despues': cur['exact_match'],
                        'ediciones_antes': ref['ediciones'], 'ediciones_despues': cur['ediciones'],
                        'cambio_prediccion': int(ref['prediction_norm'] != cur['prediction_norm']),
                        'delta_ediciones': cur['ediciones'] - ref['ediciones'],
                        'categoria_antes': ref['categoria'], 'categoria_despues': cur['categoria'],
                    })
                ins = [(a, b) for a, b, item in zip(original, current, selected) if item['group'] == 'insercion']
                ctrl = [(a, b) for a, b, item in zip(original, current, selected) if item['group'] == 'correcta_control']
                corrected = sum(not a['exact_match'] and b['exact_match'] for a,b in ins)
                harmed = sum(a['exact_match'] and not b['exact_match'] for a,b in ctrl)
                changed_bad = sum(b['ediciones'] > a['ediciones'] for a,b in ins)
                changed_good = sum(b['ediciones'] < a['ediciones'] for a,b in ins)
                stat_rows.append({
                    'etapa': stage, 'modo': mode, 'alpha': alpha,
                    'correctas_total': sum(b['exact_match'] for b in current),
                    'exact_match_total_pct': round(100 * sum(b['exact_match'] for b in current)/31, 4),
                    'ediciones_total': sum(b['ediciones'] for b in current),
                    'correctas_inserciones': sum(b['exact_match'] for _,b in ins),
                    'inserciones_corregidas_vs_original': corrected,
                    'inserciones_mejoran_ediciones': changed_good,
                    'inserciones_empeoran_ediciones': changed_bad,
                    'inserciones_puras_restantes': sum(b['categoria'] == 'insercion_pura' for _,b in ins),
                    'correctas_controles': sum(b['exact_match'] for _,b in ctrl),
                    'controles_perjudicados_vs_original': harmed,
                    'controles_mejoran_ediciones': sum(b['ediciones'] < a['ediciones'] for a,b in ctrl),
                    'controles_empeoran_ediciones': sum(b['ediciones'] > a['ediciones'] for a,b in ctrl),
                    'cambios_prediccion_total': sum(a['prediction_norm'] != b['prediction_norm'] for a,b in zip(original,current)),
                    'ganancia_neta_de_aciertos_vs_original': sum(b['exact_match'] - a['exact_match'] for a,b in zip(original,current)),
                })
    return stat_rows, cases_rows



def plot_tradeoff(ref_stats: list[dict], target: Path):
    """Curvas de correcciones y perjuicios por intensidad; mismas ROI y alphas."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1120, 575
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)
    def font(size):
        try:
            return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        except OSError:
            return ImageFont.load_default()
    title, regular, small = font(22), font(17), font(14)
    draw.text((55, 22), 'Paso 16: intensidad de tokens vs. correcciones / controles',
              font=title, fill=(32, 48, 64))
    draw.text((55, 58), 'Resultados refinados en 19 inserciones y 12 controles; cohorte seleccionada',
              font=small, fill=(80, 91, 100))
    lx, rx, ty, by = 100, 810, 110, 465
    def px(a): return lx + int(a * (rx-lx))
    def py(n): return by - int(n/19 * (by-ty))
    for tick in (0, 5, 10, 15, 19):
        y=py(tick)
        draw.line((lx,y,rx,y),fill=(226,231,235),width=2)
        draw.text((lx-45,y-10),str(tick),font=small,fill=(74,86,99))
    for a in ALPHAS:
        x=px(a)
        draw.line((x,ty,x,by),fill=(244,246,247),width=1)
        draw.text((x-15,by+12),f'{a:.1f}',font=small,fill=(74,86,99))
    palette = {
        ('roi_binaria','inserciones_corregidas_vs_original'):(33,122,181),
        ('roi_fraccional','inserciones_corregidas_vs_original'):(15,146,119),
        ('roi_binaria','controles_perjudicados_vs_original'):(199,96,63),
        ('roi_fraccional','controles_perjudicados_vs_original'):(155,88,143),
    }
    for mode in MODES:
        mode_rows=sorted((r for r in ref_stats if r['modo']==mode),key=lambda r:r['alpha'])
        for variable in ('inserciones_corregidas_vs_original','controles_perjudicados_vs_original'):
            color=palette[(mode,variable)]
            pts=[(px(r['alpha']),py(r[variable])) for r in mode_rows]
            draw.line(pts,fill=color,width=4)
            for x,y in pts:
                draw.ellipse((x-4,y-4,x+4,y+4), fill=color)
    legend=[
        ('Corregidas / ROI binaria', palette[('roi_binaria','inserciones_corregidas_vs_original')]),
        ('Corregidas / ROI fraccional', palette[('roi_fraccional','inserciones_corregidas_vs_original')]),
        ('Danadas / ROI binaria', palette[('roi_binaria','controles_perjudicados_vs_original')]),
        ('Danadas / ROI fraccional', palette[('roi_fraccional','controles_perjudicados_vs_original')]),
    ]
    for i,(label,cl) in enumerate(legend):
        y=135+i*54
        draw.line((842,y,874,y),fill=cl,width=5)
        draw.text((885,y-12),label,font=small,fill=(48,58,68))
    draw.text((342,520),'Intensidad alpha (0=original; 1=donante gris)',font=regular,fill=(32,48,64))
    draw.text((55,550),'Estudio exploratorio in-sample: NO valida alpha para otras imagenes.',font=small,fill=(100,104,112))
    target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target,optimize=True)


def report_baseline(rows, stage):
    r=[x for x in rows if x['etapa']==stage and x['condicion']=='original']
    return {
        'n': len(r), 'correctas': sum(x['exact_match'] for x in r),
        'ediciones': sum(x['ediciones'] for x in r),
        'correctas_inserciones': sum(x['exact_match'] for x in r if x['grupo']=='insercion'),
        'correctas_controles': sum(x['exact_match'] for x in r if x['grupo']=='correcta_control'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGES)
    ap.add_argument('--reference-6b', type=Path, default=REF_6B)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--prepare-only', action='store_true', help='Validar imagenes y ROI, sin GPU')
    args = ap.parse_args()

    print('=== PASO 16 | PARSeq ELPD V17 | BARRIDO DE INTENSIDAD EN COHORTE ===', flush=True)
    print('31 imagenes: 19 inserciones + 12 controles | alpha fijo por experimento', flush=True)
    print('Modos:', ', '.join(MODES), '| intensidades:', ', '.join(f'{a:.1f}' for a in ALPHAS), flush=True)
    if (args.out / 'resumen.json').exists():
        raise FileExistsError(f'No se sobrescribe un experimento completo: {args.out}')
    prepared, regions = prepare(args.selection, args.image_root)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / 'regiones.csv', regions)
    if args.prepare_only:
        print('PREFLIGHT OK: 31 imagenes y ROI comprobadas; sin GPU.', flush=True)
        print('SALIDA:', args.out, flush=True)
        return

    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('Se necesita GPU CUDA; utilizar GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core = model.model
    if not core.decode_ar or int(core.refine_iters)!=1:
        raise RuntimeError('Se esperaba PARSeq AR con refine_iters=1')
    if tuple(int(v) for v in model.hparams.img_size) != (32,128):
        raise RuntimeError('El checkpoint debe utilizar 32x128')
    pe = core.encoder.patch_embed
    if tuple(int(v) for v in pe.patch_size)!=(PH,PW) or tuple(int(v) for v in pe.grid_size)!=(GRID_H,GRID_W):
        raise RuntimeError('El checkpoint debe usar patches 4x8 y cuadricula 8x16')
    to_tensor = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3,(0.5,)*3)])
    prior = load_prior_6b(args.reference_6b)

    records = []
    start = time.perf_counter()
    for i,(item, image, gray, masks) in enumerate(prepared, 1):
        x = to_tensor(image).unsqueeze(0).cuda()
        gx = to_tensor(gray).unsqueeze(0).cuda()
        with torch.inference_mode():
            original_memory = core.encode(x).detach()
            donor_memory = core.encode(gx).detach()
        if tuple(original_memory.shape) != (1,N_VISUAL,384):
            raise RuntimeError(f'Memoria ViT inesperada: {original_memory.shape}')
        if donor_memory.shape != original_memory.shape:
            raise RuntimeError('Memoria ViT gris con dimensiones distintas')
        baseline={}
        for name, tensor in (('original',x), ('pixel_gris',gx)):
            baseline[name]={}
            for stage,nref in STAGES:
                pred=forward_with_memory(model,core,tensor,None,nref)
                baseline[name][stage]=pred
                records.append(row_for(item,name,0.0 if name=='original' else 1.0,stage,pred,0,0.0))
        if baseline['original']['REF']['normalizada'] != item['prediction_normalized']:
            raise RuntimeError(f'{item["id"]}: la baseline REF no reproduce V17')
        if prior is not None:
            for name,old_name in (('original','original'),('pixel_gris','relleno_gris')):
                old=prior[(item['id'],old_name)]
                for stage,col in (('AR','AR'),('REF','refinado')):
                    if baseline[name][stage]['normalizada'] != old[col]:
                        raise RuntimeError(f'{item["id"]} {stage}: no reproduce Paso 6B ({name})')
        output_alpha1 = {}
        for mode in MODES:
            mask=masks[mode]
            for alpha in ALPHAS:
                strengths = [alpha * w for w in mask]
                with torch.inference_mode():
                    donor_swap = swap_memory(original_memory,donor_memory,strengths)
                for stage,nref in STAGES:
                    pred=forward_with_memory(model,core,x,donor_swap,nref)
                    if alpha==0.0 and (pred['raw'] != baseline['original'][stage]['raw'] or
                                       pred['eos_1based'] != baseline['original'][stage]['eos_1based']):
                        raise RuntimeError(f'{item["id"]} {mode} alpha=0: no reproduce original {stage}')
                    if alpha==1.0 and mode=='roi_binaria':
                        output_alpha1[stage]=pred['normalizada']
                    records.append(row_for(item,mode,alpha,stage,pred,sum(s>0 for s in strengths),sum(strengths)))
        print(f'[{i:02d}/31] {item["id"]} | original={baseline["original"]["REF"]["normalizada"]}'
              f' | ROI 100%={output_alpha1["REF"]}', flush=True)

    stats, case_changes = calculate_aggregates(records,[p[0] for p in prepared])
    ref_stats = [s for s in stats if s['etapa']=='REF']
    baseline_ar = report_baseline(records,'AR')
    baseline_ref = report_baseline(records,'REF')
    if baseline_ref['correctas_inserciones'] != 0 or baseline_ref['correctas_controles'] != 12:
        raise RuntimeError('La cohorte REF no reproduce 19 errores y 12 controles correctos')
    if baseline_ar['correctas_inserciones'] != 1 or baseline_ar['correctas_controles'] != 11:
        raise RuntimeError('La cohorte AR no reproduce el Paso 12')
    write_csv(args.out/'comparacion_intensidades_cohorte.csv',records)
    write_csv(args.out/'metricas_por_alpha.csv',stats)
    write_csv(args.out/'cambios_por_caso.csv',case_changes)
    plot_tradeoff(ref_stats, args.out/'curvas'/'compromiso_inserciones_controles_REF.png')

    best_by_mode = {}
    for mode in MODES:
        candidates = [r for r in ref_stats if r['modo'] == mode]
        # Seleccion descriptiva en la misma cohorte, NO validacion independiente.
        ordered = sorted(candidates,key=lambda r:(-r['ganancia_neta_de_aciertos_vs_original'],
                        r['controles_perjudicados_vs_original'],r['alpha']))
        best_by_mode[mode] = {
            'alpha_exploratorio':ordered[0]['alpha'],
            'correctas_total':ordered[0]['correctas_total'],
            'correcciones_inserciones':ordered[0]['inserciones_corregidas_vs_original'],
            'controles_perjudicados':ordered[0]['controles_perjudicados_vs_original'],
            'nota':'Seleccion IN-SAMPLE; no es rendimiento externo ni recomendacion desplegable.',
        }
    summary = {
        'paso':16,'modelo':'PARSeq oficial pretrained', 'dataset':'ELPD V17',
        'n_cohorte_total':40,'n_con_ROI':31,'n_inserciones':19,'n_controles':12,'n_excluidas':9,
        'modos':list(MODES),'alphas':list(ALPHAS),'n_etapas':2,
        'n_predicciones':len(records),'esperadas':31*2*(len(ALPHAS)*len(MODES)+2),
        'referencia_paso_6b':'reproducida' if prior is not None else 'ausente; V17 reproducida',
        'baseline_AR':baseline_ar,'baseline_REF':baseline_ref,
        'metricas_REF_por_alpha':ref_stats,
        'mejor_resultado_en_la_propia_cohorte_solo_exploratorio':best_by_mode,
        'pesos_modificados':False,'GT_modificado':False,
        'duracion_segundos':round(time.perf_counter()-start,2),
        'limitaciones':[
            'Las ROI son manuales y la cohorte esta enriquecida en inserciones: no estima EM global de ELPD.',
            'Buscar el mejor alpha sobre esta cohorte es un ajuste in-sample, no una evaluacion independiente.',
            'El modelo real no sabe donde esta la ROI ni que alpha es mejor sin un procedimiento adicional.',
            'Los tokens del ViT tienen contexto global y los hibridos pueden quedar fuera de distribucion.',
            'Los parches de borde pueden rozar caracteres reales.',
            'Comparacion Pareada sobre las mismas 31 imagenes; sin entrenamiento.',
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n=== RESULTADOS REF POR ALPHA (31 IMAGENES) ===',flush=True)
    for r in ref_stats:
        print(f'{r["modo"]:15} alpha={r["alpha"]:.1f} '
              f'corregidas={r["inserciones_corregidas_vs_original"]:2}/19 '
              f'controles_danados={r["controles_perjudicados_vs_original"]:2}/12 '
              f'aciertos={r["correctas_total"]:2}/31', flush=True)
    print(f'RESULTADO: 31 originales reproducidos; {len(records)} predicciones AR/REF exportadas',flush=True)
    print('SALIDA:',args.out,flush=True)


if __name__ == '__main__':
    main()
