#!/usr/bin/env python3
"""PASO 15 | PARSeq / ELPD V17 | Barrido de intensidad de intercambio de tokens ViT.

Ruta en proyecto:
  scripts/analysis/parseq/analyze_parseq_elpd_roi_intensity_sweep.py

Ejecucion: GVIS: Ejecutar en GPU, sin argumentos.
Requiere en la misma carpeta:
  analyze_parseq_elpd_visual_token_ablation.py    (Paso 10)
  analyze_parseq_elpd_memory_spatial_swap.py      (Paso 14)

Para cuatro matriculas, interpola tokens visuales de la imagen original y de
la misma imagen con ROI gris. Alpha = 0, .1, ..., 1.0. Dos definiciones de ROI:
  - binaria: sustituye con igual alpha todos los patches que solapan la ROI;
  - fraccional: escala alpha por el solapamiento geometrico de cada patch.
En ABA1234 y DANKE82 se ensaya tambien ROI control, codificando una imagen
con la ROI de control gris de forma independiente (no reciclando el donante
correspondiente a la ROI objetivo).

Se contrastan los extremos alpha=0 y alpha=1 con resultados comprobados en
Pasos 10 y 14. Se guarda AR y REF, los tokens sin normalizar y probabilidades
por posicion, incluida EOS. No se modifican parametros ni datos originales.

Limitaciones: los tokens del encoder ViT ya mezclan contexto global. Una
interpolacion de memorias puede estar fuera de distribucion. El primer alpha
que corrige es el minimo EN LA MALLA PROBADA, no un umbral global ni una
intervencion desplegable sin detectar automaticamente la ROI.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from analyze_parseq_elpd_visual_token_ablation import (
    CASES, DEFAULT_IMAGE_ROOT, DEFAULT_TORCH_HOME, PROJECT,
    safe_image, load_image, gray_mask, forward_with_memory,
    edit_distance, sha256, patch_overlap_weights,
)
from analyze_parseq_elpd_memory_spatial_swap import swap_memory

OUT = PROJECT / 'results/analysis/PARSeq/ELPD/roi_intensity_sweep_v1'
GRID_H, GRID_W, PH, PW = 8, 16, 4, 8
N_TOKENS = GRID_H * GRID_W
ALPHAS = tuple(round(k/10.0, 1) for k in range(11))

# Comprobaciones independientes, ya reproducidas en los Pasos 10 y 14.
REFERENCE_ALPHA1 = {
    'P5_ABA1234': {'roi_binaria': 'ABA1234', 'roi_fraccional': 'ABA1234'},
    'P5_DANKE82': {'roi_binaria': 'DANKE82', 'roi_fraccional': 'DANSKE82'},
    'E010': {'roi_binaria': 'W584RP', 'roi_fraccional': 'W584RP'},
    'C027': {'roi_binaria': 'F0W653', 'roi_fraccional': 'F0W653'},
}


def write_csv(path: Path, records: list[dict]):
    if not records:
        raise RuntimeError(f'No hay registros para {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def roi_weights(bbox: tuple[int, int, int, int]):
    weights = patch_overlap_weights(bbox, GRID_H, GRID_W, PH, PW)
    frac = [float(weights.get(i, 0.0)) for i in range(N_TOKENS)]
    hard = [float(a > 0) for a in frac]
    return {'roi_binaria': hard, 'roi_fraccional': frac}


def threshold_summary(rows: list[dict]) -> list[dict]:
    """El primer alpha que corrige en esta malla, preservando no-monotonicidad."""
    grouped = defaultdict(list)
    for row in rows:
        if row['intervencion'] in ('original', 'pixel_gris'):
            continue
        grouped[(row['id'], row['intervencion'], row['etapa'])].append(row)
    output = []
    for (case_id, mode, stage), rr in grouped.items():
        rr = sorted(rr, key=lambda x: x['alpha'])
        if [x['alpha'] for x in rr] != list(ALPHAS):
            raise RuntimeError(f'Malla alpha incompleta: {case_id} {mode} {stage}')
        corrected = [r['alpha'] for r in rr if r['exact_match'] == 1]
        failed = [r['alpha'] for r in rr if r['exact_match'] == 0]
        transitions = sum(rr[i]['prediccion_normalizada'] != rr[i-1]['prediccion_normalizada']
                          for i in range(1, len(rr)))
        toggles = sum(rr[i]['exact_match'] != rr[i-1]['exact_match']
                      for i in range(1, len(rr)))
        output.append({
            'id': case_id, 'intervencion': mode, 'etapa': stage,
            'gt': rr[0]['gt'],
            'pred_alpha_0': rr[0]['prediccion_normalizada'],
            'pred_alpha_1': rr[-1]['prediccion_normalizada'],
            'correcta_alpha_0': rr[0]['exact_match'],
            'correcta_alpha_1': rr[-1]['exact_match'],
            'primer_alpha_correcto_en_malla': min(corrected) if corrected else '',
            'primer_alpha_incorrecto_en_malla': min(failed) if failed else '',
            'n_alphas_correctos': len(corrected),
            'alphas_correctos': ';'.join(f'{a:.1f}' for a in corrected),
            'alphas_incorrectos': ';'.join(f'{a:.1f}' for a in failed),
            'cambios_de_prediccion': transitions,
            'cambios_estado_acierto': toggles,
            'min_ediciones': min(x['ediciones'] for x in rr),
            'max_ediciones': max(x['ediciones'] for x in rr),
        })
    return output


def color_for_correct(is_correct):
    return (48, 146, 95) if is_correct else (190, 77, 62)


def draw_case_plot(case, rows: list[dict], dest: Path):
    """Grafico simple mediante Pillow, sin requerir matplotlib en el cluster."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1120, 500
    image = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(image)
    def fnt(size):
        try:
            return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
        except OSError:
            return ImageFont.load_default()
    title_font, text_font, small_font = fnt(20), fnt(16), fnt(13)
    draw.text((46, 20), f'{case["id"]} | GT {case["gt"]} | Paso 15 (REF)',
              fill=(27, 48, 68), font=title_font)
    left, top, right, bottom = 75, 85, 830, 390
    subset=[r for r in rows if r['id']==case['id'] and r['etapa']=='REF'
            and r['intervencion'] not in ('original','pixel_gris')]
    modes=sorted(set(r['intervencion'] for r in subset))
    max_ed=max(3, max((r['ediciones'] for r in subset), default=0))
    def px(a): return left+int(a*(right-left))
    def py(ed): return bottom-int(ed/max_ed*(bottom-top))
    for v in range(max_ed+1):
        y=py(v)
        draw.line([(left,y),(right,y)],fill=(224,230,235),width=1)
        draw.text((left-38,y-8),str(v),fill=(63,78,90),font=small_font)
    for a in ALPHAS:
        x=px(a)
        draw.line([(x,top),(x,bottom)],fill=(237,240,244),width=1)
        draw.text((x-15,bottom+13),f'{a:.1f}',fill=(63,78,90),font=small_font)
    palette=[(35,113,185),(199,117,30),(117,81,165)]
    for ind,mode in enumerate(modes):
        rr=sorted((r for r in subset if r['intervencion']==mode),key=lambda v:v['alpha'])
        clr=palette[ind%len(palette)]
        points=[(px(r['alpha']),py(r['ediciones'])) for r in rr]
        draw.line(points,fill=clr,width=3)
        for r,(x,y) in zip(rr,points):
            rad=5
            draw.ellipse((x-rad,y-rad,x+rad,y+rad),fill=color_for_correct(r['exact_match']),outline=clr,width=2)
        yy=110+ind*70
        draw.line([(860,yy),(895,yy)],fill=clr,width=4)
        draw.text((904,yy-13),mode.replace('_',' '),fill=(38,49,60),font=small_font)
    draw.text((left, bottom+49),'Intensidad alpha (0=original, 1=donante)',
              font=text_font,fill=(38,49,60))
    draw.text((25, 62),'Distancia de edicion (menor es mejor)',font=small_font,fill=(63,78,90))
    draw.text((48,465),'Marcador verde: prediccion exacta. Rojo: incorrecta. Barrido exploratorio.',
              font=small_font,fill=(63,78,90))
    dest.parent.mkdir(parents=True,exist_ok=True)
    image.save(dest,optimize=True)


def preflight(image_root: Path, out: Path):
    prepared=[]
    for case in CASES:
        source=safe_image(image_root,case['image_path'])
        im=load_image(source)
        gray=gray_mask(im,case['roi'])
        masks=roi_weights(case['roi'])
        donor_control=None
        if case['control']:
            donor_control=gray_mask(im,case['control'])
            masks['control_binaria']=roi_weights(case['control'])['roi_binaria']
        if not 0 < sum(masks['roi_binaria']) < N_TOKENS:
            raise RuntimeError(f'Mascara ROI no valida: {case["id"]}')
        prepared.append((case,source,im,gray,donor_control,masks))
    out.mkdir(parents=True)
    previews=out/'imagenes_preflight'
    previews.mkdir()
    for case,source,im,gray,donor_control,masks in prepared:
        from PIL import Image
        canvas=Image.new('RGB',(128*4*2+20,32*4+15),'white')
        canvas.paste(im.resize((512,128),Image.Resampling.NEAREST),(0,0))
        canvas.paste(gray.resize((512,128),Image.Resampling.NEAREST),(532,0))
        canvas.save(previews/f'{case["id"]}.png')
        print(f'[PREP] {case["id"]} ROI={int(sum(masks["roi_binaria"]))} tokens; '
              f'modos={list(masks)}; GT={case["gt"]}',flush=True)
    return prepared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', type=Path,default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--prepare-only',action='store_true',help='Sin GPU; solo validar imagenes y ROI')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'No se sobrescriben experimentos previos: {args.out}')
    print('=== PASO 15 | PARSeq ELPD V17 | INTENSIDAD DE TOKENS ViT ===',flush=True)
    print('Alpha:',', '.join(f'{v:.1f}' for v in ALPHAS),flush=True)
    prepared=preflight(args.image_root,args.out)
    if args.prepare_only:
        print('PREFLIGHT OK: 4 imagenes; sin inferencia GPU.',flush=True)
        print('SALIDA:',args.out,flush=True)
        return

    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME',str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model=torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core=model.model
    if not bool(core.decode_ar) or int(core.refine_iters)!=1:
        raise RuntimeError('Se esperaba decode_ar=True y refine_iters=1')
    if tuple(int(v) for v in model.hparams.img_size)!=(32,128):
        raise RuntimeError('Resolucion distinta de 32x128')
    pe=core.encoder.patch_embed
    if tuple(int(v) for v in pe.patch_size)!=(PH,PW) or tuple(int(v) for v in pe.grid_size)!=(GRID_H,GRID_W):
        raise RuntimeError('Patch/grid distinto de 4x8 y 8x16')
    transform=T.Compose([T.ToTensor(),T.Normalize((0.5,)*3,(0.5,)*3)])

    rows=[]
    position_rows=[]
    summaries={}
    started=time.perf_counter()
    for k,(case,source,im,gray,control_gray,spatial_masks) in enumerate(prepared,1):
        with torch.inference_mode():
            x=transform(im).unsqueeze(0).cuda()
            gx=transform(gray).unsqueeze(0).cuda()
            original=core.encode(x).detach()
            donor=core.encode(gx).detach()
            if tuple(original.shape)!=(1,N_TOKENS,384):
                raise RuntimeError(f'Memoria inesperada: {original.shape}')
            control_donor=None
            if control_gray is not None:
                control_donor=core.encode(transform(control_gray).unsqueeze(0).cuda()).detach()
            donor_by_mode={'roi_binaria': donor,'roi_fraccional':donor}
            if control_donor is not None:
                donor_by_mode['control_binaria']=control_donor

        results_by_mode_stage=defaultdict(dict)
        comparisons=[]
        baseline={}
        for mode,alpha,inp,memory_override in [
                ('original',0.0,x,None),('pixel_gris',1.0,gx,None)]:
            for stage,refines in (('AR',0),('REF',1)):
                pred=forward_with_memory(model,core,inp,memory_override,refines)
                baseline[(mode,stage)]=pred
                comparisons.append((mode,alpha,stage,pred,0,0.0))
        for stage in ('AR','REF'):
            if baseline[('original',stage)]['normalizada']!=case['ar_esperado' if stage=='AR' else 'ref_esperado']:
                raise RuntimeError(f'{case["id"]}: no reproduce original {stage}')
        if baseline[('pixel_gris','REF')]['normalizada']!=case['pixel_objetivo_esperado']:
            raise RuntimeError(f'{case["id"]}: no reproduce mascara en pixeles')

        for mode,base_mask in spatial_masks.items():
            memory_donor=donor_by_mode[mode]
            for alpha in ALPHAS:
                weights=[alpha*b for b in base_mask]
                with torch.inference_mode():
                    variant_mem=swap_memory(original,memory_donor,weights)
                if alpha==0.0 and not torch.equal(variant_mem,original):
                    raise RuntimeError('alpha=0 no reproduce exactamente la memoria original')
                if alpha==1.0 and mode=='roi_binaria':
                    # Reproduce intercambio binario del Paso 14 para una imagen.
                    pass
                for stage,refines in (('AR',0),('REF',1)):
                    pred=forward_with_memory(model,core,x,variant_mem,refines)
                    if alpha==0.0 and (pred['raw']!=baseline[('original',stage)]['raw'] or
                                       pred['eos_1based']!=baseline[('original',stage)]['eos_1based']):
                        raise RuntimeError(f'{case["id"]} {mode} {stage} alpha=0 altero prediccion')
                    if alpha==1.0 and stage=='REF' and mode in REFERENCE_ALPHA1[case['id']]:
                        expected=REFERENCE_ALPHA1[case['id']][mode]
                        if pred['normalizada']!=expected:
                            raise RuntimeError(f'{case["id"]} {mode} alpha=1: '
                                               f'{pred["normalizada"]!r} != Paso 14 {expected!r}')
                    comparisons.append((mode,alpha,stage,pred,
                                        int(sum(v>0 for v in weights)),sum(weights)))
                    results_by_mode_stage[(mode,stage)][alpha]=pred['normalizada']
        for mode,alpha,stage,pred,n_active,sum_weights in comparisons:
            p_gt_end=None
            idx=len(case['gt']) # index 0-based de la posicion len(GT)+1
            if idx<pred['probabilities'].shape[0]:
                p_gt_end=float(pred['probabilities'][idx,int(model.tokenizer.eos_id)])
            rows.append({
                'id':case['id'],'grupo':case['grupo'],'gt':case['gt'],
                'roi_nombre':case['roi_nombre'],'intervencion':mode,
                'alpha':alpha,'etapa':stage,
                'prediccion_raw':pred['raw'],'prediccion_normalizada':pred['normalizada'],
                'exact_match':int(pred['normalizada']==case['gt']),
                'ediciones':edit_distance(case['gt'],pred['normalizada']),
                'posicion_eos_1based':pred['eos_1based'] if pred['eos_1based'] is not None else '',
                'n_logits':pred['n_logits'],'p_eos_pos_gt_mas_1':round(p_gt_end,8) if p_gt_end is not None else '',
                'n_tokens_afectados':n_active,'intensidad_total_equivalente':round(sum_weights,6),
            })
            if mode in spatial_masks:
                probs=pred['probabilities']
                for pos in range(probs.shape[0]):
                    values, indices=probs[pos].topk(1)
                    position_rows.append({
                        'id':case['id'],'intervencion':mode,'alpha':alpha,'etapa':stage,
                        'posicion_1based':pos+1,'token_id_maximo':int(indices[0]),
                        'probabilidad_maxima':round(float(values[0]),8),
                        'probabilidad_EOS':round(float(probs[pos,int(model.tokenizer.eos_id)]),8),
                    })
        summaries[case['id']]={
            'gt':case['gt'],'grupo':case['grupo'],'pred_original':baseline[('original','REF')]['normalizada'],
            'pred_pixel_gris':baseline[('pixel_gris','REF')]['normalizada'],
            'n_tokens_roi_binaria':int(sum(spatial_masks['roi_binaria'])),
            'sha256_imagen':sha256(source),
            'n_modos':len(spatial_masks),
            'barrido_ref':{name:{f'{alpha:.1f}':value for alpha,value in sorted(alphas.items())}
                           for (name,stg),alphas in results_by_mode_stage.items() if stg=='REF'},
        }
        print(f'[{k:02d}/{len(prepared)}] {case["id"]} | original={summaries[case["id"]]["pred_original"]}'
              f' | alpha=1 ROI={results_by_mode_stage[("roi_binaria","REF")][1.0]}',flush=True)

    thresholds=threshold_summary(rows)
    write_csv(args.out/'comparacion_intensidades.csv',rows)
    write_csv(args.out/'umbrales_por_caso.csv',thresholds)
    write_csv(args.out/'probabilidades_por_posicion.csv',position_rows)
    (args.out/'curvas').mkdir(exist_ok=True)
    for case in CASES:
        draw_case_plot(case, rows, args.out/'curvas'/f'{case["id"]}.png')
    summary={
        'paso':15,'modelo':'PARSeq oficial pretrained','dataset':'ELPD GT V17',
        'n_imagenes':len(CASES),'n_alpha_por_modo':len(ALPHAS),'alphas':list(ALPHAS),
        'n_filas_comparacion':len(rows),'n_filas_probabilidades':len(position_rows),
        'reproduccion_pasos_10_14':'OK (original, ROI gris y extremos)',
        'criterio': 'primer alpha que produce Exact Match en la malla 0.0-1.0 de paso 0.1',
        'metodo':'Interpolacion de memorias ViT originales/donante en ROI binaria, ROI fraccional y ROI de control',
        'resultados_por_caso':summaries,
        'umbrales_ref': [row for row in thresholds if row['etapa']=='REF'],
        'duracion_segundos':round(time.perf_counter()-started,2),
        'pesos_modificados':False,'GT_modificado':False,
        'limitaciones':[
            'El minimo alpha es el minimo de una malla discreta, no un umbral exacto ni generalizable.',
            'La exactitud puede ser no monotonica con alpha; consultar curva completa.',
            'Los patches ViT contienen informacion global mezclada, no representaciones exclusivamente locales.',
            'La sustitucion puede generar representaciones fuera de distribucion.',
            'Algunos patches intersectan parcialmente caracteres; incluir controles.',
            'p_eos_pos_gt_mas_1 no tiene interpretacion causal sencilla con inserciones iniciales/internas.',
            'Cuatro matriculas seleccionadas, ROI manuales; no estimar rendimiento en ELPD completo.',
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULTADO: BARRIDO DE INTENSIDAD COMPLETADO',flush=True)
    print('Predicciones AR/REF:',len(rows),'| Umbrales:',len(thresholds),flush=True)
    print('SALIDA:',args.out,flush=True)


if __name__=='__main__':
    main()
