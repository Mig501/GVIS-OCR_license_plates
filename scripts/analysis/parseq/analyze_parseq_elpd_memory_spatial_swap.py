#!/usr/bin/env python3
"""PASO 14 | PARSeq + ELPD V17 | Trasplante espacial de memoria del ViT.

Ruta: scripts/analysis/parseq/analyze_parseq_elpd_memory_spatial_swap.py
Ejecutar SIN argumentos mediante GVIS: Ejecutar en GPU.

Requiere en la misma carpeta el script del Paso 10:
    analyze_parseq_elpd_visual_token_ablation.py

Hipotesis: en el ViT, los cambios asociados a una ROI grafica se propagan a
representaciones de otras posiciones. Comparando dos memorias completas
(encoder de imagen original y de imagen con ROI gris), se intercambian
selectivamente los tokens EN la ROI o FUERA de ella. Un barrido por anillos
espaciales evalua si basta ampliar la vecindad del escudo.

IMPORTANTE: Los tokens del ViT mezclan informacion global. Intercambiarlos
es una intervencion sobre el interfaz encoder-decoder, no prueba que un
bloque ViT individual, la MHA2 o un elemento semantico concreto sea la causa.
Las mezclas pueden generar memorias fuera de distribucion. Las cuatro imagenes
son ejemplos seleccionados, no una estimacion de eficacia general.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from analyze_parseq_elpd_visual_token_ablation import (
    CASES, DEFAULT_IMAGE_ROOT, DEFAULT_TORCH_HOME, PROJECT,
    safe_image, load_image, gray_mask, patch_overlap_weights,
    forward_with_memory, normalized, edit_distance, sha256, make_preview,
)

DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/memory_spatial_swap_v1'
N_TOKENS = 128
GRID_H, GRID_W, PH, PW = 8, 16, 4, 8


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError('No hay registros para guardar')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_alpha_roi(bbox):
    weights=patch_overlap_weights(bbox, GRID_H, GRID_W, PH, PW)
    return [float(weights.get(i, 0.0)) for i in range(N_TOKENS)]


def ring_expansion(hard_roi, radius: int):
    """Incluye ROI y patches a distancia Chebyshev <= radius; seleccion binaria."""
    centers=[divmod(i,GRID_W) for i,a in enumerate(hard_roi) if a>0]
    values=[]
    for idx in range(N_TOKENS):
        r,c=divmod(idx,GRID_W)
        mind=min(max(abs(r-rr),abs(c-cc)) for rr,cc in centers)
        values.append(1.0 if mind<=radius else 0.0)
    return values


def swap_memory(memory_original, memory_gris, alpha):
    """Interpolacion por token: (1-alpha)*original + alpha*donante."""
    import torch
    if memory_original.shape != memory_gris.shape or memory_original.ndim!=3:
        raise RuntimeError('Formas de memoria incompatibles')
    if len(alpha)!=memory_original.shape[1]:
        raise RuntimeError('Mascara espacial no coincide con numero de tokens')
    a=torch.tensor(alpha, dtype=memory_original.dtype, device=memory_original.device).reshape(1,-1,1)
    if float(a.min())<0 or float(a.max())>1:
        raise ValueError('Se requiere mezcla convexa con 0<=alpha<=1')
    return (memory_original*(1-a)+memory_gris*a).detach()


def condition_alphas(bbox):
    fractional=get_alpha_roi(bbox)
    roi_hard=[float(a>0) for a in fractional]
    out={
        'memoria_original': [0.0]*N_TOKENS,
        # Igual al Paso 10: los parches fronterizos se mezclan parcialmente.
        'solo_roi_fraccional': fractional,
        # Comparacion causal espacial: ambos compartimentos forman una particion.
        'solo_roi_donante': roi_hard,
        'solo_exterior_donante': [1.0-a for a in roi_hard],
        'roi_mas_anillo_1': ring_expansion(roi_hard,1),
        'roi_mas_anillo_2': ring_expansion(roi_hard,2),
        'roi_mas_anillo_3': ring_expansion(roi_hard,3),
        'toda_memoria_donante': [1.0]*N_TOKENS,
    }
    return out


def verify_conditions(alphas):
    import math
    a=alphas['solo_roi_donante']
    b=alphas['solo_exterior_donante']
    if not all(math.isclose(x+y,1.0,abs_tol=1e-10) for x,y in zip(a,b)):
        raise RuntimeError('ROI y exterior no son complementarios')
    if sum(x>0 for x in a)>=N_TOKENS or sum(x>0 for x in b)<=0:
        raise RuntimeError('ROI ocupa toda la imagen; no hay exterior')
    frac=alphas['solo_roi_fraccional']
    if not all(0.0<=f<=h for f,h in zip(frac,a)):
        raise RuntimeError('La intervencion fraccional toca parches fuera de ROI')
    prev=a
    for key in ('roi_mas_anillo_1','roi_mas_anillo_2','roi_mas_anillo_3','toda_memoria_donante'):
        current=alphas[key]
        if not all(y+1e-9>=x for x,y in zip(prev,current)):
            raise RuntimeError(f'El anillo no es monotono: {key}')
        prev=current
    if any(v!=1.0 for v in alphas['toda_memoria_donante']):
        raise RuntimeError('La ultima condicion no sustituye toda la memoria')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root',type=Path,default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--prepare-only',action='store_true',help='Valida ROI e imagenes sin GPU')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Salida ya existente (no se sobrescribe): {args.out}')
    print('=== PASO 14 | PARSEQ ELPD V17 | INTERCAMBIO ESPACIAL DE MEMORIA ViT ===',flush=True)
    prepared=[]
    for case in CASES:
        source=safe_image(args.image_root,case['image_path'])
        image=load_image(source)
        image_gris=gray_mask(image,case['roi'])
        alphas=condition_alphas(case['roi'])
        verify_conditions(alphas)
        prepared.append((case,source,image,image_gris,alphas))
    args.out.mkdir(parents=True)
    (args.out/'vistas_previas').mkdir()
    for case,source,image,image_gris,alphas in prepared:
        roi_ids=[i for i,a in enumerate(alphas['solo_roi_donante']) if a>0]
        make_preview(case,image,(GRID_W,GRID_H,PW,PH),roi_ids,[],
                     args.out/'vistas_previas'/f'{case["id"]}.png')
        print(f'[PREP] {case["id"]}: {len(roi_ids)} patches ROI | '
              f'{N_TOKENS-len(roi_ids)} exterior',flush=True)
    if args.prepare_only:
        print('PREFLIGHT OK: 4 imagenes, ROI y anillos; sin inferencias GPU',flush=True)
        print('SALIDA:',args.out,flush=True)
        return

    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: utilizar GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME',str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model=torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core=model.model
    if not bool(core.decode_ar) or int(core.refine_iters)!=1:
        raise RuntimeError('Checkpoint no tiene decode_ar=True, refine_iters=1')
    if tuple(int(v) for v in model.hparams.img_size)!=(32,128):
        raise RuntimeError('Preprocesamiento distinto de 32x128')
    patch=core.encoder.patch_embed
    if tuple(int(v) for v in patch.patch_size)!=(PH,PW) or tuple(int(v) for v in patch.grid_size)!=(GRID_H,GRID_W):
        raise RuntimeError('Grid o patch size inesperado: se esperaba 8x16, patch 4x8')
    transform=T.Compose([T.ToTensor(),T.Normalize((0.5,)*3,(0.5,)*3)])
    started=time.perf_counter()
    rows=[]
    token_rows=[]
    result_summary={}
    n_ref_original=0
    n_ref_gris=0
    for num,(case,source,image,image_gris,alphas) in enumerate(prepared,1):
        with torch.inference_mode():
            x=transform(image).unsqueeze(0).cuda()
            xgris=transform(image_gris).unsqueeze(0).cuda()
            original_memory=core.encode(x).detach()
            gray_memory=core.encode(xgris).detach()
            if tuple(original_memory.shape[:2])!=(1,N_TOKENS):
                raise RuntimeError(f'Encoder devuelve forma inesperada: {original_memory.shape}')
            variants=[('original',x,None,'control_original'),('pixel_gris_objetivo',xgris,None,'control_pixeles')]
            for name,weights in alphas.items():
                if name=='memoria_original':
                    continue  # El original real ya se mide; evita duplicado de logits.
                patched=swap_memory(original_memory,gray_memory,weights)
                variants.append((name,x,patched,'memory_swap'))
            # Prueba de equivalencia del swap total con la imagen gris.
            if not torch.allclose(swap_memory(original_memory,gray_memory,alphas['toda_memoria_donante']),gray_memory,rtol=0,atol=0):
                raise RuntimeError('La memoria completa no coincide exactamente con el donante gris')
        predictions={}
        for variant,inp,override,kind in variants:
            predictions[variant]={}
            for stage,refines in (('AR',0),('REF',1)):
                p=forward_with_memory(model,core,inp,override,refines)
                prediction=p['normalizada']
                predictions[variant][stage]=prediction
                row={
                    'id':case['id'],'tipo_caso':case['grupo'],'image_path':case['image_path'],
                    'gt':case['gt'],'region':case['roi_nombre'],'variante':variant,
                    'tipo_intervencion':kind,'etapa':stage,
                    'prediccion_raw':p['raw'],'prediccion_normalizada':prediction,
                    'exact_match':int(prediction==case['gt']),
                    'ediciones':edit_distance(case['gt'],prediction),
                    'posicion_EOS_1based':p['eos_1based'] or '',
                    'n_logits':p['n_logits'],
                    'n_tokens_intervenidos':0 if override is None else sum(a>0 for a in alphas[variant]),
                    'peso_tokens_equivalentes':0 if override is None else round(sum(alphas[variant]),6),
                    'n_tokens_roi':sum(a>0 for a in alphas['solo_roi_donante']),
                }
                rows.append(row)
                # Diagnostico resumido de la prediccion por posicion, incluido EOS.
                if kind=='memory_swap':
                    probs=p['probabilities']
                    for pos in range(probs.shape[0]):
                        top=probs[pos].topk(1)
                        token_rows.append({
                            'id':case['id'],'variante':variant,'etapa':stage,
                            'posicion_1based':pos+1,'token_id_argmax':int(top.indices[0]),
                            'prob_argmax':round(float(top.values[0]),8),
                            'p_EOS':round(float(probs[pos,int(model.tokenizer.eos_id)]),8),
                        })
        for stage in ('AR','REF'):
            # Equivalencia: donor completo tras encoder = encoder de imagen gris,
            # por lo que las predicciones, incluida posicion EOS, deben reproducirse.
            if predictions['toda_memoria_donante'][stage]!=predictions['pixel_gris_objetivo'][stage]:
                raise RuntimeError(f'{case["id"]} {stage}: donante completo difiere de mascara en pixeles')
            if predictions['original'][stage]!=case[f'{"ar" if stage=="AR" else "ref"}_esperado']:
                raise RuntimeError(f'{case["id"]} {stage}: no reproduce checkpoint original')
        n_ref_original+=1
        if predictions['pixel_gris_objetivo']['REF'] != case['pixel_objetivo_esperado']:
            raise RuntimeError(f'{case["id"]}: la mascara en pixeles no reproduce pasos 5/6B')
        n_ref_gris+=1
        # Reproduce la intervencion fraccional del Paso 10, con el mismo
        # preprocesamiento, imagen donante y pesos de mezcla geometrica.
        paso10_fraccional={
            'P5_ABA1234':'ABA1234',
            'P5_DANKE82':'DANSKE82',
            'E010':'W584RP',
            'C027':'F0W653',
        }
        got_frac=predictions['solo_roi_fraccional']['REF']
        if got_frac!=paso10_fraccional[case['id']]:
            raise RuntimeError(f'{case["id"]}: ROI fraccional no reproduce Paso 10: '
                               f'{got_frac!r} vs {paso10_fraccional[case["id"]]!r}')
        result_summary[case['id']]={
            'GT':case['gt'],'pred_original':predictions['original']['REF'],
            'pred_gris':predictions['pixel_gris_objetivo']['REF'],
            'sha256_imagen':sha256(source),'ROI_128x32':list(case['roi']),
            'n_tokens_roi':sum(a>0 for a in alphas['solo_roi_donante']),
            'refinado_por_condicion':{name:v['REF'] for name,v in predictions.items()},
            'autoregresivo_por_condicion':{name:v['AR'] for name,v in predictions.items()},
        }
        print(f'[{num:02d}/{len(prepared)}] {case["id"]} | '
              + ' | '.join(f'{name}={v["REF"]}' for name,v in predictions.items()),flush=True)
    write_csv(args.out/'comparacion_memorias.csv',rows)
    write_csv(args.out/'tokens_por_posicion.csv',token_rows)
    summary={
        'paso':14,'modelo':'PARSeq original pretrained','dataset':'ELPD GT V17',
        'casos':len(CASES),'originales_reproducidos':n_ref_original,
        'mascaras_pixeles_reproducidas':n_ref_gris,
        'regla_de_mezcla':'mem=(1-alpha)*mem_original+alpha*mem_gris; ROI/exterior binarios disjuntos, ademas ROI fraccional para reproducir Paso 10',
        'condiciones_memoria':['solo_roi_fraccional','solo_roi_donante','solo_exterior_donante','roi_mas_anillo_1','roi_mas_anillo_2','roi_mas_anillo_3','toda_memoria_donante'],
        'predicciones_totales':len(rows),'resultados_por_caso':result_summary,
        'duracion_segundos':round(time.perf_counter()-started,2),
        'pesos_modificados':False,'GT_modificado':False,
        'limitaciones':[
            'Intercambiar activaciones del ViT mide sensibilidad a cambios de memoria; no atribuye causalidad a una capa concreta.',
            'Los tokens originales y del donante gris ya contienen contexto global de la imagen.',
            'Los parches que rozan ROI se sustituyen completos en las condiciones binarias; pueden incluir parte de caracteres.',
            'La condicion ROI fraccional replica el tratamiento de bordes del Paso 10.',
            'Las memorias hibridas pueden quedar fuera de la distribucion de entrenamiento.',
            'Una correccion al sustituir tokens externos no demuestra que esos tokens codifiquen semanticamente el escudo.',
            'Solo cuatro ejemplos seleccionados; no se estima mejora general sobre ELPD.',
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('RESULTADO: TRASPLANTES DE MEMORIA ViT ANALIZADOS; SALIDAS EXPORTADAS',flush=True)
    print('SALIDA:',args.out,flush=True)


if __name__=='__main__':
    main()
