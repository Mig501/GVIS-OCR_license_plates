#!/usr/bin/env python3
"""PASO 13 | ELPD V17 | Propagacion espacial de cambios a lo largo del encoder ViT.

Ruta: scripts/analysis/parseq/analyze_parseq_elpd_vit_feature_spread.py
Ejecutar SIN argumentos con GVIS: Ejecutar en GPU.

Requiere en la misma carpeta:
    analyze_parseq_elpd_visual_token_ablation.py   (Paso 10)

Compara representaciones del encoder con imagen original y misma imagen con
ROI grafica enmascarada en gris. Registra la salida de patch_embed, bloques
1/3/6/9/12 y LayerNorm final, manteniendo las 128 posiciones.

La diferencia de activaciones fuera de la ROI cuantifica propagacion de un
CAMBIO DE ENTRADA, no identifica informacion semantica, causalidad de una capa
ni garantiza que el modelo haya reconocido el escudo como un caracter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from analyze_parseq_elpd_visual_token_ablation import (
    CASES, DEFAULT_IMAGE_ROOT, DEFAULT_TORCH_HOME, safe_image,
    load_image, gray_mask, patch_overlap_weights, forward_with_memory,
    make_preview, sha256,
)

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
OUT_DEFAULT = PROJECT / 'results/analysis/PARSeq/ELPD/vit_feature_spread_v1'
BLOCKS = (0, 2, 5, 8, 11)  # Indices Python; equivale a bloques ViT 1, 3, 6, 9, 12


def write_csv(path, rows):
    if not rows:
        raise ValueError('Sin datos que guardar')
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        wr = csv.DictWriter(file, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def encoder_features(core, x):
    """Captura activaciones del ViT sin alterar forward ni parametros.

    Detiene hooks de forma segura incluso si ocurre una excepcion.
    """
    import torch
    modules = [('patch_embed', core.encoder.patch_embed)]
    modules.extend((f'block_{idx+1:02d}', core.encoder.blocks[idx]) for idx in BLOCKS)
    modules.append(('encoder_norm', core.encoder.norm))
    captures = {}
    handles = []

    def hook_for(name):
        def capture(_m, _args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f'{name}: salida no es Tensor: {type(output)}')
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 128:
                raise RuntimeError(f'{name}: forma de tokens inesperada: {tuple(output.shape)}')
            captures[name] = output.detach().float().cpu()[0].clone()
        return capture

    try:
        for name, module in modules:
            handles.append(module.register_forward_hook(hook_for(name)))
        with torch.inference_mode():
            memory = core.encode(x)
        if tuple(memory.shape[:2]) != (1, 128):
            raise RuntimeError(f'Memoria de ViT inesperada: {tuple(memory.shape)}')
    finally:
        for handle in handles:
            handle.remove()
    required = [name for name, _ in modules]
    if list(captures) != required:
        raise RuntimeError(f'No se capturaron todos los bloques: {list(captures)} != {required}')
    if not torch.allclose(captures['encoder_norm'], memory.float().cpu()[0], rtol=1e-5, atol=1e-6):
        raise RuntimeError('encoder_norm no reproduce la memoria visual final')
    return captures


def distances(activ_orig, activ_gray, idx_in):
    import torch
    import numpy as np
    if tuple(activ_orig.shape) != tuple(activ_gray.shape) or activ_orig.shape[0] != 128:
        raise RuntimeError('Formas incompatibles en activaciones')
    diff = (activ_orig - activ_gray).float()
    d = torch.linalg.vector_norm(diff, ord=2, dim=1).numpy()
    energy = d ** 2
    mask = np.zeros(128, dtype=bool)
    mask[list(idx_in)] = True
    ein = float(energy[mask].sum(dtype=np.float64))
    eout = float(energy[~mask].sum(dtype=np.float64))
    etotal = ein + eout
    # Medida relativa adicional: media de L2(delta)/L2(token original).
    orig_norm = torch.linalg.vector_norm(activ_orig.float(), ord=2, dim=1).numpy()
    rel = d / np.maximum(orig_norm, 1e-8)
    stats = {
        'n_patches_roi': int(mask.sum()),
        'n_patches_fuera': int((~mask).sum()),
        'energia_dentro_roi': ein,
        'energia_fuera_roi': eout,
        'energia_total': etotal,
        'fraccion_energia_fuera': (eout / etotal if etotal else 0.0),
        'L2_medio_dentro': float(d[mask].mean()),
        'L2_medio_fuera': float(d[~mask].mean()),
        'L2_rel_medio_dentro': float(rel[mask].mean()),
        'L2_rel_medio_fuera': float(rel[~mask].mean()),
        'L2_max_fuera': float(d[~mask].max()),
    }
    return d, stats


def make_map_sheet(out_path, image, gray, roi, maps, metrics):
    """Mapas con escala de color COMUN dentro de cada caso entre capas."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    SCALE = 6
    W, H = 128*SCALE, 32*SCALE
    try:
        f_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        f_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    except OSError:
        f_title = ImageFont.load_default()
        f_small = ImageFont.load_default()
    labels = ['Original', 'ROI gris'] + list(maps)
    sheet = Image.new('RGB', (W+55, 50 + len(labels)*(H+49)), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((22, 12), 'Cambio de activaciones del encoder ViT | escala comun por imagen', fill='#18324A', font=f_title)
    high = max([float(np.max(d)) for d in maps.values()] + [1e-9])
    y = 50
    orig_big = image.resize((W,H), Image.Resampling.BICUBIC)
    for label in labels:
        description = label
        if label in maps:
            s = metrics[label]
            description += f'  | energia fuera ROI: {s["fraccion_energia_fuera"]*100:.1f}%'
        draw.text((22,y),description,fill='#18324A',font=f_small)
        y0 = y+22
        if label == 'Original':
            displayed = orig_big
        elif label == 'ROI gris':
            displayed = gray.resize((W,H),Image.Resampling.BICUBIC)
        else:
            # Heatmap rojo continuo, intensidad global por caso; transparencia proporcional.
            arr = maps[label].reshape(8,16)
            h_img = Image.fromarray(np.uint8(np.clip(arr/high,0,1)*255), 'L')
            h_img = h_img.resize((W,H),Image.Resampling.BILINEAR)
            alpha = h_img.point(lambda v: int(v*0.78))
            base = orig_big.copy().convert('RGBA')
            layer = Image.new('RGBA', (W,H), (237,64,31,255))
            base.paste(layer,(0,0),alpha)
            displayed = base.convert('RGB')
        sheet.paste(displayed,(22,y0))
        if label not in ('Original','ROI gris'):
            x0, yy0, x1, yy1 = roi
            draw.rectangle((22+x0*SCALE,y0+yy0*SCALE,22+x1*SCALE-1,y0+yy1*SCALE-1),
                           outline='#14B8A6',width=3)
        y += H+49
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.crop((0,0,W+55,y+3)).save(out_path,optimize=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root',type=Path,default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--out',type=Path,default=OUT_DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if (args.out/'resumen.json').exists():
        raise FileExistsError(f'Ya existen resultados finales: {args.out}')
    print('=== PASO 13 | PARSeq ELPD V17 | DIFUSION DE CAMBIOS EN ViT ===',flush=True)
    prepared=[]
    for case in CASES:
        source=safe_image(args.image_root,case['image_path'])
        image=load_image(source)
        gray=gray_mask(image,case['roi'])
        weights=patch_overlap_weights(case['roi'],8,16,4,8)
        if not weights or len(weights)>=128:
            raise RuntimeError(f'{case["id"]}: ROI invalida')
        prepared.append((case,source,image,gray,weights))
    print(f'Casos: {len(prepared)} | bloques: patch_embed, 1, 3, 6, 9, 12, norm',flush=True)
    if args.prepare_only:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/'vistas_previas').mkdir(exist_ok=True)
        for case,source,image,gray,weights in prepared:
            make_preview(case,image,(16,8,8,4),list(weights),[],
                         args.out/'vistas_previas'/f'{case["id"]}.png')
            print(f'[OK] {case["id"]}: {len(weights)} patches, {source.name}',flush=True)
        print('PREFLIGHT OK: no se han ejecutado inferencias GPU',flush=True)
        return

    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('No hay CUDA; ejecutar con GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME',str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model=torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core=model.model
    if not bool(core.decode_ar) or int(core.refine_iters)!=1:
        raise RuntimeError('Se requiere PARSeq oficial decode_ar=True, refine_iters=1')
    if tuple(int(v) for v in model.hparams.img_size)!=(32,128):
        raise RuntimeError('Resolucion de imagen distinta de (32,128)')
    patch=core.encoder.patch_embed
    if tuple(int(v) for v in patch.patch_size)!=(4,8) or tuple(int(v) for v in patch.grid_size)!=(8,16):
        raise RuntimeError('La distribucion de 128 parches (8x16) no coincide')
    if len(core.encoder.blocks)!=12 or core.encoder.norm is None:
        raise RuntimeError('Estructura del ViT distinta de 12 bloques + norm')

    t0=time.perf_counter()
    transform=T.Compose([T.ToTensor(),T.Normalize((0.5,)*3,(0.5,)*3)])
    rows=[]
    cases_out={}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'mapas_distribucion').mkdir(exist_ok=True)
    for ci,(case,source,image,gray,weights) in enumerate(prepared,1):
        gt=case['gt']
        with torch.inference_mode():
            x=transform(image).unsqueeze(0).cuda()
            xm=transform(gray).unsqueeze(0).cuda()
            orig=encoder_features(core,x)
            changed=encoder_features(core,xm)
            # Comprobacion: mismos resultados de las mascaras del Paso 5/6B.
            pred_orig=forward_with_memory(model,core,x,None,refine_iters=1)['normalizada']
            pred_gray=forward_with_memory(model,core,xm,None,refine_iters=1)['normalizada']
        if pred_orig!=case['ref_esperado'] or pred_gray!=case['pixel_objetivo_esperado']:
            raise RuntimeError(f'{case["id"]}: predicciones no reproducidas; '
                               f'original {pred_orig!r} vs {case["ref_esperado"]!r}; '
                               f'gris {pred_gray!r} vs {case["pixel_objetivo_esperado"]!r}')
        layer_maps={}
        layer_metrics={}
        for name in orig:
            magnitudes,stats=distances(orig[name],changed[name],weights.keys())
            layer_maps[name]=magnitudes
            layer_metrics[name]=stats
            rows.append({
                'id':case['id'],'GT':gt,'pred_original':pred_orig,
                'pred_gris':pred_gray,'region':case['roi_nombre'],
                'capa':name, **{k:round(v,9) if isinstance(v,float) else v for k,v in stats.items()},
            })
        first=layer_metrics['patch_embed']
        if first['energia_total']<=1e-10:
            raise RuntimeError(f'{case["id"]}: la ROI no afecta siquiera a patch_embed')
        if first['energia_fuera_roi']>max(1e-9,first['energia_total']*1e-5):
            raise RuntimeError(f'{case["id"]}: patch_embed cambia fuera de ROI; comprobar preprocesamiento')
        make_map_sheet(args.out/'mapas_distribucion'/f'{case["id"]}.png',
                       image,gray,case['roi'],layer_maps,layer_metrics)
        cases_out[case['id']]={
            'GT':gt,'imagen':case['image_path'],'sha256':sha256(source),
            'roi':list(case['roi']),'n_patches_roi':len(weights),
            'prediccion_original':pred_orig,'prediccion_con_roi_gris':pred_gray,
            'fraccion_energia_fuera_por_capa':{
                key:round(s['fraccion_energia_fuera'],6) for key,s in layer_metrics.items()
            },
        }
        print(f'[{ci:02d}/{len(prepared)}] {case["id"]} | '
              f'{pred_orig} -> {pred_gray} | fuera ROI: '
              f'patch={100*first["fraccion_energia_fuera"]:.2f}%, '
              f'ViT12={100*layer_metrics["block_12"]["fraccion_energia_fuera"]:.2f}%',flush=True)

    write_csv(args.out/'difusion_por_capa.csv',rows)
    summary={
        'paso':13,'modelo':'PARSeq oficial pretrained','dataset':'ELPD V17',
        'casos':len(CASES),'capas_analizadas':list(orig),
        'resultados':cases_out,'duracion_segundos':round(time.perf_counter()-t0,2),
        'checkpoint_modificado':False,'GT_modificado':False,
        'descripcion_medida':'Fraccion de energia L2^2 del cambio entre activaciones original y ROI gris fuera de parches que solapan la ROI',
        'limitaciones':[
            'La diferencia de activaciones no identifica por si misma la informacion semantica del escudo.',
            'Medir propagacion no prueba que el encoder cause una insercion.',
            'ROI y parches de borde pueden solaparse con caracteres.',
            'Pocas imagenes seleccionadas; no estimar tasas generales.',
            'La escala de los mapas es comun dentro de cada imagen, no entre imagenes.',
            'La diferencia depende del tamano y tipo de perturbacion (relleno gris).',
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULTADO: DIFUSION VISUAL EN EL ENCODER ANALIZADA',flush=True)
    print('SALIDA:',args.out,flush=True)

if __name__=='__main__':
    main()
