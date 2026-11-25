#!/usr/bin/env python3
"""PASO 10 | PARSeq + ELPD V17 | Ablacion e intercambio de tokens visuales.

Ubicacion recomendada:
  scripts/analysis/parseq/analyze_parseq_elpd_visual_token_ablation.py

Ejecutar sin argumentos con la tarea GVIS: Ejecutar en GPU. No altera pesos,
checkpoint, dataset o resultados de pasos anteriores.

Comparaciones:
  - Original (sin intervenciones)
  - Imagen con ROI gris (codificada de nuevo, como paso 5/6B)
  - Solo memoria visual en ROI sustituida por la memoria de esa imagen gris
  - Solo memoria visual en ROI sustituida por la media de tokens no ROI
  - Controles graficos equivalentes (dos casos con ROI control auditadas)

Las intervenciones se aplican a la salida final del encoder, DESPUES de que
las capas ViT hayan mezclado informacion espacial. NO equivalen a borrar
informacion de la ROI del encoder entero ni demuestran que MHA2 sea la unica
causa del fallo. Se guardan AR (0 refinamientos) y REF (1 refinamiento).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/visual_token_ablation_v1'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'
IMAGE_SIZE = (128, 32)  # W,H

# Coordenadas sobre la imagen REDIMENSIONADA 128 x 32, (x0, y0, x1, y1),
# limites x1/y1 EXCLUSIVOS. Provienen de los pasos 5 y 6B.
CASES = [
    {
        'id': 'P5_ABA1234', 'grupo': 'insercion_inicial',
        'image_path': 'images/train__ABA1234.JPG', 'gt': 'ABA1234',
        'ar_esperado': 'BABA1234', 'ref_esperado': 'BABA1234',
        'roi_nombre': 'banda_azul_izquierda', 'roi': (0, 4, 11, 30),
        'control_nombre': 'escudo_central', 'control': (40, 3, 54, 28),
        'pixel_objetivo_esperado': 'ABA1234', 'pixel_control_esperado': 'BABA1234',
    },
    {
        'id': 'P5_DANKE82', 'grupo': 'insercion_interna',
        'image_path': 'images/test__DANKE82.png', 'gt': 'DANKE82',
        'ar_esperado': 'DANSKE82', 'ref_esperado': 'DANSKE82',
        'roi_nombre': 'escudos_centrales', 'roi': (50, 8, 61, 28),
        'control_nombre': 'margen_izquierdo', 'control': (0, 8, 11, 28),
        'pixel_objetivo_esperado': 'DANKE82', 'pixel_control_esperado': 'DANSKE82',
    },
    {
        'id': 'E010', 'grupo': 'insercion_a_omision',
        'image_path': 'images/train__W584RP.png', 'gt': 'W584RP',
        'ar_esperado': 'WV584RP', 'ref_esperado': 'WV584RP',
        'roi_nombre': 'escudo_central', 'roi': (31, 8, 46, 21),
        'control_nombre': None, 'control': None,
        'pixel_objetivo_esperado': '584RP', 'pixel_control_esperado': None,
    },
    {
        'id': 'C027', 'grupo': 'correcta_empeora_con_mascara',
        'image_path': 'images/train__FOW653.png', 'gt': 'FOW653',
        'ar_esperado': 'FOW653', 'ref_esperado': 'FOW653',
        'roi_nombre': 'escudos_centrales', 'roi': (37, 4, 47, 27),
        'control_nombre': None, 'control': None,
        'pixel_objetivo_esperado': 'F0W653', 'pixel_control_esperado': None,
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_image(root: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.parts[0] != 'images':
        raise ValueError(f'Ruta no permitida: {rel}')
    full = (root / path).resolve()
    if not full.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera de ELPD: {rel}')
    if not full.is_file():
        raise FileNotFoundError(f'Falta imagen ELPD: {full}')
    return full


def normalized(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())


def edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ch in enumerate(a, 1):
        curr = [i]
        for j, other in enumerate(b, 1):
            curr.append(min(curr[j-1] + 1, previous[j] + 1,
                            previous[j-1] + int(ch != other)))
        previous = curr
    return previous[-1]


def load_image(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.BICUBIC)


def gray_mask(img, bbox):
    from PIL import Image
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 32):
        raise ValueError(f'ROI fuera de rango: {bbox}')
    out = img.copy()
    out.paste(Image.new('RGB', (x1-x0, y1-y0), (235,235,235)), (x0,y0))
    return out


def patch_overlap_weights(bbox, grid_h: int, grid_w: int,
                          patch_h: int, patch_w: int) -> dict[int, float]:
    """Peso geometrico = fraccion de superficie del parche cubierta por ROI.

    Al intervenir, mezcla original/donante (o media) en esa proporcion, para
    no reemplazar totalmente patches que apenas rozan una ROI.
    """
    x0, y0, x1, y1 = bbox
    result = {}
    for row in range(grid_h):
        for col in range(grid_w):
            px0, py0 = col*patch_w, row*patch_h
            px1, py1 = px0+patch_w, py0+patch_h
            overlap_w=max(0, min(x1,px1)-max(x0,px0))
            overlap_h=max(0, min(y1,py1)-max(y0,py0))
            if overlap_w and overlap_h:
                result[row*grid_w+col] = overlap_w*overlap_h/(patch_w*patch_h)
    if not result:
        raise ValueError(f'ROI no intersecta parches: {bbox}')
    return result


def mix_roi_memory(original, donor, weights: dict[int,float]):
    """Intervencion suave; no altera original ni donor."""
    import torch
    result=original.clone()
    idx=list(weights)
    alpha=torch.tensor([weights[i] for i in idx],device=original.device,
                       dtype=original.dtype).view(1,-1,1)
    result[:,idx,:]=(1-alpha)*original[:,idx,:]+alpha*donor[:,idx,:]
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def make_preview(case, img, patch_grid, target_ids, control_ids, out):
    """Muestra ROI RGB y celdas de tokens intervenidas sin tocar imagen fuente."""
    from PIL import Image, ImageDraw
    gw, gh, pw, ph = patch_grid
    scale = 7
    W, H = 128*scale, 32*scale
    canvas = Image.new('RGB', (W + 40, H*3 + 200), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((20, 10), f'{case["id"]} | GT={case["gt"]} | ref={case["ref_esperado"]}',
           font=font(18), fill=(24,41,59))
    variants = [
        ('Imagen original', img, None, None),
        ('ROI objetivo / parches seleccionados', img, case['roi'], target_ids),
    ]
    if control_ids:
        variants.append(('ROI control / parches seleccionados', img, case['control'], control_ids))
    else:
        variants.append(('Imagen con ROI gris (control de reproducibilidad)',
                         gray_mask(img, case['roi']), None, None))
    for k, (label, picture, bbox, indices) in enumerate(variants):
        y = 52 + k*(H+42)
        d.text((20, y), label, font=font(16), fill=(24,41,59))
        top = y+23
        canvas.paste(picture.resize((W,H), Image.Resampling.NEAREST), (20, top))
        if indices:
            for idx in indices:
                rr, cc = divmod(idx, gw)
                rect=(20+cc*pw*scale, top+rr*ph*scale,
                      20+(cc+1)*pw*scale-1, top+(rr+1)*ph*scale-1)
                d.rectangle(rect, outline=(245,163,36), width=2)
        if bbox:
            x0,y0,x1,y1=bbox
            d.rectangle((20+x0*scale,top+y0*scale,20+x1*scale-1,top+y1*scale-1),
                        outline=(208,37,37),width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out,optimize=True)


def forward_with_memory(model, core, tensor, memory_override, refine_iters: int):
    """Mantiene intacto forward original. Sustituye solo el tensor encoder->decoder.

    El hook devuelve un tensor nuevo; no escribe sobre la salida ni los parametros.
    """
    import torch
    old_refines = int(core.refine_iters)
    handle = None
    try:
        core.refine_iters = refine_iters
        if memory_override is not None:
            def substitute(_module, _args, output):
                if tuple(output.shape) != tuple(memory_override.shape):
                    raise RuntimeError(f'Encoder cambio forma: {tuple(output.shape)} vs '
                                       f'{tuple(memory_override.shape)}')
                if output.dtype != memory_override.dtype or output.device != memory_override.device:
                    raise RuntimeError('Tipo/device distinto en memoria intervenida')
                return memory_override
            handle = core.encoder.register_forward_hook(substitute)
        with torch.inference_mode():
            logits = model(tensor)
            probs = logits.softmax(-1)
            raw_texts, _ = model.tokenizer.decode(probs)
        raw = str(raw_texts[0])
        ids = probs.argmax(-1)[0].detach().cpu().tolist()
        eos_id = int(model.tokenizer.eos_id)
        eos = ids.index(eos_id)+1 if eos_id in ids else None
        return {
            'raw': raw,
            'normalizada': normalized(raw),
            'eos_1based': eos,
            'n_logits': int(logits.shape[1]),
            'edit_from_gt': None,
            'probabilities': probs[0].detach().float().cpu(),
        }
    finally:
        if handle is not None:
            handle.remove()
        core.refine_iters = old_refines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Validar las cuatro imagenes y crear ROI sin GPU')
    args = parser.parse_args()
    print('=== PASO 10 | PARSeq ELPD V17 | TOKENS VISUALES DEL ENCODER ===', flush=True)
    if args.out.exists():
        raise FileExistsError(f'No se sobrescribe una salida existente: {args.out}')
    sources = {}
    for case in CASES:
        path = safe_image(args.image_root, case['image_path'])
        sources[case['id']] = {'image': load_image(path), 'sha256': sha256(path)}
    if args.prepare_only:
        patch_grid=(16,8,8,4)  # configuracion oficial del checkpoint estandar
    else:
        import torch
        from torchvision import transforms as T
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA no disponible: ejecutar con GVIS: Ejecutar en GPU')
        os.environ.setdefault('TORCH_HOME',str(DEFAULT_TORCH_HOME))
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True).eval().cuda()
        core = model.model
        if not bool(core.decode_ar) or int(core.refine_iters) != 1:
            raise RuntimeError('Se esperaba decode_ar=True, refine_iters=1')
        if tuple(int(v) for v in model.hparams.img_size) != (32,128):
            raise RuntimeError('Resolucion inesperada: no es 32x128')
        pe = core.encoder.patch_embed
        patch_size = tuple(int(v) for v in pe.patch_size)
        if patch_size != (4,8):
            raise RuntimeError(f'Patch size distinto: {patch_size}')
        gh, gw = tuple(int(v) for v in pe.grid_size)
        if (gh,gw) != (8,16):
            raise RuntimeError(f'Grid inesperada: {gh,gw}')
        patch_grid=(gw,gh,patch_size[1],patch_size[0])
        to_tensor = T.Compose([T.ToTensor(),T.Normalize(0.5,0.5)])
        started = time.perf_counter()
        comparison=[]
        token_details=[]
        roi_records=[]
        n_cases = len(CASES)
        print(f'GPU: {torch.cuda.get_device_name(0)}; casos: {n_cases}', flush=True)
    gw,gh,pw,ph=patch_grid
    args.out.mkdir(parents=True)
    (args.out/'vistas_previas').mkdir()

    for case_num,case in enumerate(CASES,1):
        name=case['id']
        img=sources[name]['image']
        roi_weights=patch_overlap_weights(case['roi'],gh,gw,ph,pw)
        ctrl_weights=(patch_overlap_weights(case['control'],gh,gw,ph,pw)
                      if case['control'] is not None else {})
        roi_ids=list(roi_weights)
        ctrl_ids=list(ctrl_weights)
        if set(roi_ids) & set(ctrl_ids):
            raise RuntimeError(f'ROI objetivo/control superpuestas: {name}')
        make_preview(case,img,patch_grid,roi_ids,ctrl_ids,
                     args.out/'vistas_previas'/f'{name}.png')
        print(f'[{case_num:02d}/{len(CASES)}] {name} | ROI: {len(roi_ids)} patches'
              + (f' | control: {len(ctrl_ids)} patches' if ctrl_ids else ''),flush=True)
        if args.prepare_only:
            continue

        with torch.inference_mode():
            x=to_tensor(img).unsqueeze(0).cuda()
            masked_target=gray_mask(img,case['roi'])
            x_mask=to_tensor(masked_target).unsqueeze(0).cuda()
            mem_original=core.encode(x).detach()
            mem_target=core.encode(x_mask).detach()
            if tuple(mem_original.shape[:2])!=(1,gh*gw):
                raise RuntimeError(f'Encoder no devuelve {gh*gw} tokens: {mem_original.shape}')
            nonroi=[i for i in range(gw*gh) if i not in set(roi_ids)]
            mean_target=mem_original[:,nonroi,:].mean(dim=1,keepdim=True)
            patched_target_mean=mix_roi_memory(mem_original,
                mean_target.expand_as(mem_original),roi_weights)
            patched_target_donor=mix_roi_memory(mem_original,mem_target,roi_weights)
            interventions=[
                ('original', x, None,'baseline'),
                ('pixel_gris_objetivo',x_mask,None,'pixeles'),
                ('memory_roi_donante',x,patched_target_donor,'tokens'),
                ('memory_roi_media',x,patched_target_mean,'tokens'),
            ]
            if ctrl_ids:
                x_ctrl=to_tensor(gray_mask(img,case['control'])).unsqueeze(0).cuda()
                mem_ctrl=core.encode(x_ctrl).detach()
                nonctrl=[i for i in range(gw*gh) if i not in set(ctrl_ids)]
                mean_ctrl=mem_original[:,nonctrl,:].mean(dim=1,keepdim=True)
                patched_ctrl_mean=mix_roi_memory(mem_original,
                    mean_ctrl.expand_as(mem_original),ctrl_weights)
                patched_ctrl_donor=mix_roi_memory(mem_original,mem_ctrl,ctrl_weights)
                interventions.extend([
                    ('pixel_gris_control',x_ctrl,None,'control'),
                    ('memory_control_donante',x,patched_ctrl_donor,'control'),
                    ('memory_control_media',x,patched_ctrl_mean,'control'),
                ])
        case_results={}
        for variant,inp,patch_memory,kind in interventions:
            stage_outputs={}
            for stage,n_refine in (('AR',0),('REF',1)):
                pred=forward_with_memory(model,core,inp,patch_memory,n_refine)
                pred['edit_from_gt']=edit_distance(case['gt'],pred['normalizada'])
                stage_outputs[stage]=pred
                row={
                    'id':name, 'grupo':case['grupo'], 'ground_truth':case['gt'],
                    'variante':variant, 'tipo_intervencion':kind,
                    'etapa':stage, 'prediccion_raw':pred['raw'],
                    'prediccion_normalizada':pred['normalizada'],
                    'em':int(pred['normalizada']==case['gt']),
                    'distancia_edicion':pred['edit_from_gt'],
                    'eos_posicion_1based':pred['eos_1based'] or '',
                    'num_logits':pred['n_logits'],
                    'n_tokens_roi':len(roi_ids),
                    'n_tokens_control':len(ctrl_ids),
                    'roi':case['roi_nombre'],
                    'image_path':case['image_path'],
                }
                comparison.append(row)
                vocab=list(model.tokenizer._itos)
                probs=pred['probabilities']
                for j in range(probs.shape[0]):
                    vec=probs[j]
                    top=vec.topk(3)
                    token_details.append({
                        'id':name, 'variante':variant, 'etapa':stage,
                        'posicion_1based':j+1,
                        'token_argmax':vocab[int(top.indices[0])],
                        'p_argmax':round(float(top.values[0]),8),
                        'alternativas_top3':' | '.join(f'{vocab[int(t)]}:{float(p):.5f}'
                                                      for p,t in zip(top.values,top.indices)),
                        'p_EOS':round(float(vec[int(model.tokenizer.eos_id)]),8),
                    })
            case_results[variant]=stage_outputs

        # Fallar si no reproduce originales; si difieren pixel masks de
        # pasos previos, impedir atribuir los cambios a una intervencion.
        for stage, expected in (('AR',case['ar_esperado']),('REF',case['ref_esperado'])):
            got=case_results['original'][stage]['normalizada']
            if got!=expected:
                raise RuntimeError(f'{name} original {stage}: esperado {expected!r}, obtenido {got!r}')
        got_pixel=case_results['pixel_gris_objetivo']['REF']['normalizada']
        if got_pixel!=case['pixel_objetivo_esperado']:
            raise RuntimeError(f'{name} mascara objetivo no reproduce paso 5/6B: '
                               f'esperado {case["pixel_objetivo_esperado"]!r}, got {got_pixel!r}')
        if ctrl_ids and case['pixel_control_esperado'] is not None:
            got_ctrl=case_results['pixel_gris_control']['REF']['normalizada']
            if got_ctrl!=case['pixel_control_esperado']:
                raise RuntimeError(f'{name} mascara control no reproduce paso 5: '
                                   f'esperado {case["pixel_control_esperado"]!r}, got {got_ctrl!r}')
        print('   '+ ' | '.join(f'{v}: {case_results[v]["REF"]["normalizada"]}'
                                for v in case_results), flush=True)
        roi_records.append({
            'id':name, 'image_path':case['image_path'], 'sha256_imagen':sources[name]['sha256'],
            'bbox_objetivo':str(case['roi']), 'roi_nombre':case['roi_nombre'],
            'indices_patches_roi_0based':' '.join(map(str,roi_ids)),
            'fracciones_roi':' '.join(f'{roi_weights[i]:.4f}' for i in roi_ids),
            'bbox_control':str(case['control']) if case['control'] else '',
            'indices_patches_control_0based':' '.join(map(str,ctrl_ids)),
            'fracciones_control':' '.join(f'{ctrl_weights[i]:.4f}' for i in ctrl_ids),
        })

    if args.prepare_only:
        print('PREFLIGHT OK: 4 imagenes, ROI y vistas previas; inferencia NO ejecutada',flush=True)
        print('SALIDA:',args.out,flush=True)
        return

    write_csv(args.out/'comparacion_condiciones.csv',comparison)
    write_csv(args.out/'tokens_por_posicion.csv',token_details)
    write_csv(args.out/'regiones_y_tokens.csv',roi_records)
    bycase={}
    for case in CASES:
        a=[r for r in comparison if r['id']==case['id'] and r['etapa']=='REF']
        bycase[case['id']]={r['variante']:{'prediccion':r['prediccion_normalizada'],
                                        'exact_match':bool(r['em']),
                                        'ediciones':r['distancia_edicion']}
                           for r in a}
    summary={
        'paso':10,'modelo':'PARSeq original preentrenado',
        'dataset':'ELPD V17','casos':len(CASES),
        'casos_con_region_control':sum(bool(c['control']) for c in CASES),
        'predicciones_comparadas':len(comparison),
        'n_patches_encoder':gh*gw,'grid_filas_columnas':[gh,gw],
        'patch_size_hw':[ph,pw],
        'reproduccion_originales_y_mascaras':'OK',
        'resultados_refinados_por_caso':bycase,
        'segundos':round(time.perf_counter()-started,2),
        'pesos_modificados':False,'GT_modificado':False,
        'advertencias':[
            'Los tokens de salida del encoder ViT ya mezclan informacion de toda la imagen.',
            'Sustituir tokens del encoder es una intervencion en el interfaz encoder-decoder, NO exclusiva de MHA2.',
            'El trasplante de tokens de imagen gris conserva tokens globales originales en las otras posiciones.',
            'La sustitucion por media puede generar activaciones fuera de distribucion.',
            'Las ROI se definieron en 128x32. Cada parche se modifica en la proporcion de su solapamiento geometrico.',
            'Los parches de borde pueden incluir parte de caracteres aunque la ROI de pixeles no los tape.',
            'Solo cuatro casos elegidos; no se estiman tasas globales ni causalidad de capas individuales.'
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('RESULTADO: INTERVENCIONES EN TOKENS VISUALES COMPLETADAS',flush=True)
    print('SALIDA:',args.out,flush=True)

if __name__ == '__main__':
    main()
