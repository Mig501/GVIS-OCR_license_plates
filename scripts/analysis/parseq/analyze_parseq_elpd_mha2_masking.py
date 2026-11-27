#!/usr/bin/env python3
"""Paso 11 | PARSeq + ELPD V17 | Intervencion DIRECTA de atencion cruzada MHA2.

Ruta en el repositorio:
    scripts/analysis/parseq/analyze_parseq_elpd_mha2_masking.py

Ejecutar con la task GVIS: Ejecutar en GPU, sin argumentos.

Comparacion sobre cuatro matriculas de los pasos 5, 6B y 10:
  original                 : PARSeq intacto
  mha2_mascara_vacia       : prueba placebo de implementacion (ningun token oculto)
  pixel_gris_objetivo      : referencia experimental del paso anterior
  mha2_objetivo_bloqueo    : MHA2 NO puede consultar los tokens de la ROI
  mha2_objetivo_atenuacion : penalizacion aditiva -3 en esas conexiones
  mha2_control_*           : bloqueo/atenuacion de ROI control, si esta auditada

Se ejecutan AR (refine_iters=0) y REF (refine_iters=1) manteniendo los
pesos, la imagen original y la salida del ViT INTACTOS en condiciones MHA2.
La intervencion se implementa con un forward_pre_hook sobre nn.MultiheadAttention:
se pasa 'attn_mask' a cross_attn sin modificar source code ni pesos.

LIMITACIONES: cada token final del ViT contiene informacion global; el bloqueo
solo impide consultar esos tokens como claves/valores en MHA2, no borra su
informacion si persiste en otros tokens. Bloquear tokens es una intervencion
causal sobre el acceso de MHA2 a esa memoria, NO atribucion exclusiva a una capa.
Las ROI a nivel parche pueden rozar caracteres. No estimar mejoras globales.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
TORCH_HOME = DATASETS / 'models/torch_hub'
OUTPUT_DIR = PROJECT / 'results/analysis/PARSeq/ELPD/mha2_masking_v1'
IMAGE_SIZE = (128, 32)  # W, H, tras redimensionar: 16 columnas x 8 filas
PATCH_H, PATCH_W = 4, 8
OVERLAP_THRESHOLD = 0.50    # bloqueo de patches con >=50% de su area en ROI
SOFT_PENALTY = -3.0        # sesgo aditivo de atencion (no una probabilidad)

CASES = [
    dict(id='P5_ABA1234', gt='ABA1234', path='images/train__ABA1234.JPG',
         ar='BABA1234', ref='BABA1234', roi=(0, 4, 11, 30),
         roi_label='banda_azul_izquierda', control=(40, 3, 54, 28),
         pixel_ref='ABA1234'),
    dict(id='P5_DANKE82', gt='DANKE82', path='images/test__DANKE82.png',
         ar='DANSKE82', ref='DANSKE82', roi=(50, 8, 61, 28),
         roi_label='escudos_centrales', control=(0, 8, 11, 28),
         pixel_ref='DANKE82'),
    dict(id='E010', gt='W584RP', path='images/train__W584RP.png',
         ar='WV584RP', ref='WV584RP', roi=(31, 8, 46, 21),
         roi_label='escudo_central', control=None, pixel_ref='584RP'),
    dict(id='C027', gt='FOW653', path='images/train__FOW653.png',
         ar='FOW653', ref='FOW653', roi=(37, 4, 47, 27),
         roi_label='escudos_centrales', control=None, pixel_ref='F0W653'),
]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalizar(x):
    return re.sub(r'[^A-Z0-9]', '', str(x).upper())


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[-1]+1, prev[j]+1, prev[j-1]+int(ca!=cb)))
        prev = row
    return prev[-1]


def safe_image(root, rel):
    p = Path(rel)
    if p.is_absolute() or p.parts[0] != 'images' or len(p.parts) != 2 or '..' in p.parts:
        raise ValueError(f'Ruta relativa no segura: {rel}')
    result = (root / p).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise FileNotFoundError(f'No se encuentra imagen ELPD: {result}')
    return result


def read_image(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.BICUBIC)


def mask_gray(image, roi):
    from PIL import Image
    x0,y0,x1,y1 = roi
    result = image.copy()
    result.paste(Image.new('RGB',(x1-x0,y1-y0),(235,235,235)),(x0,y0))
    return result


def roi_indices(roi, threshold=OVERLAP_THRESHOLD):
    x0,y0,x1,y1 = roi
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 32):
        raise ValueError(f'ROI fuera de 128x32: {roi}')
    records=[]
    for row in range(8):
        for col in range(16):
            bx0,bx1=col*PATCH_W,(col+1)*PATCH_W
            by0,by1=row*PATCH_H,(row+1)*PATCH_H
            dx=max(0,min(x1,bx1)-max(x0,bx0))
            dy=max(0,min(y1,by1)-max(y0,by0))
            frac=dx*dy/(PATCH_W*PATCH_H)
            if frac >= threshold:
                records.append((row*16+col,round(frac,5)))
    if not records:
        raise ValueError(f'ROI no selecciona patches: {roi}, umbral={threshold}')
    if len(records)>=128:
        raise ValueError('No se puede bloquear toda la memoria visual')
    return records


def save_preview(image, case, target, control, path):
    from PIL import Image, ImageDraw, ImageFont
    scale=6
    original=image.resize((128*scale,32*scale),Image.Resampling.NEAREST)
    page=Image.new('RGB',(128*scale+24,32*scale*2+116),'white')
    drawer=ImageDraw.Draw(page)
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:
        font=ImageFont.load_default()
    drawer.text((12,7),f'{case["id"]}   GT={case["gt"]}  REF={case["ref"]}',font=font,fill=(22,39,54))
    page.paste(original,(12,35))
    y=35+32*scale+24
    page.paste(original,(12,y))
    for idx,_ in target:
        rr,cc=divmod(idx,16)
        drawer.rectangle((12+cc*8*scale,y+rr*4*scale,12+(cc+1)*8*scale-1,
                          y+(rr+1)*4*scale-1),outline=(239,98,32),width=3)
    if control:
        for idx,_ in control:
            rr,cc=divmod(idx,16)
            drawer.rectangle((12+cc*8*scale,y+rr*4*scale,12+(cc+1)*8*scale-1,
                              y+(rr+1)*4*scale-1),outline=(41,119,193),width=2)
    path.parent.mkdir(parents=True,exist_ok=True)
    page.save(path)


class MHA2Intervention:
    """Hook efimero sobre la MHA2. None=observacion, vacio=placebo.

    mode: 'none', 'noop', 'hard' o 'soft'.
    Guarda la masa media de atencion en los tokens seleccionados. El bloqueo
    se comprueba sobre las atenciones devueltas por la MHA.
    """
    def __init__(self, mha, indices, mode):
        self.mha=mha
        self.ids=list(indices)
        self.mode=mode
        self.handles=[]
        self.calls=0
        self.mass=[]
        self.max_mass=0.0
        self.q_positions=[]
        self.mass_by_position=[]

    def __enter__(self):
        if self.mode not in ('none','noop','hard','soft'):
            raise ValueError(self.mode)
        if self.mode!='none':
            self.handles.append(self.mha.register_forward_pre_hook(self.before,with_kwargs=True))
        self.handles.append(self.mha.register_forward_hook(self.after))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles=[]

    def before(self,module,args,kwargs):
        import torch
        if len(args)<3:
            raise RuntimeError('MHA2 debe recibir query, key y value como posicionales')
        q,k,v=args[:3]
        if q.ndim!=3 or k.ndim!=3 or v.ndim!=3 or q.shape[0]!=1 or k.shape[0]!=1:
            raise RuntimeError(f'Forma MHA2 inesperada: {q.shape}, {k.shape}, {v.shape}')
        if k.shape[1]!=128 or v.shape[1]!=128:
            raise RuntimeError(f'Memoria visual inesperada: {k.shape[1]} tokens')
        if kwargs.get('attn_mask') is not None:
            raise RuntimeError('MHA2 ya contiene attn_mask; no combinar con otra mascara')
        if self.mode=='hard':
            mask=torch.zeros((q.shape[1],128),dtype=torch.bool,device=q.device)
            mask[:,self.ids]=True  # True = posicion prohibida en PyTorch MHA
        else:
            mask=torch.zeros((q.shape[1],128),dtype=q.dtype,device=q.device)
            if self.mode=='soft':
                mask[:,self.ids]=SOFT_PENALTY
        kw=dict(kwargs)
        kw['attn_mask']=mask
        return args,kw

    def after(self,module,args,output):
        if not isinstance(output,tuple) or len(output)!=2 or output[1] is None:
            raise RuntimeError('MHA2 no ha devuelto pesos de atencion')
        weights=output[1]   # [batch=1, query_positions, keys=128], media cabezas
        if weights.ndim!=3 or weights.shape[0]!=1 or weights.shape[-1]!=128:
            raise RuntimeError(f'Pesos de MHA2 con forma inesperada: {weights.shape}')
        self.calls+=1
        self.q_positions.append(int(weights.shape[1]))
        if self.ids:
            selected=weights[:,:,self.ids].sum(-1)
            self.mass.append(float(selected.mean().detach()))
            self.max_mass=max(self.max_mass,float(selected.max().detach()))
            if weights.shape[1]==1:
                self.mass_by_position.append(float(selected[0,0].detach()))
            else:
                # En refinamiento se consulta la secuencia completa en una llamada.
                self.mass_by_position=[float(x) for x in selected[0].detach().cpu()]
            if self.mode=='hard' and selected.abs().max().item()>1e-6:
                raise RuntimeError('El bloqueo duro de MHA2 no suprimio la atencion objetivo')


def infer(model,core,tensor,indices,mode,refinements):
    import torch
    if core.refine_iters not in (0,1):
        raise RuntimeError('refine_iters inesperado')
    previous=int(core.refine_iters)
    mha=core.decoder.layers[0].cross_attn
    try:
        core.refine_iters=refinements
        with torch.inference_mode(), MHA2Intervention(mha,indices,mode) as intervention:
            logits=model(tensor)
            probs=logits.softmax(-1)
            strings,_=model.tokenizer.decode(probs)
        if intervention.calls==0:
            raise RuntimeError('No se ejecuto la MHA2 instrumentada')
        raw=str(strings[0])
        ids=probs.argmax(-1)[0].detach().cpu().tolist()
        eos_id=int(model.tokenizer.eos_id)
        eos=ids.index(eos_id)+1 if eos_id in ids else None
        top3=[]
        tokens=list(model.tokenizer._itos)
        for j,vector in enumerate(probs[0]):
            best=vector.topk(3)
            top3.append({
                'position_1based':j+1,
                'token':str(tokens[int(best.indices[0])]),
                'p_token':float(best.values[0]),
                'p_eos':float(vector[eos_id]),
                'top3':' | '.join(f'{tokens[int(ix)]}:{float(p):.6f}'
                                  for p,ix in zip(best.values,best.indices)),
            })
        if len(intervention.mass_by_position)!=int(logits.shape[1]):
            raise RuntimeError('No se pudo alinear atencion MHA2 con posiciones de logits: '
                               f'{len(intervention.mass_by_position)} vs {logits.shape[1]}')
        for r, m in zip(top3, intervention.mass_by_position):
            r['mha2_masa_en_region']=round(m,8)
        return {'raw':raw,'pred':normalizar(raw),'eos':eos,
                'num_logits':int(logits.shape[1]),
                'attn_calls':intervention.calls,
                'attn_mass_target':sum(intervention.mass)/len(intervention.mass) if intervention.mass else None,
                'attn_mass_max':intervention.max_mass,
                'probabilities':probs.detach().float().cpu(),
                'rows':top3}
    finally:
        core.refine_iters=previous


def save_csv(path,rows):
    if not rows:
        raise ValueError(f'No hay filas para {path}')
    with path.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0].keys()),extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def check_noop(original,placebo,case,stage):
    import torch
    if original['pred']!=placebo['pred'] or original['raw']!=placebo['raw']:
        raise RuntimeError(f'{case} {stage}: mascara vacia no reproduce baseline '
                           f'({original["raw"]!r} vs {placebo["raw"]!r})')
    if original['probabilities'].shape!=placebo['probabilities'].shape:
        raise RuntimeError(f'{case} {stage}: logits distintos con mascara vacia')
    delta=float((original['probabilities']-placebo['probabilities']).abs().max())
    if delta>2e-4:
        raise RuntimeError(f'{case} {stage}: mascara vacia cambia probabilidades (max={delta})')
    return delta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root',type=Path,default=IMAGE_ROOT)
    parser.add_argument('--out',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--prepare-only',action='store_true',help='Solo comprobar imagenes, ROI y vistas previas (sin GPU)')
    args=parser.parse_args()
    print('=== PASO 11 | PARSeq ELPD V17 | MHA2: BLOQUEO DE ATENCION CRUZADA ===',flush=True)
    if (args.out/'resumen.json').exists():
        raise FileExistsError(f'Analisis completo existente; no se sobrescribe: {args.out}')

    sources=[]
    for case in CASES:
        path=safe_image(args.image_root,case['path'])
        image=read_image(path)
        target=roi_indices(case['roi'])
        control=roi_indices(case['control']) if case['control'] else []
        if set(i for i,_ in target) & set(i for i,_ in control):
            raise RuntimeError(f'Patches objetivo y control solapados: {case["id"]}')
        sources.append((case,path,image,target,control))

    if not args.prepare_only:
        import torch
        from torchvision import transforms as T
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA no disponible; usar GVIS: Ejecutar en GPU')
        torch.manual_seed(42)
        torch.set_num_threads(2)
        os.environ.setdefault('TORCH_HOME',str(TORCH_HOME))
        model=torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
        core=model.model
        if not bool(core.decode_ar) or int(core.refine_iters)!=1:
            raise RuntimeError('El checkpoint debe usar decode_ar=True, refine_iters=1')
        if tuple(int(v) for v in model.hparams.img_size)!=(32,128):
            raise RuntimeError('El checkpoint debe usar imagenes de 32x128')
        if len(core.decoder.layers)!=1:
            raise RuntimeError('Este protocolo solo esta validado para el decoder PARSeq de 1 capa')
        patch=core.encoder.patch_embed
        if tuple(int(v) for v in patch.patch_size)!=(4,8) or tuple(int(v) for v in patch.grid_size)!=(8,16):
            raise RuntimeError('Distribucion de patches distinta de 8x16')
        mha=core.decoder.layers[0].cross_attn
        if not isinstance(mha,torch.nn.MultiheadAttention) or not mha.batch_first:
            raise RuntimeError('La segunda MHA no coincide con nn.MultiheadAttention(batch_first=True)')
        to_tensor=T.Compose([T.ToTensor(),T.Normalize(0.5,0.5)])
        comparison=[]
        positions=[]
        t0=time.perf_counter()
        print('GPU:',torch.cuda.get_device_name(0),flush=True)

    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'vistas_previas').mkdir(exist_ok=True)
    regions=[]
    for counter,(case,path,img,target,control) in enumerate(sources,1):
        name=case['id']
        target_idx=[i for i,_ in target]
        control_idx=[i for i,_ in control]
        print(f'[{counter:02d}/04] {name}: {len(target_idx)} patches objetivo'
              + (f'; {len(control_idx)} control' if control_idx else ''),flush=True)
        save_preview(img,case,target,control,args.out/'vistas_previas'/f'{name}.png')
        regions.append({
            'id':name,'image_path':case['path'],'sha256_image':sha256(path),
            'roi_label':case['roi_label'],'roi_128x32':str(case['roi']),
            'indices_objetivo': ' '.join(map(str,target_idx)),
            'fraccion_objetivo':' '.join(str(frac) for _,frac in target),
            'control_128x32':str(case['control']) if case['control'] else '',
            'indices_control':' '.join(map(str,control_idx)),
            'fraccion_control':' '.join(str(frac) for _,frac in control),
            'umbral_solapamiento':OVERLAP_THRESHOLD,
        })
        if args.prepare_only:
            continue

        with torch.inference_mode():
            inp=to_tensor(img).unsqueeze(0).cuda()
            gray=to_tensor(mask_gray(img,case['roi'])).unsqueeze(0).cuda()

        conditions=[
            ('original', inp, target_idx, 'none', 'baseline'),
            ('mha2_mascara_vacia',inp,target_idx,'noop','placebo'),
            ('pixel_gris_objetivo',gray,target_idx,'none','pixeles'),
            ('mha2_objetivo_bloqueo',inp,target_idx,'hard','mha2'),
            ('mha2_objetivo_atenuacion',inp,target_idx,'soft','mha2'),
        ]
        if control_idx:
            conditions.extend([
                ('mha2_control_bloqueo',inp,control_idx,'hard','control'),
                ('mha2_control_atenuacion',inp,control_idx,'soft','control'),
            ])
        case_stage={}
        for label,tensor,idx,mode,kind in conditions:
            case_stage[label]={}
            for stage,nref in [('AR',0),('REF',1)]:
                pred=infer(model,core,tensor,idx,mode,nref)
                case_stage[label][stage]=pred
                comparison.append({
                    'id':name, 'gt':case['gt'], 'image_path':case['path'],
                    'condicion':label,'tipo':kind,'etapa':stage,
                    'prediction_raw':pred['raw'],
                    'prediction_norm':pred['pred'],
                    'exact_match':int(pred['pred']==case['gt']),
                    'ediciones':levenshtein(case['gt'],pred['pred']),
                    'eos_1based':pred['eos'] or '',
                    'num_logits':pred['num_logits'],
                    'n_patches_bloqueados':len(idx) if mode in ('hard','soft') else 0,
                    'mha2_attn_calls':pred['attn_calls'],
                    'mha2_masa_media_en_region':round(pred['attn_mass_target'],8)
                        if pred['attn_mass_target'] is not None else '',
                    'mha2_masa_max_en_region':round(pred['attn_mass_max'],8),
                })
                for pr in pred['rows']:
                    positions.append({'id':name,'condicion':label,'etapa':stage,**pr})

        for stage,expected in [('AR',case['ar']),('REF',case['ref'])]:
            actual=case_stage['original'][stage]['pred']
            if actual!=expected:
                raise RuntimeError(f'{name}: original {stage} esperado={expected}, obtenido={actual}')
            delta=check_noop(case_stage['original'][stage],case_stage['mha2_mascara_vacia'][stage],name,stage)
            print(f'  {stage}: placebo max diferencia de probabilidades={delta:.2g}',flush=True)
        pixel=case_stage['pixel_gris_objetivo']['REF']['pred']
        if pixel!=case['pixel_ref']:
            raise RuntimeError(f'{name}: imagen gris no reproduce paso previo: '
                               f'esperado={case["pixel_ref"]}, obtenido={pixel}')
        print('   REF: '+' | '.join(f'{label}={result["REF"]["pred"]}'
                                  for label,result in case_stage.items()),flush=True)

    save_csv(args.out/'regiones.csv',regions)
    if args.prepare_only:
        print('PREFLIGHT OK: 4 imagenes y propuestas de bloqueo MHA2; GPU no utilizada',flush=True)
        print('SALIDA:',args.out,flush=True)
        return

    save_csv(args.out/'comparacion_condiciones.csv',comparison)
    save_csv(args.out/'probabilidades_por_posicion.csv',positions)
    summary={
        'paso':11,
        'modelo':'PARSeq oficial pretrained',
        'dataset':'ELPD V17',
        'casos':len(CASES),
        'n_comparaciones':len(comparison),
        'num_refinamientos_por_etapa':{'AR':0,'REF':1},
        'umbral_fraccion_patch':OVERLAP_THRESHOLD,
        'penalizacion_soft_logits_atencion':SOFT_PENALTY,
        'n_patches_visual':128,
        'n_cabezas_MHA2':int(mha.num_heads),
        'pesos_modificados':False,
        'imagen_original_modificada_en_experimentos_MHA2':False,
        'reproducciones_originales_y_pixeles':'OK',
        'placebo_mascara_vacia':'OK',
        'resultados_ref':{
            case['id']:{row['condicion']:row['prediction_norm']
                        for row in comparison if row['id']==case['id'] and row['etapa']=='REF'}
            for case in CASES
        },
        'duracion_segundos':round(time.perf_counter()-t0,2),
        'advertencias':[
            'La mascara se inyecta SOLO en cross_attn (MHA2) del decoder, no en MHA1.',
            'La memoria visual y el ViT permanecen intactos en condiciones MHA2.',
            'Una misma representacion del ViT puede contener contexto de multiples regiones.',
            'Bloquear patches que solapan parcialmente ROI puede interferir con caracteres.',
            'Los pesos de atencion observados no prueban causalidad de un unico componente.',
            'Solo cuatro ejemplos seleccionados; no estimar mejoria poblacional.',
        ],
    }
    (args.out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULTADO: MHA2 DIRECTA ANALIZADA; SALIDAS EXPORTADAS',flush=True)
    print('SALIDA:',args.out,flush=True)


if __name__=='__main__':
    main()
