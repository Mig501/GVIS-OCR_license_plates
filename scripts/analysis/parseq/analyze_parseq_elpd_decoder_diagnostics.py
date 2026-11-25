#!/usr/bin/env python3
"""PASO 7 | PARSeq ELPD V17 | diagnostico del decoder (MHA1/MHA2, logits, EOS).

Ruta: scripts/analysis/parseq/analyze_parseq_elpd_decoder_diagnostics.py
Task: GVIS: Ejecutar en GPU, sin argumentos.

Analiza E006, E010 y C027, cada uno original/roi gris/roi desenfocada,
reutilizando EXACTAMENTE las imagenes 128x32 guardadas en el Paso 6B.

Captura las dos MHA del decoder mediante forward hooks no invasivos:
- self_attn: atencion al contexto (MHA1)
- cross_attn: atencion visual (MHA2)

Registra las probabilidades de cada token, especialmente EOS, en AR y
refinamiento. Verifica las 18 predicciones contra el CSV del Paso 6B.

El hook no cambia los pesos ni hace una atribucion causal. Los pesos de
atencion de nn.MultiheadAttention estan promediados entre cabezas.

Preflight sin CUDA:
  python scripts/analysis/parseq/analyze_parseq_elpd_decoder_diagnostics.py --prepare-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
PREVIOUS = PROJECT / 'results/analysis/PARSeq/ELPD/cohorte_intervenciones_gpu_v1'
DEFAULT_CSV = PREVIOUS / 'comparacion_intervenciones.csv'
DEFAULT_IMAGES = PREVIOUS / 'imagenes_128x32'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/decoder_diagnostics_v1'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'
IDS = ('E006', 'E010', 'C027')
VARIANTS = ('original', 'relleno_gris', 'desenfoque')
# Posiciones interesantes en los LOGITS de salida (0-based).
# Es importante: la misma posicion no implica el mismo caracter bajo intervencion.
FOCUS = {'E006': (0, 1, 2, 7), 'E010': (0, 1, 2, 3, 6), 'C027': (0, 1, 2, 3, 6)}
HEATMAP_POS = {'E006': 0, 'E010': 1, 'C027': 1}


def norm_plate(value: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(value).upper())


def csv_write(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_previous(path: Path, image_dir: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'id', 'image_path', 'GT', 'variante', 'AR', 'refinado', 'region', 'grupo'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'CSV Paso 6B incompleto: faltan {sorted(required - set(rows[0] if rows else []))}')
    subset = {}
    for r in rows:
        if r['id'] not in IDS:
            continue
        key = (r['id'], r['variante'])
        if key in subset:
            raise ValueError(f'Registro duplicado: {key}')
        subset[key] = r
    expected = {(case, mode) for case in IDS for mode in VARIANTS}
    if set(subset) != expected:
        raise ValueError(f'Faltan/sobran pares: {sorted(expected-set(subset))} {sorted(set(subset)-expected)}')
    for id_ in IDS:
        entries = [subset[id_, mode] for mode in VARIANTS]
        if len({r['GT'] for r in entries}) != 1 or len({r['image_path'] for r in entries}) != 1:
            raise ValueError(f'GT/ruta inconsistente entre variantes {id_}')
        for mode in VARIANTS:
            file = image_dir / f'{id_}__{mode}.png'
            if not file.is_file():
                raise FileNotFoundError(f'Falta imagen del Paso 6B: {file}')
    return subset


def class_change(gt, ar, refined):
    if ar == gt and refined != gt:
        return 'introducido_en_refinamiento'
    if ar != gt and refined == gt:
        return 'corregido_por_refinamiento'
    if ar == gt and refined == gt:
        return 'correcto_en_ambas'
    if ar == refined:
        return 'error_igual_en_ambas'
    return 'error_modificado_en_refinamiento'


def summarize_tokens(probs, itos: list[str], eos_id: int, pos: int):
    import torch
    p = probs[pos]
    vals, indices = torch.topk(p, min(5, len(p)))
    top = [(itos[int(i)], float(v)) for v, i in zip(vals, indices)]
    entropy = float(-(p * p.clamp_min(1e-15).log()).sum())
    return {
        'token_ganador': top[0][0],
        'p_ganador': round(top[0][1], 8),
        'top5': '|'.join(f'{name}:{prob:.6f}' for name, prob in top),
        'p_EOS': round(float(p[eos_id]), 8),
        'margen_top1_top2': round(top[0][1]-top[1][1], 8) if len(top)>1 else '',
        'entropia_nats': round(entropy, 6),
    }


def eos_pos(probs, eos_id):
    ids = probs.argmax(-1).tolist()
    return ids.index(eos_id) + 1 if eos_id in ids else None


def _get_weights(output):
    # Repo oficial PARSeq usa MultiheadAttention(...), need_weights=True.
    if not isinstance(output, (tuple, list)) or len(output) < 2 or output[1] is None:
        raise RuntimeError('El modulo MHA no expone pesos de atencion (revision de version requerida)')
    w = output[1].detach().float().cpu()
    if w.ndim != 3 or w.shape[0] != 1:
        raise RuntimeError(f'Forma inesperada en MHA: {tuple(w.shape)}')
    return w[0]  # [queries, keys], media de las cabezas


def run_model(model, core, x, iters: int, observe=False):
    import torch
    record = {'contexto': [], 'visual': []}
    handles = []
    if observe:
        layer = core.decoder.layers[-1]
        handles.append(layer.self_attn.register_forward_hook(
            lambda _m, _inp, out: record['contexto'].append(_get_weights(out))))
        handles.append(layer.cross_attn.register_forward_hook(
            lambda _m, _inp, out: record['visual'].append(_get_weights(out))))
    saved = int(core.refine_iters)
    try:
        core.refine_iters = iters
        with torch.inference_mode():
            logits = model(x)
            probabilities = logits.softmax(-1).detach().cpu()[0]
            text, _ = model.tokenizer.decode(probabilities.unsqueeze(0))
    finally:
        core.refine_iters = saved
        for h in handles:
            h.remove()
    return norm_plate(text[0]), probabilities, record


def extract_attention(record, stage, pos, ar_probs):
    # Cuando refine_iters=1, los N primeros registros son los N pasos AR
    # y el ultimo registro es la pasada paralela del refinamiento.
    n_steps = int(ar_probs.shape[0])
    name = 'contexto'
    if len(record['contexto']) != n_steps+1 or len(record['visual']) != n_steps+1:
        raise RuntimeError(f'Numero de hooks inesperado: AR={n_steps}, '
                           f'contexto={len(record["contexto"])}, visual={len(record["visual"])}')
    idx = pos if stage == 'AR' else n_steps
    q = 0 if stage == 'AR' else pos
    sa = record['contexto'][idx]
    ca = record['visual'][idx]
    if q >= sa.shape[0] or q >= ca.shape[0]:
        raise ValueError('Posicion de consulta fuera de rango')
    w_sa, w_ca = sa[q], ca[q]
    if w_ca.numel() != 128:
        raise RuntimeError(f'Grid inesperada: {w_ca.numel()} parches; se esperaban 128')
    # ATENCION: el paso AR j solo ve BOS y tokens AR anteriores; el
    # refinamiento puede consultar otros tokens del contexto segun su mascara.
    return w_sa, w_ca


def describe_context_key(k: int, ar_probs, itos):
    if k == 0:
        return '[BOS]'
    idx = k-1
    if idx >= len(ar_probs):
        return '[FUERA]'
    return f'{itos[int(ar_probs[idx].argmax())]} (AR-pos-{idx+1})'


def heat_overlay(base, weights, pos, label, scale=4):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    if not len(weights) == 128:
        raise ValueError('Atencion visual debe tener 128 elementos')
    # El color es normalizado DENTRO de cada mapa. No comparar intensidad
    # absoluta entre imagenes ni tratar como atribucion causal.
    mat = np.asarray(weights, dtype=np.float32).reshape(8, 16)
    peak = float(mat.max())
    weights_norm = mat / peak if peak > 0 else mat
    frame = base.copy().convert('RGBA').resize((128*scale, 32*scale), Image.Resampling.NEAREST)
    tiny = np.zeros((8,16,4),dtype=np.uint8)
    tiny[...,0],tiny[...,1],tiny[...,2] = 250,103,32
    tiny[...,3] = np.minimum(205, np.power(weights_norm, 0.7) * 205).astype('uint8')
    # En PIL el resize bilinear suaviza la cuadricula 8x16 para mostrarla.
    over = Image.fromarray(tiny, 'RGBA').resize(frame.size, Image.Resampling.BILINEAR)
    frame = Image.alpha_composite(frame, over).convert('RGB')
    return frame


def make_visual_sheet(case, variants, plots, out):
    from PIL import Image, ImageDraw, ImageFont
    width,height=1160, 775
    page=Image.new('RGB',(width,height),'white')
    d=ImageDraw.Draw(page)
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
        titlefont=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    except OSError:
        font=ImageFont.load_default();titlefont=font
    d.text((22,12),f'PARSeq ELPD V17 | Paso 7 | {case} | MHA2 visual (media 12 cabezas)',fill='#1D3546',font=titlefont)
    d.text((22,44),'La posicion de salida es fija; el caracter puede cambiar. Escala de color POR MAPA.',fill='#556576',font=font)
    for j, mode in enumerate(VARIANTS):
        rowy=92+j*215
        d.text((22,rowy),f'{mode} | AR={plots[(mode,"AR")]["pred"]} | refinado={plots[(mode,"refinado")]["pred"]}',fill='#203647',font=font)
        for col,stage in enumerate(('AR','refinado')):
            info=plots[(mode,stage)]
            base=variants[mode]
            img=heat_overlay(base,info['visual'],info['pos'],stage)
            x=24+col*560
            page.paste(img,(x,rowy+32))
            d.text((x,rowy+165),f'{stage}: pos={info["pos"]+1}, token={info["token"]}, p(EOS)={info["p_eos"]:.5f}',fill='#425B6A',font=font)
    out.parent.mkdir(parents=True,exist_ok=True)
    page.save(out,optimize=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous-csv',type=Path,default=DEFAULT_CSV)
    ap.add_argument('--images',type=Path,default=DEFAULT_IMAGES)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()

    previous=read_previous(args.previous_csv,args.images)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'No sobrescribir resultados de analisis anteriores: {args.out}')
    print('=== PASO 7 | PARSEQ ELPD V17 | DIAGNOSTICO DEL DECODER ===',flush=True)
    print('Casos: E006 (insercion tras refinamiento), E010 (insercion->omision), C027 (control empeora)',flush=True)
    print('Variantes: original + relleno_gris + desenfoque = 9 imagenes',flush=True)
    if args.prepare_only:
        print('PREFLIGHT OK: CSV y las 9 imagenes 128x32 del Paso 6B localizados. SIN GPU.',flush=True)
        return

    import torch
    from PIL import Image
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; ejecutar con GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME',str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model=torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core=model.model
    if not bool(core.decode_ar) or int(core.refine_iters)!=1:
        raise RuntimeError('Se esperaba modelo PARSeq: AR + 1 refinamiento')
    if tuple(int(v) for v in model.hparams.img_size)!=(32,128):
        raise RuntimeError('El checkpoint no usa 32x128')
    if len(core.decoder.layers) != 1:
        raise RuntimeError('Numero de capas del decoder distinto del checkpoint esperado')
    itos=list(model.tokenizer._itos)
    eos_id=int(model.tokenizer.eos_id)
    transform=T.Compose([T.ToTensor(),T.Normalize(0.5,0.5)])

    stage_rows=[]; token_rows=[]; ctx_rows=[]; vis_rows=[]; plots_by_case={}
    originals_reproduced=0
    count=0
    for case in IDS:
        variants={}
        viz={}
        for mode in VARIANTS:
            row=previous[(case,mode)]
            file=args.images/f'{case}__{mode}.png'
            with Image.open(file) as im:
                image=im.convert('RGB')
                if image.size!=(128,32):
                    raise ValueError(f'{file.name} no tiene resolucion 128x32')
            variants[mode]=image
            x=transform(image).unsqueeze(0).cuda()
            pred_ar,p_ar,_=run_model(model,core,x,0,observe=False)
            pred_ref,p_ref,hooks=run_model(model,core,x,1,observe=True)
            if pred_ar!=row['AR'] or pred_ref!=row['refinado']:
                raise RuntimeError(f'No reproduce Paso 6B {case}/{mode}: '
                                   f'AR={pred_ar!r} esperado={row["AR"]!r}; '
                                   f'refinado={pred_ref!r} esperado={row["refinado"]!r}')
            originals_reproduced+=1
            # PARSeq permite parada temprana de AR al generar EOS. Sin embargo,
            # el refinamiento consulta todas las posiciones configuradas
            # (max_label_length + 1). No deben exigirse longitudes iguales.
            n_ar=int(p_ar.shape[0])
            n_ref=int(p_ref.shape[0])
            if p_ar.shape[1] != p_ref.shape[1]:
                raise RuntimeError(f'Vocabulario de logits distinto: AR={tuple(p_ar.shape)}, '
                                   f'refinado={tuple(p_ref.shape)}')
            if not (1 <= n_ar <= n_ref <= int(core.max_label_length)+1):
                raise RuntimeError(f'Longitudes de logits inesperadas: AR={n_ar}, '
                                   f'refinado={n_ref}, max={int(core.max_label_length)+1}')
            # Hooks de la segunda inferencia: las n_ar primeras llamadas
            # corresponden a pasos AR; la ultima es el refinamiento paralelo.
            if len(hooks['contexto'])!=n_ar+1 or len(hooks['visual'])!=n_ar+1:
                raise RuntimeError(f'Numero de llamadas del decoder inesperado: '
                                   f'AR={n_ar}, MHA1={len(hooks["contexto"])}, '
                                   f'MHA2={len(hooks["visual"])}')
            if len(itos)<p_ar.shape[-1]:
                raise RuntimeError('Vocabulario del checkpoint incompatible con logits')
            stage_rows.append({
                'id':case,'variante':mode,'GT':row['GT'],'region':row['region'],
                'AR':pred_ar,'refinado':pred_ref,
                'comparacion_etapas':class_change(row['GT'],pred_ar,pred_ref),
                'EOS_primero_AR_1based':eos_pos(p_ar,eos_id) or '',
                'EOS_primero_refinado_1based':eos_pos(p_ref,eos_id) or '',
                'n_logit_positions_AR':n_ar,
                'n_logit_positions_refinado':n_ref,
                'n_logit_positions_compartidas':min(n_ar,n_ref),
                'reproduce_paso6b':1,
            })
            for stage,probs,pred in [('AR',p_ar,pred_ar),('refinado',p_ref,pred_ref)]:
                # Cada etapa tiene su propio numero de posiciones: la salida
                # refinada puede contener posiciones sin equivalente en AR.
                n_stage=int(probs.shape[0])
                for pos in range(n_stage):
                    data=summarize_tokens(probs,itos,eos_id,pos)
                    token_rows.append({'id':case,'variante':mode,'etapa':stage,
                                       'posicion_1based':pos+1,'GT':row['GT'],
                                       'prediccion_etapa':pred,**data})
                for pos in FOCUS[case]:
                    if pos>=n_stage:continue
                    sa,ca=extract_attention(hooks,stage,pos,p_ar)
                    k=min(5,len(sa))
                    top_ctx=sorted(range(len(sa)),key=lambda i:float(sa[i]),reverse=True)[:k]
                    for key in top_ctx:
                        ctx_rows.append({
                            'id':case,'variante':mode,'etapa':stage,'posicion_1based':pos+1,
                            'clave_contexto_index_0based':key,
                            'clave_contexto':describe_context_key(key,p_ar,itos),
                            'peso_medio_12_cabezas':round(float(sa[key]),8),
                            'nota':'posicion de query; tokens de AR como contexto, NO importancia causal',
                        })
                    top_visual=sorted(range(len(ca)),key=lambda i:float(ca[i]),reverse=True)[:5]
                    for patch in top_visual:
                        vis_rows.append({
                            'id':case,'variante':mode,'etapa':stage,'posicion_1based':pos+1,
                            'patch_id_0based':patch,'fila_patch_0based':patch//16,
                            'columna_patch_0based':patch%16,
                            'peso_medio_12_cabezas':round(float(ca[patch]),8),
                            'nota':'atencion visual promedio, NO atribucion causal',
                        })
                selected_pos=HEATMAP_POS[case]
                if selected_pos<n_stage:
                    sa,ca=extract_attention(hooks,stage,selected_pos,p_ar)
                    viz[(mode,stage)]={'pred':pred,'pos':selected_pos,
                        'token':itos[int(probs[selected_pos].argmax())],
                        'p_eos':float(probs[selected_pos,eos_id]),
                        'visual':ca.tolist()}
            count+=1
            print(f'[{count:02d}/09] {case} {mode}: AR={pred_ar} ({n_ar} logits) | '
                  f'refinado={pred_ref} ({n_ref} logits)',flush=True)
        plots_by_case[case]=(variants,viz)

    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'mapas_atencion').mkdir()
    csv_write(args.out/'comparacion_etapas.csv',stage_rows,list(stage_rows[0]))
    csv_write(args.out/'probabilidades_por_posicion.csv',token_rows,list(token_rows[0]))
    csv_write(args.out/'top5_atencion_contexto_MHA1.csv',ctx_rows,list(ctx_rows[0]))
    csv_write(args.out/'top5_atencion_visual_MHA2.csv',vis_rows,list(vis_rows[0]))
    for case,(variants,viz) in plots_by_case.items():
        make_visual_sheet(case,variants,viz,args.out/'mapas_atencion'/f'{case}_comparacion.png')
    counts=Counter(r['comparacion_etapas'] for r in stage_rows)
    summary={
        'paso':7,'dataset':'ELPD GT V17','modelo':'PARSeq pretrained',
        'casos':list(IDS),'variantes_por_caso':list(VARIANTS),
        'comparaciones_reproducidas_paso6b':originals_reproduced,
        'total_comparaciones':len(stage_rows),'recuento_cambio_AR_refinamiento':dict(counts),
        'salidas':['comparacion_etapas.csv','probabilidades_por_posicion.csv',
                   'top5_atencion_contexto_MHA1.csv','top5_atencion_visual_MHA2.csv',
                   'mapas_atencion/'],
        'observaciones_metodologicas':[
            'self_attn y cross_attn son las dos MultiheadAttention del decoder oficial de PARSeq.',
            'Atencion de ambas MHA promediada entre las cabezas; NO equivale a causalidad.',
            'Se compara la misma posicion de salida, que puede corresponder a distintos caracteres tras una intervencion.',
            'Las distribuciones se calculan tanto para AR (0 refinamientos) como para AR + un refinamiento.',
            'AR puede detenerse al predecir EOS y devolver menos logits; el refinamiento consulta todas las posiciones configuradas.',
            'Las posiciones solo presentes en refinamiento no tienen equivalente AR y no deben compararse posicion a posicion.',
            'Si la prediccion refinada no contiene EOS, puede deberse a la longitud de logits heredada de la pasada AR.',
            'Las ROI son preliminares y el diagnostico se limita a tres imagenes seleccionadas.',
            'No se modifican pesos, GT ni imagenes originales.',
        ]}
    (args.out/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'RESULTADO: {originals_reproduced}/9 comparaciones reproducidas; '
          'MHA1, MHA2, logits y EOS exportados',flush=True)
    print('SALIDA:',args.out,flush=True)


if __name__=='__main__':
    main()
