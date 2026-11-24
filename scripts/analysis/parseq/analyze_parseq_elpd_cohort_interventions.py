#!/usr/bin/env python3
"""PASO 6B | PARSeq ELPD V17 | intervenciones de ROI sobre cohorte revisada.

Ruta en proyecto:
  scripts/analysis/parseq/analyze_parseq_elpd_cohort_interventions.py

Task GPU (sin argumentos): aplica mascaras a 31 ROI preliminares de la
cohorte V17 y compara 19 errores de insercion con 12 controles correctos.

Preflight opcional SIN GPU:
  python scripts/analysis/parseq/analyze_parseq_elpd_cohort_interventions.py --prepare-only

IMPORTANTE: las ROI proceden de inspeccion visual de la cohorte y son
propuestas; revisar todas las previsualizaciones antes de interpretar los
resultados. Este script no modifica los pesos, el GT ni las imagenes originales.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_SELECTION = PROJECT / 'results/analysis/PARSeq/ELPD/cohorte_intervenciones_v1/seleccion_casos.csv'
DEFAULT_IMAGES = DATASETS / 'processed/ELPD_LPO'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/cohorte_intervenciones_gpu_v1'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'

# Coordenadas (x0, y0, x1, y1) de la imagen del checkpoint: 128x32; x1,y1 exclusivos.
# "No ROI" significa que no corresponde intervenir en este experimento; las
# imagenes no desaparecen de la cohorte documental, pero no se infieren aqui.
ROI = {
    'E001': ('banda_izquierda', (10,5,22,23)),
    'E002': ('banda_izquierda', (13,1,25,22)),
    'E003': ('banda_izquierda', (2,8,15,25)),
    'E004': ('banda_izquierda', (6,7,17,27)),
    'E005': ('banda_izquierda', (7,5,18,28)),
    'E006': ('banda_izquierda', (4,5,16,27)),
    'E007': ('banda_izquierda', (2,7,16,29)),
    'E008': ('banda_izquierda', (4,7,16,25)),
    'E010': ('escudo_central', (31,8,46,21)),
    'E013': ('escudo_central', (43,11,54,26)),
    'E015': ('escudos_centrales', (50,9,59,26)),
    'E016': ('escudos_centrales', (38,5,49,25)),
    'E017': ('grafico_derecho', (99,7,121,28)),
    'E018': ('banda_derecha', (115,4,126,28)),
    'E019': ('escudo_derecho', (110,9,122,26)),
    'E021': ('escudo_derecho', (109,9,123,25)),
    'E022': ('banda_derecha', (108,6,119,25)),
    'E023': ('banda_derecha', (114,13,125,28)),
    'E024': ('banda_derecha', (111,5,124,25)),
    'C026': ('banda_izquierda', (6,6,16,26)),
    'C027': ('escudos_centrales', (37,4,47,27)),
    'C028': ('banda_izquierda', (0,10,12,27)),
    'C029': ('escudo_izquierdo', (8,9,17,26)),
    'C030': ('banda_izquierda', (0,4,10,26)),
    'C033': ('banda_derecha', (112,8,123,27)),
    'C034': ('banda_izquierda', (5,5,16,26)),
    'C035': ('banda_derecha', (112,8,122,26)),
    'C036': ('banda_derecha', (110,9,122,29)),
    'C037': ('sello_central', (58,7,69,21)),
    'C038': ('banda_derecha', (111,9,124,28)),
    'C039': ('banda_izquierda', (3,9,15,27)),
}
EXCLUDED = {
    'E009': 'Inserciones simultaneas en extremos distintos; experimento factorial necesario',
    'E011': 'Inserciones en dos posiciones; banda + sellos; experimento factorial necesario',
    'E012': 'Repeticion de 8, alineacion ambigua',
    'E014': 'Inserciones inicial y final; intervencion de dos bandas necesaria',
    'E020': 'ROI no aislable con seguridad en imagen muy pequena',
    'C025': 'No hay elemento grafico seguro que enmascarar',
    'C031': 'No hay elemento grafico seguro que enmascarar',
    'C032': 'Imagen demasiado oscura/desenfocada',
    'C040': 'ROI pendiente de cotejar con la imagen 128x32 original',
}
assert len(ROI) == 31 and len(EXCLUDED) == 9
MODES = ('original', 'relleno_gris', 'desenfoque')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_image(root: Path, image_path: str) -> Path:
    rel = Path(image_path)
    if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != 'images' or '..' in rel.parts:
        raise ValueError(f'Ruta insegura: {image_path}')
    absolute = (root / rel).resolve()
    if not absolute.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera de ELPD: {image_path}')
    if not absolute.is_file():
        raise FileNotFoundError(f'Imagen no encontrada: {absolute}')
    return absolute


def load_selection(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'id', 'image_path', 'group', 'type', 'ground_truth',
                    'prediction_normalized', 'original_exact_match'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Faltan campos: {required - set(reader.fieldnames or [])}')
        rows = list(reader)
    if len(rows) != 40 or {r['id'] for r in rows} != set(ROI) | set(EXCLUDED):
        raise RuntimeError('La cohorte debe contener exactamente los 40 IDs del paso 6A')
    if len({r['image_path'] for r in rows}) != 40:
        raise RuntimeError('Imagen repetida en la cohorte')
    if Counter(r['group'] for r in rows) != {'insercion':24, 'correcta_control':16}:
        raise RuntimeError('No coincide el numero de errores y controles')
    for row in rows:
        gt, pred, em = row['ground_truth'], row['prediction_normalized'], row['original_exact_match']
        if int(gt == pred) != int(em):
            raise ValueError(f'EM inconsistente: {row["id"]}')
        if row['id'][0] == 'E' and row['group'] != 'insercion':
            raise ValueError(f'Grupo inconsistente: {row["id"]}')
        if row['id'][0] == 'C' and row['group'] != 'correcta_control':
            raise ValueError(f'Grupo inconsistente: {row["id"]}')
    return rows


def load_img(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert('RGB').resize((128,32), Image.Resampling.BICUBIC)


def variants(image, box):
    from PIL import Image, ImageFilter
    x0,y0,x1,y1 = box
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 32):
        raise ValueError(f'Bbox invalido: {box}')
    baseline=image.copy()
    gray=image.copy()
    gray.paste(Image.new('RGB', (x1-x0,y1-y0), (235,235,235)), (x0,y0))
    blurred=image.copy()
    crop=image.crop(box).filter(ImageFilter.GaussianBlur(radius=3))
    blurred.paste(crop, (x0,y0))
    return {'original':baseline, 'relleno_gris':gray, 'desenfoque':blurred}


def write_csv(path: Path, records: list[dict], columns: list[str]):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def get_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(rows, originals, output, preds=None):
    from PIL import Image, ImageDraw
    f=get_font(19); fs=get_font(15)
    for page in range((len(rows)+7)//8):
        section=rows[page*8:page*8+8]
        w,h=980, 1780
        canvas=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(canvas)
        d.text((22,12),f'PARSeq ELPD V17 | PASO 6B | hoja {page+1}/{(len(rows)+7)//8}',fill=(26,41,58),font=f)
        for j,r in enumerate(section):
            y=58+j*214
            rid=r['id']
            label=f'{rid} | GT={r["ground_truth"]} | original OCR={r["prediction_normalized"]}'
            d.text((22,y),label,fill=(25,41,59),font=fs)
            reg,box=ROI[rid]
            for idx,mod in enumerate(MODES):
                v=originals[rid][mod]
                x=22+idx*318
                expanded=v.resize((300,75),Image.Resampling.NEAREST)
                canvas.paste(expanded,(x,y+29))
                if mod!='original':
                    x0,y0,x1,y1=box
                    d.rectangle((x+x0*300/128,y+29+y0*75/32,
                                 x+x1*300/128,y+29+y1*75/32), outline=(226,25,25),width=2)
                caption=mod
                if preds and (rid,mod) in preds:
                    pr=preds[(rid,mod)]
                    caption+=f': {pr["refinado"]}'
                d.text((x,y+109),caption,fill=(31,83,105),font=fs)
            d.text((22,y+143),f'{reg} | roi={box} | '+('CONTROL' if rid.startswith('C') else 'INSERCION'),
                   fill=(80,85,90),font=fs)
        output.mkdir(parents=True,exist_ok=True)
        canvas.save(output/f'hoja_{page+1:02d}.png',optimize=True)


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        row=[i]
        for j,cb in enumerate(b,1):
            row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+int(ca!=cb)))
        prev=row
    return prev[-1]


def is_subseq(gt,pred):
    it=iter(pred)
    return all(any(c==d for d in it) for c in gt)


def classify(gt,pred):
    if gt==pred: return 'correcta'
    if len(pred)>len(gt) and is_subseq(gt,pred):return 'insercion_pura'
    if len(pred)>len(gt):return 'salida_larga_con_otros_errores'
    if len(pred)==len(gt):return 'misma_longitud_incorrecta'
    return 'salida_corta'


def infer(model, core, x):
    import torch
    stages={}
    original_refines=int(core.refine_iters)
    try:
        for stage, n in [('AR',0), ('refinado',1)]:
            core.refine_iters=n
            with torch.inference_mode():
                logits=model(x)
                probs=logits.softmax(-1)
                texts, conf=model.tokenizer.decode(probs)
            prediction=re.sub('[^A-Z0-9]','',str(texts[0]).upper())
            stages[stage]={'prediction':prediction,
                           'confidence':float(conf[0].prod().item())}
    finally:
        core.refine_iters=original_refines
    return stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,default=DEFAULT_SELECTION)
    parser.add_argument('--image-root',type=Path,default=DEFAULT_IMAGES)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    selection=load_selection(args.selection)
    selected=[r for r in selection if r['id'] in ROI]
    if Counter(r['group'] for r in selected) != {'insercion':19,'correcta_control':12}:
        raise RuntimeError('Seleccion de ROI inconsistente')
    if args.out.exists():
        raise FileExistsError(f'No sobrescribir resultados existentes: {args.out}')
    print('=== PASO 6B · PARSeq · ELPD V17 · INTERVENCIONES EN COHORTE ===',flush=True)
    print('Muestra seleccionada: 40 | con ROI: 31 (19 errores; 12 controles)',flush=True)
    images={}
    for case in selected:
        rid=case['id']
        images[rid]=variants(load_img(safe_image(args.image_root,case['image_path'])),ROI[rid][1])
    args.out.mkdir(parents=True)
    (args.out/'comparativas').mkdir()
    (args.out/'imagenes_128x32').mkdir()
    plan=[{'id':r['id'],'image_path':r['image_path'],'gt':r['ground_truth'],
           'pred_original_v17':r['prediction_normalized'],
           'grupo':r['group'],'region':ROI[r['id']][0] if r['id'] in ROI else None,
           'bbox_128x32':ROI[r['id']][1] if r['id'] in ROI else None,
           'exclusion':EXCLUDED.get(r['id'],'')} for r in selection]
    (args.out/'plan_roi.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for r in selected:
        rid=r['id']
        for mode,im in images[rid].items():
            im.save(args.out/'imagenes_128x32'/f'{rid}__{mode}.png')
    make_sheet(selected,images,args.out/'comparativas')
    if args.prepare_only:
        print('RESULTADO: 31 ROI preparadas, 93 variantes; NO se ha ejecutado inferencia',flush=True)
        print('SALIDA:',args.out,flush=True)
        return

    import torch
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('Se necesita CUDA; utilizar GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME',str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model=torch.hub.load('baudm/parseq:main','parseq',pretrained=True,trust_repo=True).eval().cuda()
    core=model.model
    if not bool(core.decode_ar) or int(core.refine_iters)!=1:
        raise RuntimeError('Se esperaba decode_ar=True y refine_iters=1')
    if tuple(int(v) for v in model.hparams.img_size) != (32,128):
        raise RuntimeError('El checkpoint no usa img_size 32x128')
    to_tensor=T.Compose([T.ToTensor(),T.Normalize(0.5,0.5)])
    results=[];by_pair={}
    for idx,case in enumerate(selected,1):
        rid=case['id']; gt=case['ground_truth']
        stage_predictions={}
        for mode in MODES:
            x=to_tensor(images[rid][mode]).unsqueeze(0).cuda()
            out=infer(model,core,x)
            pred=out['refinado']['prediction']
            if mode=='original' and pred!=case['prediction_normalized']:
                raise RuntimeError(f'NO reproduce baseline original {rid}: {pred!r} vs {case["prediction_normalized"]!r}')
            item={'id':rid,'image_path':case['image_path'],'grupo':case['group'],
                  'tipo_error_original':case['type'],'GT':gt,
                  'variante':mode,'region':ROI[rid][0],
                  'bbox_128x32':str(ROI[rid][1]),
                  'AR':out['AR']['prediction'],'refinado':pred,
                  'EM_AR':int(out['AR']['prediction']==gt),'EM_refinado':int(pred==gt),
                  'CER_ediciones_AR':edit_distance(gt,out['AR']['prediction']),
                  'CER_ediciones_refinado':edit_distance(gt,pred),
                  'clase_AR':classify(gt,out['AR']['prediction']),
                  'clase_refinado':classify(gt,pred),
                  'confianza_AR':f'{out["AR"]["confidence"]:.6f}',
                  'confianza_refinado':f'{out["refinado"]["confidence"]:.6f}'}
            results.append(item)
            by_pair[(rid,mode)]={'refinado':pred}
        print(f'[{idx:02d}/31] {rid} GT={gt} '
              + ' | '.join(f'{mode}={by_pair[(rid,mode)]["refinado"]}' for mode in MODES),flush=True)
    make_sheet(selected,images,args.out/'comparativas',by_pair)
    columns=list(results[0])
    write_csv(args.out/'comparacion_intervenciones.csv',results,columns)
    byid={(r['id'],r['variante']):r for r in results}
    aggregations={}
    for stage in ('AR','refinado'):
        for group in ('insercion','correcta_control'):
            ids=[r['id'] for r in selected if r['group']==group]
            for mode in MODES:
                subset=[byid[id,mode] for id in ids]
                ncorrect=sum(r['EM_'+stage] for r in subset)
                aggregations[f'{stage}|{group}|{mode}']={'n':len(subset),'n_correctas':ncorrect,
                    'n_incorrectas':len(subset)-ncorrect,
                    'suma_ediciones':sum(r['CER_ediciones_'+stage] for r in subset)}
    notes=['Las ROI son preliminares y deben verificarse visualmente contra las matriculas.',
           'El efecto de una ROI no se atribuye directamente al encoder ni al decoder.',
           'La cohorte esta enriquecida en errores; no estima la prevalencia poblacional.',
           'Los controles son matrículas correctas, no necesariamente emparejadas por tipo de elemento grafico.',
           'Las imagenes modificadas pueden estar fuera de la distribucion natural.',
           'La clasificacion como insercion pura no implica un unico caracter insertado.']
    summary={'paso':'6B','modelo':'PARSeq oficial pretrained','dataset':'ELPD V17',
             'originales_reproducidos':len(selected),
             'cohorte_total':len(selection),'cohorte_intervenida':len(selected),
             'errores_intervenidos':19,'controles_intervenidos':12,
             'casos_sin_intervencion':len(EXCLUDED),'variantes_totales':len(results),
             'mascaras_por_objetivo':2,'image_size':[32,128],
             'sha256_csv_seleccion':sha256(args.selection),
             'resumen_por_etapa_grupo_mascara':aggregations,
             'checkpoint_modificado':False,'GT_modificado':False,
             'limitaciones':notes}
    (args.out/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('RESULTADO: 31 originales reproducidos, 93 predicciones por etapa; CSV y resumen guardados',flush=True)
    print('SALIDA:',args.out,flush=True)

if __name__=='__main__':
    main()
