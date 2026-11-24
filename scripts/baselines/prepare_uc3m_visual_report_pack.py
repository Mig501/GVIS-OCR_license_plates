#!/usr/bin/env python3
"""Exporta solo los recortes UC3M-LP necesarios para el informe comparativo.

No ejecuta inferencia ni modifica GT/predicciones. Uso: python scripts/baselines/prepare_uc3m_visual_report_pack.py
Opcional --images-root <ruta> si el dataset procesado no se detecta automaticamente.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

MODELS = ('parseq_pretrained', 'svtrv2_tiny', 'svtrv2_small', 'mdiff4str_small')
ANCHOR_GT = [
 '336*LS*','36*7KW*','694**XM','949**TS','505**WW','686**MM','2*28L*L',
 '597*LK*','R*366BD*','774**JC','718**NT','01*6H*J','7*03C*M',
 '8*14G*Y','624*GX*','*166DW*','211**NG','077*KS*','543*FZ*','56*6*CR',
 'MA*117*S','80*8*YF','5*30*FF','92*2FP*','677**XV','98*0*CY','M218*Z*',
]

def read_csv(path):
 with path.open('r', encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, columns):
 with path.open('w', encoding='utf-8', newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def candidate_image_roots(project: Path):
 home=Path.home(); ds=home/'datasets'/'ocr_matriculas'; bases=[
  ds/'processed',project/'data'/'processed',project/'datasets'/'processed',
  project/'data',project/'datasets',ds,
 ]
 roots=[]
 for base in bases:
  for k in ('UC3M_LPO','UC3M_LP','UC3M-LP','uc3m_lp','uc3m_lpo','uc3m-lp'):
   roots.extend([base/k/'images',base/k])
  if base.is_dir():
   try:
    for q in base.iterdir():
     if q.is_dir() and ('uc3m' in q.name.lower()): roots.extend([q/'images',q])
   except OSError: pass
 return list(dict.fromkeys(roots))

def locate_images(root, image_names):
 roots=[root] if root else []
 roots+=candidate_image_roots(PROJECT)
 roots=list(dict.fromkeys(p.resolve() for p in roots if p and p.is_dir()))
 matches={}
 for p in roots:
  found=sum((p/name).is_file() or (p/Path(name).name).is_file() for name in image_names[:20])
  if found:
   print(f'[INFO] Directorio de recortes: {p} ({found}/20 sondas)')
   for name in image_names:
    for z in (p/name,p/Path(name).name):
     if z.is_file():matches[name]=z;break
   if len(matches)==len(image_names):break
 return matches, roots

def intv(x):
 try:return int(x)
 except (TypeError,ValueError):return 0

def floatv(x):
 try:return float(x)
 except (TypeError,ValueError):return 0.

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--images-root',type=Path,default=None)
 ap.add_argument('--max-images',type=int,default=110)
 args=ap.parse_args()
 preds={}; summaries={}
 for key in MODELS:
  basedir=PROJECT/'results'/'baselines'/'UC3M-LP'/key
  pf=basedir/'predictions.csv'; jf=basedir/'summary.json'
  if not pf.is_file() or not jf.is_file(): raise FileNotFoundError(f'Falta {pf} o {jf}')
  preds[key]=read_csv(pf)
  summaries[key]=json.loads(jf.read_text(encoding='utf-8'))
  if len(preds[key])!=491: raise ValueError(f'{key}: {len(preds[key])} != 491')
 ref=preds[MODELS[0]]
 ids=[r['image_path'] for r in ref]
 if len(set(ids))!=491:raise ValueError('IDs duplicados en referencia')
 for key in MODELS[1:]:
  if {r['image_path'] for r in preds[key]}!=set(ids):raise ValueError('Imagenes distintas entre modelos')
  if any(r['ground_truth_masked']!=ref[i]['ground_truth_masked'] for i,r in enumerate(preds[key])):
   # Permitir orden distinto mediante verificacion por clave
   base={r['image_path']:r['ground_truth_masked'] for r in ref}
   if any(base[r['image_path']]!=r['ground_truth_masked'] for r in preds[key]):raise ValueError('GT no coincide entre modelos')
 bymodel={k:{r['image_path']:r for r in preds[k]} for k in MODELS}
 out=PROJECT/'results'/'audit'/'UC3M-LP'/'informe_visual_v1';out.mkdir(parents=True,exist_ok=True)
 rows=[]; scores=defaultdict(int); reasons=defaultdict(list)
 gtindex=defaultdict(list)
 for r in ref:gtindex[r['ground_truth_masked']].append(r['image_path'])
 for gt in ANCHOR_GT:
  for path in gtindex.get(gt,[])[:2]:scores[path]+=120;reasons[path].append('revisado_previamente:'+gt)
 # Categorias objetivas; priorizar cobertura por modelo, sin atribuir causas visuales
 for model in MODELS:
  rs=preds[model]
  buckets={
   'vacia':[r for r in rs if not r['prediction_normalized']],
   'muy_corta':[r for r in rs if 0<intv(r['pred_length'])<=1],
   'excede_longitud_gt':[r for r in rs if intv(r['pred_length'])>intv(r['gt_length'])],
   'diferencia_posicion_conocida':[r for r in rs if intv(r['length_match'])==1 and intv(r['known_errors_if_aligned'])>0],
   'confianza_alta_con_mismatch_conocido':[r for r in rs if intv(r['length_match'])==1 and intv(r['known_errors_if_aligned'])>0 and floatv(r['confidence'])>=0.9],
   'prediccion_visible_igual':[r for r in rs if r['prediction_normalized']==r['ground_truth_visible'] and intv(r['partial_exact_match'])==0],
   'prediccion_muy_corta_vs_conocido':[r for r in rs if intv(r['pred_length'])>1 and intv(r['pred_length'])<intv(r['known_characters'])-1],
  }
  for c,v in buckets.items():
   if c=='excede_longitud_gt':v.sort(key=lambda x:(intv(x['pred_length'])-intv(x['gt_length']),floatv(x['confidence'])),reverse=True)
   elif c=='prediccion_visible_igual':v.sort(key=lambda x:floatv(x['confidence']),reverse=True)
   elif c=='diferencia_posicion_conocida':v.sort(key=lambda x:(intv(x['known_errors_if_aligned']),floatv(x['confidence'])),reverse=True)
   else:v.sort(key=lambda x:floatv(x['confidence']),reverse=True)
   limit={'vacia':3,'muy_corta':5,'excede_longitud_gt':12,'diferencia_posicion_conocida':8,'confianza_alta_con_mismatch_conocido':6,'prediccion_visible_igual':4,'prediccion_muy_corta_vs_conocido':6}[c]
   for rank,r in enumerate(v[:limit]):
    path=r['image_path'];scores[path]+=max(4,36-rank*2);reasons[path].append(f'{model}:{c}')
 selected=sorted(scores,key=lambda x:(-scores[x],x))[:args.max_images]
 images, search_roots=locate_images(args.images_root,selected)
 if not images:
  print('[ERROR] No se encontraron los recortes procesados de UC3M-LP.')
  print('Directorios buscados:',*[str(p) for p in search_roots],sep='\n  ')
  print('Prueba: python scripts/baselines/prepare_uc3m_visual_report_pack.py --images-root <carpeta que contiene test__00000__lp00.png>')
  raise SystemExit(2)
 originals_dir=out/'imagenes_originales';originals_dir.mkdir(exist_ok=True)
 for path in selected:
  r=ref[ids.index(path)]
  record={'image_path':path,'ground_truth_masked':r['ground_truth_masked'],'ground_truth_visible':r['ground_truth_visible'],'gt_length':r['gt_length'],'masked_characters':r['masked_characters'],'seleccion_motivos':' | '.join(reasons[path]), 'imagen_incluida':'SI' if path in images else 'NO'}
  for key in MODELS:
   v=bymodel[key][path];record[key+'_prediccion']=v['prediction_normalized'];record[key+'_confianza']=v['confidence'];record[key+'_partial_em']=v['partial_exact_match'];record[key+'_masked_distance']=v['masked_edit_distance']
  rows.append(record)
  if path in images:shutil.copy2(images[path],originals_dir/Path(path).name)
 fields=list(rows[0]); write_csv(out/'seleccion_para_informe.csv',rows,fields)
 stats=[]
 for key in MODELS:
  r=preds[key]; j=summaries[key]
  visible=sum(x['prediction_normalized']==x['ground_truth_visible'] for x in r)
  stat={'modelo':key,'imagenes':len(r),'em_parcial_estricto_pct':round(j['partial_EM_percent'],2),'error_mascara_proxy_pct':round(j['masked_error_rate_proxy_percent'],2),'coincidencia_texto_visible':visible,'coincidencia_texto_visible_pct':round(100*visible/len(r),2),'discrepancias_texto_visible':len(r)-visible,'salidas_vacias':sum(not x['prediction_normalized'] for x in r),'predicciones_mas_largas_que_gt_total':sum(intv(x['pred_length'])>intv(x['gt_length']) for x in r),'longitud_igual_gt':sum(intv(x['length_match'])==1 for x in r),'parametros':j.get('num_parameters','')}
  stats.append(stat)
 write_csv(out/'resumen_modelos_uc3m.csv',stats,list(stats[0]))
 (out/'LEEME.txt').write_text('UC3M-LP: imagenes seleccionadas para el informe (no todos los recortes).\nLas etiquetas * ocultan caracteres: NO son GT completos.\nLas discrepancias con ground_truth_visible no equivalen a errores OCR confirmados.\nLas categorias son reglas sobre longitud y posiciones conocidas; las causas visuales deben confirmarse.\n',encoding='utf-8')
 zippath=out/'UC3M_LP_Imagenes_Para_Informe.zip'
 with ZipFile(zippath,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for file in [out/'seleccion_para_informe.csv',out/'resumen_modelos_uc3m.csv',out/'LEEME.txt']:
   z.write(file,file.name)
  for f in originals_dir.iterdir():z.write(f,'imagenes_originales/'+f.name)
 print(f'[OK] Predicciones: {len(ref)} por modelo; los 4 CSV usan las mismas imagenes y etiquetas.')
 for s in stats:print(f"[OK] {s['modelo']}: EM parcial {s['em_parcial_estricto_pct']}%, error con mascara {s['error_mascara_proxy_pct']}%, texto visible {s['coincidencia_texto_visible_pct']}%")
 print(f'[OK] Imagenes seleccionadas: {len(selected)}; encontradas: {len(images)}; ausentes: {len(selected)-len(images)}')
 print('[OK] ZIP:',zippath)
 if len(images)<len(selected):print('[AVISO] Faltan recortes. Comprobar la ruta del dataset de UC3M-LP procesado.')

if __name__=='__main__':
 PROJECT=Path(__file__).resolve().parents[2]
 main()
