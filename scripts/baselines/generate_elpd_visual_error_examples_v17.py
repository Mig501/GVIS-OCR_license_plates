#!/usr/bin/env python3
"""Crea catalogos y ejemplos visuales de errores OCR con el GT ELPD V17 validado.

Solo lectura: usa 4 CSV de predicciones guardadas (733 imagenes cada uno),
no carga modelos ni reescribe el dataset. Exporta TODOS los errores a CSV y
hojas de contacto y selecciona ejemplos representativos por patron textual.
Las causas visuales (escudos, desenfoque, perspectiva) NO se infieren de texto.

Ubicacion: scripts/baselines/generate_elpd_visual_error_examples_v17.py
Ejecucion: CPU, 2 cores, 4GB, 00:20:00, sin argumentos.
Dependencia: Pillow (ya usada por el generador original).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASET = Path('/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO')
MODELS = {
    'parseq': 'PARSeq',
    'svtrv2_tiny': 'SVTRv2-Tiny',
    'svtrv2_small': 'SVTRv2-Small',
    'mdiff4str_small': 'MDiff4STR-Small',
}
N = 733

# Categorias NO excluyentes y basadas solo en diferencias textuales.
CATEGORIES = {
    'truncamiento_severo': 'Truncamiento severo',
    'omision': 'Omision de caracteres',
    'insercion_prefijo': 'Insercion al comienzo',
    'insercion_sufijo': 'Insercion al final',
    'insercion_interna': 'Insercion dentro de la secuencia',
    'repeticion_aparente': 'Repeticion aparente de caracteres/secuencia',
    'confusion_similar': 'Sustitucion de caracteres parecidos',
    'otras_sustituciones': 'Otras sustituciones',
    'errores_mixtos': 'Combinacion de varios tipos de edicion',
    'alta_confianza': 'Errores con confianza numerica >= 0.85',
    'error_extremo': 'Distancia de edicion >= 3',
    'inspeccionar_distintivos': 'Inserciones en extremos: posible simbolo (no verificado)',
}
CONFUSIONS = [set('I1'), set('O0'), set('B8'), set('S5'), set('Z2'), set('G6')]


def font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_csv(path: Path, must: set[str]) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        absent = must - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f'Columnas ausentes en {path}: {sorted(absent)}')
        return list(reader)


def save_csv(path: Path, data: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def aligned_edits(gt: str, pred: str) -> list[tuple[str, str, str]]:
    """Operacion (S,I,D) sobre alineacion Levenshtein minima no necesariamente unica."""
    n, m = len(gt), len(pred)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1]+(gt[i-1] != pred[j-1]))
    ops = []
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (gt[i-1] != pred[j-1]):
            if gt[i-1] != pred[j-1]:
                ops.append(('S', gt[i-1], pred[j-1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i-1][j]+1:
            ops.append(('D', gt[i-1], ''))
            i -= 1
        else:
            ops.append(('I', '', pred[j-1]))
            j -= 1
    return list(reversed(ops))


def repeated(gt: str, pred: str) -> bool:
    if len(pred) <= len(gt):
        return False
    # Secuencia repetida de forma contigua; el patron es solo candidato.
    for size in range(2, min(8, len(pred)//2)+1):
        for start in range(len(pred)-2*size+1):
            if pred[start:start+size] == pred[start+size:start+2*size]:
                return True
    if gt and pred.count(gt) >= 2:
        return True
    # Rachas de un mismo caracter no presentes en GT.
    for char in set(pred):
        if char*3 in pred and char*3 not in gt:
            return True
    return False


def classify(gt: str, pred: str, confidence: float | None, ops: list[tuple[str,str,str]]) -> set[str]:
    kinds = Counter(op for op, _, _ in ops)
    tags: set[str] = set()
    if len(pred) <= 3 and len(gt)-len(pred) >= 3:
        tags.add('truncamiento_severo')
    if kinds['D']:
        tags.add('omision')
    if kinds['I']:
        if len(pred) > len(gt) and pred.endswith(gt):
            tags.add('insercion_prefijo')
        elif len(pred) > len(gt) and pred.startswith(gt):
            tags.add('insercion_sufijo')
        else:
            tags.add('insercion_interna')
        if repeated(gt, pred):
            tags.add('repeticion_aparente')
        if len(pred) > len(gt) and (pred.startswith(gt) or pred.endswith(gt)):
            tags.add('inspeccionar_distintivos')
    for op, g, p in ops:
        if op == 'S':
            if any(g in pair and p in pair for pair in CONFUSIONS):
                tags.add('confusion_similar')
            else:
                tags.add('otras_sustituciones')
    if sum(bool(kinds[k]) for k in ('S','I','D')) > 1:
        tags.add('errores_mixtos')
    if confidence is not None and confidence >= 0.85:
        tags.add('alta_confianza')
    if len(ops) >= 3:
        tags.add('error_extremo')
    return tags


def path_ok(rel: str) -> bool:
    p = Path(rel)
    return not p.is_absolute() and '..' not in p.parts and len(p.parts) == 2 and p.parts[0] == 'images'


def load_model(project: Path, images: Path, slug: str) -> tuple[list[dict], dict]:
    base = project/'results/audit/ELPD/gt_corrections_v17'/slug
    infile = base/'predictions_verified_v17.csv'
    summaryfile = base/'summary_gt_corrected_v17.json'
    required = {'image_path','ground_truth','prediction_normalized','exact_match',
                'edit_distance','gt_length','gt_validation_status_v17',
                'include_in_verified_eval_v17'}
    rows = read_csv(infile, required)
    if len(rows) != N:
        raise ValueError(f'{slug}: hay {len(rows)} filas en vez de {N}')
    summary = json.loads(summaryfile.read_text(encoding='utf-8'))
    if summary.get('version') != 'v17' or summary.get('gt_corrections') != 39 or summary.get('no_verificables') != 2:
        raise ValueError(f'{slug}: el resumen no corresponde a GT V17')
    seen = set()
    for row in rows:
        rel = row['image_path']
        if not path_ok(rel) or rel in seen:
            raise ValueError(f'{slug}: ruta repetida o insegura: {rel}')
        seen.add(rel)
        original = images/rel
        if not original.is_file():
            raise FileNotFoundError(f'Imagen ausente {original}; revisar --image-root')
        gt, pred = row['ground_truth'], row['prediction_normalized']
        if not gt or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in gt+pred):
            raise ValueError(f'{slug}: GT/prediccion no normalizados: {rel}')
        if row['include_in_verified_eval_v17'].lower() != 'si':
            raise ValueError(f'{slug}: caso excluido incluido en 733: {rel}')
        if int(row['exact_match']) != int(gt == pred):
            raise ValueError(f'{slug}: EM inconsistente en {rel}')
        ops = aligned_edits(gt, pred)
        if len(ops) != int(row['edit_distance']) or int(row['gt_length']) != len(gt):
            raise ValueError(f'{slug}: ediciones/longitud no coinciden en {rel}')
        try:
            confidence = float(row.get('confidence', ''))
        except (ValueError, TypeError):
            confidence = None
        row['confidence_num'] = confidence
        row['ops'] = ops
        row['tags'] = classify(gt, pred, confidence, ops) if pred != gt else set()
    exact = sum(int(row['exact_match']) for row in rows)
    edits = sum(int(row['edit_distance']) for row in rows)
    chars = sum(int(row['gt_length']) for row in rows)
    expected = summary['verified_metrics_v17_733']
    if (exact, edits, chars) != (expected['exact_matches'], expected['levenshtein_errors'], expected['gt_characters']):
        raise ValueError(f'{slug}: metricas incompatibles con summary V17')
    return rows, summary


def clip(s: str, max_chars: int = 52) -> str:
    return s if len(s) <= max_chars else s[:max_chars-3]+'...'


def put_text(draw: ImageDraw.ImageDraw, xy, value: str, face, color: str,
             max_width: int) -> None:
    value = str(value)
    while len(value) > 3 and draw.textbbox((0,0), value, font=face)[2] > max_width:
        value = value[:-4]+'...'
    draw.text(xy, value, fill=color, font=face)


def tile(row: dict, img_root: Path, model: str, width=580, height=270) -> Image.Image:
    image = Image.new('RGB', (width, height), '#FFFFFF')
    d = ImageDraw.Draw(image)
    d.rectangle((0,0,width-1,height-1),outline='#C9D3E0',width=2)
    d.rectangle((0,0,width-1,34),fill='#1C304A')
    put_text(d, (12,7), model, font(18), '#FFFFFF',width-24)
    visual_area=(width-30, max(85, height-145))
    with Image.open(img_root/row['image_path']) as src:
        src = ImageOps.exif_transpose(src).convert('RGB')
        src = ImageOps.contain(src, visual_area, method=Image.Resampling.LANCZOS)
        x=(width-src.width)//2
        y=39+(visual_area[1]-src.height)//2
        image.paste(src, (x,y))
    ypos=45+visual_area[1]
    put_text(d, (12,ypos), 'GT: '+row['ground_truth'],font(20),'#0C5B39',width-24)
    put_text(d, (12,ypos+27), 'OCR: '+(row['prediction_normalized'] or '(VACIO)'),font(20),'#A22926',width-24)
    put_text(d, (12,ypos+55),clip(row['image_path'],55),font(13),'#566579',width-24)
    put_text(d, (12,ypos+75),f"Distancia: {row['edit_distance']} | Confianza: {row.get('confidence','')}",font(13),'#566579',width-24)
    return image


def compose(rows: list[dict], img_root: Path, title: str, model: str,
            out: Path, cols=2, per_sheet=12) -> list[Path]:
    if not rows:
        return []
    saved=[]
    w,h=580,270
    for idx in range(0,len(rows),per_sheet):
        batch=rows[idx:idx+per_sheet]
        grid_rows=math.ceil(len(batch)/cols)
        page=Image.new('RGB',(w*cols,52+grid_rows*h),'#EEF2F7')
        d=ImageDraw.Draw(page)
        put_text(d,(15,9),title+f' | {idx+1}-{idx+len(batch)}',font(22),'#10233B',w*cols-30)
        for j,row in enumerate(batch):
            page.paste(tile(row,img_root,model,w,h),((j%cols)*w,52+(j//cols)*h))
        target = out if len(rows)<=per_sheet else out.with_name(f'{out.stem}_{idx//per_sheet+1:03d}{out.suffix}')
        target.parent.mkdir(parents=True,exist_ok=True)
        page.save(target,quality=88,optimize=True)
        saved.append(target)
    return saved


def rank(row: dict, cat: str) -> tuple:
    # Favorecer lecturas cortas, un solo tipo de error y nitidez estimada (sin
    # afirmar calidad visual). Confianza no es comparable entre arquitecturas.
    dist = int(row['edit_distance'])
    confidence=row['confidence_num'] or 0
    if cat in ('truncamiento_severo','error_extremo','repeticion_aparente'):
        return (-dist,-confidence,row['image_path'])
    if cat == 'alta_confianza':
        return (-confidence,dist,row['image_path'])
    if cat == 'confusion_similar':
        return (dist,-confidence,row['image_path'])
    return (dist,-confidence,row['image_path'])


def export_model(slug: str, rows: list[dict], img_root: Path,
                 out_root: Path, per_category: int, sheet_size: int) -> tuple[dict,list[dict]]:
    errors=[r for r in rows if r['exact_match']=='0']
    model_root=out_root/slug
    total_count=Counter()
    catalog=[]
    grouped=defaultdict(list)
    for r in errors:
        kinds=Counter(x[0] for x in r['ops'])
        for tag in r['tags']:
            grouped[tag].append(r)
            total_count[tag] += 1
        catalog.append({
            'modelo':MODELS[slug], 'image_path':r['image_path'],
            'scenario':r.get('scenario',''), 'ground_truth_v17':r['ground_truth'],
            'prediccion':r['prediction_normalized'], 'confianza':r.get('confidence',''),
            'edit_distance':r['edit_distance'],'sustituciones':kinds['S'],
            'inserciones':kinds['I'],'omisiones':kinds['D'],
            'categorias_textuales':'|'.join(sorted(r['tags'])),
            'causa_visual_verificada':'NO','revision_cualitativa':'',
        })
    fields=['modelo','image_path','scenario','ground_truth_v17','prediccion','confianza',
            'edit_distance','sustituciones','inserciones','omisiones',
            'categorias_textuales','causa_visual_verificada','revision_cualitativa']
    save_csv(model_root/'todos_los_errores_v17.csv',catalog,fields)
    compose(errors,img_root,f'{MODELS[slug]} - TODOS los errores GT V17',MODELS[slug],
            model_root/'todos_errores_hojas'/'errores.jpg',per_sheet=sheet_size)

    selected=[]
    for cat,desc in CATEGORIES.items():
        cases=sorted(grouped[cat],key=lambda r:rank(r,cat))[:per_category]
        if not cases:
            continue
        compose(cases,img_root,f'{MODELS[slug]} - {desc}',MODELS[slug],
                model_root/'categorias'/f'{cat}.jpg',per_sheet=max(12,per_category))
        for r in cases:
            key=hashlib.sha1(r['image_path'].encode('utf-8')).hexdigest()[:9]
            prefix=Path(r['image_path']).stem[:60]
            unique=f'{prefix}_{key}'
            original=model_root/'originales_seleccionados'/(unique+Path(r['image_path']).suffix.lower())
            labeled=model_root/'ejemplos_etiquetados'/f'{unique}.png'
            original.parent.mkdir(parents=True,exist_ok=True)
            labeled.parent.mkdir(parents=True,exist_ok=True)
            if not original.exists():
                shutil.copyfile(img_root/r['image_path'],original)
                tile(r,img_root,MODELS[slug],width=1000,height=360).save(labeled)
            selected.append({
                'modelo':MODELS[slug],'categoria':cat,'descripcion':desc,
                'image_path':r['image_path'],'ground_truth_v17':r['ground_truth'],
                'prediccion':r['prediction_normalized'],'edit_distance':r['edit_distance'],
                'imagen_original':str(original.relative_to(out_root)),
                'imagen_etiquetada':str(labeled.relative_to(out_root)),
                'confirmacion_visual':'PENDIENTE','comentario':'',
            })
    return {'modelo':MODELS[slug], 'imagenes':len(rows), 'errores':len(errors),
            'aciertos':len(rows)-len(errors),
            'categorias_no_excluyentes':dict(total_count),
            'seleccionados':len(selected)}, selected


def compare_models(used: dict[str,list[dict]], root: Path, image_root: Path) -> list[dict]:
    by_model={k:{r['image_path']:r for r in rows} for k,rows in used.items()}
    shared=[]
    for ip in sorted(next(iter(by_model.values()))):
        ref=next(iter(by_model.values()))[ip]
        gt=ref['ground_truth']
        if any(by_model[k][ip]['ground_truth'] != gt for k in by_model):
            raise ValueError(f'GT inconsistente entre modelos en {ip}')
        incorrect=[k for k in MODELS if by_model[k][ip]['exact_match']=='0']
        if len(incorrect)!=len(MODELS):
            continue
        entry={'image_path':ip,'ground_truth_v17':gt,'numero_modelos_erroneos':4}
        for k in MODELS:
            entry[k+'_prediction']=by_model[k][ip]['prediction_normalized']
            entry[k+'_distancia']=by_model[k][ip]['edit_distance']
        shared.append(entry)
    fields=['image_path','ground_truth_v17','numero_modelos_erroneos']+[f'{k}_{suffix}' for k in MODELS for suffix in ('prediction','distancia')]
    save_csv(root/'comparacion'/'errores_compartidos_4_modelos_v17.csv',shared,fields)
    # Hojas comparativas reales con una imagen y las cuatro lecturas.
    for start in range(0,min(len(shared),24),4):
        pack=shared[start:start+4]
        cards=[]
        for item in pack:
            ip=item['image_path']; gt=item['ground_truth_v17']
            card=Image.new('RGB',(1160,330),'#FFFFFF')
            d=ImageDraw.Draw(card)
            d.rectangle((0,0,1159,329),outline='#BCC8D8',width=2)
            d.rectangle((0,0,1159,34),fill='#1C304A')
            put_text(d,(14,7),f'{ip} | GT: {gt}',font(18),'#FFFFFF',1125)
            with Image.open(image_root/ip) as src:
                src=ImageOps.exif_transpose(src).convert('RGB')
                src=ImageOps.contain(src,(535,276),Image.Resampling.LANCZOS)
                card.paste(src,((550-src.width)//2,41+(276-src.height)//2))
            y=55
            for slug,name in MODELS.items():
                pred=item[slug+'_prediction'] or '(VACIO)'
                put_text(d,(575,y),f'{name}: {pred}',font(19),'#9A2525',575)
                y+=58
            cards.append(card)
        page=Image.new('RGB',(1160,45+len(cards)*330),'#EEF2F7')
        ImageDraw.Draw(page).text((14,8),'ELPD V17 | Matrículas con error en los 4 modelos',font=font(22),fill='#1C304A')
        for idx,card in enumerate(cards):
            page.paste(card,(0,45+idx*330))
        fn=root/'comparacion'/f'comparativa_{start//4+1:02d}.jpg'
        fn.parent.mkdir(parents=True,exist_ok=True)
        page.save(fn,quality=90,optimize=True)
    return shared


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=PROJECT)
    parser.add_argument('--image-root',type=Path,default=DATASET)
    parser.add_argument('--output-root',type=Path,default=None)
    parser.add_argument('--examples-per-category',type=int,default=5)
    parser.add_argument('--images-per-sheet',type=int,default=12)
    args=parser.parse_args()
    if args.examples_per_category < 1 or args.images_per_sheet < 1:
        parser.error('Los limites deben ser positivos')
    out=args.output_root or args.project_root/'results/audit/ELPD/visual_errors_v17'
    print('=== ELPD V17: CATALOGO Y EJEMPLOS VISUALES DE ERRORES ===',flush=True)
    print('Origen:',args.project_root/'results/audit/ELPD/gt_corrections_v17',flush=True)
    loaded={}
    summaries=[]
    selection=[]
    for slug in MODELS:
        rows,_ = load_model(args.project_root,args.image_root,slug)
        loaded[slug]=rows
        s,sel=export_model(slug,rows,args.image_root,out,args.examples_per_category,args.images_per_sheet)
        summaries.append(s)
        selection.extend(sel)
        print(f"[OK] {s['modelo']}: 733 imagenes, {s['errores']} errores, "
              f"{s['aciertos']} aciertos, {s['seleccionados']} entradas de ejemplo",flush=True)
    shared=compare_models(loaded,out,args.image_root)
    select_fields=['modelo','categoria','descripcion','image_path','ground_truth_v17',
                   'prediccion','edit_distance','imagen_original','imagen_etiquetada',
                   'confirmacion_visual','comentario']
    save_csv(out/'seleccion_ejemplos_v17.csv',selection,select_fields)
    counts=[]
    for s in summaries:
        for cat in CATEGORIES:
            counts.append({'modelo':s['modelo'],'categoria':cat,'descripcion':CATEGORIES[cat],
                           'numero_errores':s['categorias_no_excluyentes'].get(cat,0),
                           'errores_totales_modelo':s['errores']})
    save_csv(out/'resumen_categorias_v17.csv',counts,
             ['modelo','categoria','descripcion','numero_errores','errores_totales_modelo'])
    info={'dataset':'ELPD','version_gt':'v17','imagenes_validadas':733,
          'correcciones_gt':39,'exclusiones_gt_no_verificables':2,
          'modelos':summaries,'errores_compartidos_los_4':len(shared),
          'nota':'Categorias inferidas del texto (no mutuamente excluyentes). No implican causa visual comprobada.',
          'protocolo':'Dataset completo en splits originales train/val/test; no es test de generalizacion externa.'}
    (out/'resumen_general_v17.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'LEEME.txt').write_text(
        'ELPD V17 - EJEMPLOS VISUALES DE ERRORES (733 GT verificados)\n'
        'No se han ejecutado modelos ni modificado los recortes originales.\n\n'
        'Carpeta de cada modelo:\n'
        '  todos_los_errores_v17.csv: cada error por separado con taxonomia textual.\n'
        '  todos_errores_hojas/: mosaicos con TODOS los errores del modelo.\n'
        '  categorias/: mosaicos representativos por tipo de error.\n'
        '  originales_seleccionados/: copia binaria de imagenes originales seleccionadas.\n'
        '  ejemplos_etiquetados/: originales con GT, OCR y distancia superpuestos.\n'
        'Raiz: seleccion_ejemplos_v17.csv y resumen_categorias_v17.csv.\n'
        'Comparacion: errores_compartidos_4_modelos_v17.csv + comparativas.\n\n'
        'IMPORTANTE: Truncamientos, inserciones, sustituciones, etc. son diagnósticos\n'
        'de diferencias textuales. Confirmar visualmente escudos, bandas y causas\n'
        'antes de afirmar un mecanismo causal en el informe Word.\n',encoding='utf-8')
    # ZIP portable para Word, sin hojas de TODOS los errores (guardadas en el nodo).
    bundle=out/'elpd_ejemplos_word_v17.zip'
    with ZipFile(bundle,'w',ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(out.rglob('*')):
            if not f.is_file() or f == bundle or 'todos_errores_hojas' in f.parts:
                continue
            z.write(f,arcname=str(f.relative_to(out)))
    print(f'[OK] Compartidos en los 4 modelos: {len(shared)}',flush=True)
    print('[OK] Catalogo: ',out/'seleccion_ejemplos_v17.csv',flush=True)
    print('[OK] Paquete Word:',bundle,flush=True)
    print('RESULTADO: EJEMPLOS VISUALES V17 GENERADOS; REVISION VISUAL DE CAUSAS PENDIENTE',flush=True)

if __name__=='__main__':
    main()
