#!/usr/bin/env python3
"""Genera lotes auditables de las etiquetas ELPD pendientes, priorizando casos críticos.

Colocar en scripts/baselines/ junto a prepare_elpd_gt_corrections.py.
Registro requerido: configs/elpd_gt_review_registry_v2.csv (55 entradas: 50 cerradas y 5 ambiguas).
Cola requerida: results/audit/ELPD/gt_correction_review/review_queue.csv (655 filas).
Salida: results/audit/ELPD/gt_correction_review/batches_v2/ con ZIPs de 48 imágenes,
        índice CSV y resumen JSON. No modifica imágenes, etiquetas ni predicciones.

La coincidencia entre modelos SOLO PRIORIZA revisión; jamás corrige GT por consenso.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import math
from zipfile import ZipFile, ZIP_DEFLATED

from prepare_elpd_gt_corrections import render_sheet, csv_bytes

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
IMAGE_ROOT = Path('/home/miguel.sanchez/datasets/ocr_matriculas/processed/ELPD_LPO')
MANIFEST_REL = Path('results/audit/ELPD/eval_manifests/external_eval_manifest.csv')
QUEUE_REL = Path('results/audit/ELPD/gt_correction_review/review_queue.csv')
REGISTRY_REL = Path('configs/elpd_gt_review_registry_v2.csv')
OUT_REL = Path('results/audit/ELPD/gt_correction_review/batches_v2')
EXPECTED_SHA = '5854abd860c89a17e35ef225627b17725c60d23088197f280e34eb5272097f13'
MODEL_IDS = ('parseq','svtrv2_tiny','svtrv2_small','mdiff4str_small')
IMMEDIATE = ('images/train__20CEY44.jpg', 'images/train__6176YM27.jpg',
             'images/train__FQ032EC.jpg','images/train__BQ360YQ.jpg',
             'images/train__G0CIAPA.png')


def read(path: Path, required: set[str]) -> list[dict[str,str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Archivo necesario no encontrado: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        missing = required - set(rd.fieldnames or [])
        if missing:
            raise ValueError(f'Columnas ausentes en {path}: {sorted(missing)}')
        return list(rd)


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def dist(a: str, b: str) -> int:
    before = list(range(len(b) + 1))
    for i, c in enumerate(a, 1):
        now = [i]
        for j,d in enumerate(b,1):
            now.append(min(now[-1]+1, before[j]+1, before[j-1]+(c!=d)))
        before = now
    return before[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=PROJECT)
    ap.add_argument('--image-root', type=Path, default=IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--metadata-only', action='store_true', help='Valida y genera solo la planificación CSV, sin ZIPs ni imágenes')
    args = ap.parse_args()
    if not 1 <= args.batch_size <= 96:
        ap.error('batch-size debe estar entre 1 y 96')
    project = args.project_root
    manifest = project / MANIFEST_REL
    if sha256(manifest) != EXPECTED_SHA:
        raise ValueError('El manifiesto original no coincide con el SHA-256 validado.')
    dataset = read(manifest, {'image_path', 'text_normalized', 'scenario'})
    if len(dataset)!=735 or len(set(r['image_path'] for r in dataset))!=735:
        raise ValueError('Se esperan exactamente 735 registros sin duplicados.')
    indexed={r['image_path']:r for r in dataset}
    queue = read(project/QUEUE_REL, {'image_path','gt_before_review','gt_source','review_order',
                                    'priority_num','suggested_gt_NOT_VERIFIED','agreeing_models',
                                    'models_matching_current_gt',*[f'pred_{m}' for m in MODEL_IDS]})
    if len(queue)!=655 or len(set(r['image_path'] for r in queue))!=655:
        raise ValueError('La cola debe contener exactamente 655 imágenes únicas.')
    for r in queue:
        path = Path(r['image_path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0]!='images' or len(path.parts)!=2:
            raise ValueError(f'Ruta de imagen insegura: {r["image_path"]}')
        if r['image_path'] not in indexed:
            raise ValueError('Cola con ruta ajena al manifiesto.')
        if r['gt_source']=='filename_candidate' and indexed[r['image_path']]['text_normalized']!=r['gt_before_review']:
            raise ValueError(f'GT original distinto de cola: {r["image_path"]}')
    registry = read(project/REGISTRY_REL,{'image_path','estado','gt_inicial','gt_corregido','observacion'})
    if len(registry)!=55 or len({r['image_path'] for r in registry})!=55:
        raise ValueError('Se esperan 55 decisiones previas sin duplicados (registro V2).')
    reg={r['image_path']:r for r in registry}
    qidx={r['image_path']:r for r in queue}
    for ip,entry in reg.items():
        if ip not in qidx or entry['gt_inicial']!=qidx[ip]['gt_before_review']:
            raise ValueError(f'Registro fuera de cola o con GT inicial distinto: {ip}')
        if entry['estado'] not in {'corregido','confirmado','ambiguo'}:
            raise ValueError(f'Estado de revisión inválido: {entry}')
        if entry['estado']=='corregido' and not entry['gt_corregido']:
            raise ValueError(f'Corrección sin GT nuevo: {ip}')
    counts=Counter(r['estado'] for r in registry)
    if counts != Counter({'corregido':18,'confirmado':32,'ambiguo':5}):
        raise ValueError(f'Registro V2 inesperado: {counts}')
    closed = {r['image_path'] for r in registry if r['estado'] in ('corregido','confirmado')}
    pending=[]
    for r in queue:
        ip=r['image_path']
        if ip in closed:
            continue
        gt=r['gt_before_review']
        preds=[r[f'pred_{m}'] for m in MODEL_IDS]
        dists=[dist(gt,p) for p in preds]
        mind=min(dists)
        parseq_dist=dists[0]
        consensus=int(r['agreeing_models'])
        # La gran discrepancia de TODOS los modelos prima sobre coincidencia entre ellos.
        # Una desviación aislada de un modelo no se toma como indicio de GT erróneo.
        score = 10*min(mind,7)+6*consensus+2*min(parseq_dist,7)
        if ip in IMMEDIATE:
            score+=150
        if ip in reg and reg[ip]['estado']=='ambiguo':
            score+=15
        reason=[]
        if ip in IMMEDIATE:reason.append('Revisión crítica ya identificada; lectura pendiente')
        if mind>=3:reason.append('Los cuatro OCR difieren del GT en al menos 3 ediciones')
        if consensus>=3:reason.append(f'{consensus}/4 OCR coinciden en otra lectura (no prueba de GT)')
        if ip in reg and reg[ip]['estado']=='ambiguo':reason.append('Revisión anterior ambigua: segunda inspección')
        if not reason:reason.append('Auditoría completa de la etiqueta candidata')
        pending.append({**r,'priority_score_v2':score,'minimum_edits_all_models':mind,
                        'parseq_edit_distance':parseq_dist,'priority_reason_v2':' | '.join(reason),
                        'review_state_v2':'pendiente' if ip not in reg else 'revisar_ambiguedad'})
    pending.sort(key=lambda r:(-r['priority_score_v2'],int(r['priority_num']),r['image_path']))
    if len(pending)!=605:
        raise ValueError(f'Pendientes inesperados: {len(pending)} != 605')
    if not args.metadata_only:
        missing = [r['image_path'] for r in pending if not (args.image_root / r['image_path']).is_file()]
        if missing:
            raise FileNotFoundError(f'Faltan {len(missing)} imágenes originales. Primera: {missing[0]}')
    out = args.out or project/OUT_REL
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Ya existe material en {out}. No se sobrescribe.')
    out.mkdir(parents=True,exist_ok=True)
    header=list(pending[0]);
    index=[]
    total=math.ceil(len(pending)/args.batch_size)
    for start in range(0,len(pending),args.batch_size):
        number=start//args.batch_size+1
        batch = [dict(r) for r in pending[start:start+args.batch_size]]
        package=f'gt_review_lote_{number:02d}.zip'
        for j,r in enumerate(batch):
            r['review_order']=start+j+1
            r['contact_sheet']=f'contact_sheets/gt_review_{(j//8)+1:03d}.jpg'
        index.append({'lote':number,'archivo_zip':package,'imagenes':len(batch),
                      'primer_caso':batch[0]['image_path'],
                      'puntaje_maximo':batch[0]['priority_score_v2'],
                      'puntaje_minimo':batch[-1]['priority_score_v2']})
        if args.metadata_only:
            continue
        with ZipFile(out/package,'w',ZIP_DEFLATED,compresslevel=6) as z:
            z.writestr('review_batch.csv',csv_bytes(batch,header))
            for j in range(0,len(batch),8):
                dest=out/f'_temporary_sheet_{number}_{j//8}.jpg'
                render_sheet(batch[j:j+8], args.image_root, dest)
                z.write(dest,arcname='contact_sheets/gt_review_{:03d}.jpg'.format(j//8+1))
                dest.unlink()
            for r in batch:
                original = args.image_root/r['image_path']
                if not original.is_file():
                    raise FileNotFoundError(original)
                z.write(original,arcname='original_images/'+Path(r['image_path']).name)
        print(f'Lote {number:02}/{total:02}: {len(batch)} imágenes -> {out/package}',flush=True)
    with (out/'indice_lotes_v2.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(index[0]));wr.writeheader();wr.writerows(index)
    with (out/'revision_pendiente_v2.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=header);wr.writeheader();wr.writerows(pending)
    summary={'total_images':735,'protected_strict_audited':80,'candidate_queue':655,
             'closed_reviewed_candidates':50,'previous_review_ambiguous':5,
             'pending_review_including_ambiguous':605,'corrections_cumulative_v2':18,
             'batch_size':args.batch_size,'batch_count':total,
             'only_planning':bool(args.metadata_only),
             'no_automatic_corrections':True,'source_manifest_sha256':EXPECTED_SHA}
    (out/'estado_revision_v2.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('=== GVIS: AUDITORIA GT ELPD - LOTES V2 ===',flush=True)
    print('735 imágenes; 80 GT estrictos protegidos; 655 en cola candidata',flush=True)
    print('Revisión cerrada:',len(closed), '| Ambiguos anteriores: 5',flush=True)
    print(f'Pendientes: {len(pending)} | Lotes: {total} | Correcciones automáticas: 0',flush=True)
    print('PRÓXIMO LOTE:', out/'gt_review_lote_01.zip',flush=True)
    print('RESULTADO: LOTES DE REVISION PREPARADOS; GT SIN MODIFICAR',flush=True)

if __name__ == '__main__':
    main()
