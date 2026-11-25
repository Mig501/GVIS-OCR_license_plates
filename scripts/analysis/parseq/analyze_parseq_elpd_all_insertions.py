#!/usr/bin/env python3
"""PASO 8: PARSeq, ELPD GT V17, origen de todas las inserciones puras.

Investiga las 127 predicciones refinadas con insercion pura, sin modificar el
checkpoint, el GT ni las predicciones previas. Registra los tokens antes de la
normalizacion para diferenciar inserciones AR, activaciones de simbolos en el
refinamiento y cambios de EOS.

Ruta prevista:
  scripts/analysis/parseq/analyze_parseq_elpd_all_insertions.py

Por defecto NO recibe argumentos en las tareas GVIS de VS Code.
Comprobacion de seleccion sin GPU (opcional): --prepare-only
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import time

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_CSV = PROJECT / 'results/audit/ELPD/gt_corrections_v17/parseq/predictions_verified_v17.csv'
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/all_insertions_stages_v1'
COUNT_EXPECTED = {'inicial': 46, 'interna': 34, 'final': 47}
RE_TEXT = re.compile(r'[^A-Z0-9]')
RE_CHAR = re.compile(r'^[A-Z0-9]$')


def normalize(text: str) -> str:
    return RE_TEXT.sub('', str(text).upper())


def is_ascii_alnum(token: str) -> bool:
    return bool(RE_CHAR.fullmatch(str(token).upper()))


def insertion_type(gt: str, pred: str) -> str | None:
    """Replica exactamente el criterio del Paso 3 (SequenceMatcher)."""
    if len(pred) <= len(gt):
        return None
    differences = [(t, i1, i2, j1, j2) for t, i1, i2, j1, j2
                   in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes()
                   if t != 'equal']
    if not differences or any(t != 'insert' for t, *_ in differences):
        return None
    if len(differences) == 1 and differences[0][1] == 0:
        return 'inicial'
    if len(differences) == 1 and differences[0][1] == len(gt):
        return 'final'
    return 'interna'


def align_insertions(gt: str, pred: str) -> list[dict]:
    return [{'en_gt_0based': i1, 'pred_inicio_0based': j1,
             'pred_fin_exclusivo_0based': j2, 'texto_insertado': pred[j1:j2]}
            for op, i1, _i2, j1, j2
            in SequenceMatcher(None, gt, pred, autojunk=False).get_opcodes()
            if op == 'insert']


def read_selection(path: Path) -> tuple[list[dict], list[dict], dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_path', 'ground_truth', 'prediction_normalized',
                    'exact_match', 'prediction_raw'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Faltan columnas en V17: {sorted(missing)}')
        rows = list(reader)
    if len(rows) != 733:
        raise ValueError(f'V17 esperado: 733 registros verificados, recibidos {len(rows)}')
    if len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Rutas repetidas en CSV de predicciones V17')
    if sum(r['exact_match'].strip() == '0' for r in rows) != 216:
        raise ValueError('El CSV V17 no contiene los 216 errores esperados de PARSeq')
    selected = []
    for row in rows:
        gt, pred = row['ground_truth'].strip(), row['prediction_normalized'].strip()
        if normalize(gt) != gt or normalize(pred) != pred:
            raise ValueError(f'Cadena no normalizada en V17: {row["image_path"]}')
        if str(int(gt == pred)) != row['exact_match'].strip():
            raise ValueError(f'Exact match inconsistente: {row["image_path"]}')
        if row['exact_match'].strip() != '0':
            continue
        kind = insertion_type(gt, pred)
        if kind:
            item = row.copy()
            item['tipo_insercion_pura'] = kind
            selected.append(item)
    counts = dict(Counter(r['tipo_insercion_pura'] for r in selected))
    if counts != COUNT_EXPECTED or len(selected) != 127:
        raise ValueError(f'No coincide la seleccion del Paso 3: {counts}, '
                         f'esperado {COUNT_EXPECTED}')
    selected.sort(key=lambda r: ('inicial', 'interna', 'final').index(r['tipo_insercion_pura']))
    # Python conserva el orden del CSV V17 dentro de cada categoria.
    return rows, selected, counts


def safe_image(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or '..' in p.parts or p.parts[0] != 'images':
        raise ValueError(f'Ruta insegura en V17: {rel}')
    full = root / p
    if not full.is_file():
        raise FileNotFoundError(f'No existe recorte ELPD: {full}')
    return full


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def dump_csv(path: Path, records: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def describe_logits(probabilities, itos: list[str], eos_id: int):
    """Preserva todos los tokens argmax hasta EOS, *antes* de normalizar."""
    import torch
    if probabilities.ndim != 2 or probabilities.shape[1] > len(itos):
        raise ValueError(f'Logits no compatibles con vocabulario: {tuple(probabilities.shape)}')
    argmax = probabilities.argmax(-1).tolist()
    eos_ix = argmax.index(eos_id) if eos_id in argmax else None
    limit = eos_ix if eos_ix is not None else len(argmax)
    tokens = [itos[i] for i in argmax[:limit]]
    raw = ''.join(tokens)
    mapping = [i for i, tok in enumerate(tokens) if is_ascii_alnum(tok)]
    norm = normalize(raw)
    # Verifica que la correspondencia normalizada conserva 1 token por carácter.
    if len(norm) != len(mapping):
        raise RuntimeError(f'La normalizacion no coincide con el mapeo de tokens: {raw!r}')
    return {'ids': argmax, 'tokens': tokens, 'raw': raw, 'normalized': norm,
            'eos_position_1based': eos_ix + 1 if eos_ix is not None else None,
            'length_logits': len(argmax), 'char_to_position_0based': mapping}


def classify_temporal(gt: str, ar: dict, refined: dict) -> str:
    ar_pred = ar['normalized']
    ref_pred = refined['normalized']
    if insertion_type(gt, ref_pred) is None:
        return 'refinado_no_es_insercion_pura'
    if ar_pred == gt:
        return 'insercion_visible_solo_tras_refinamiento'
    if insertion_type(gt, ar_pred) is not None:
        if ar_pred == ref_pred:
            return 'insercion_AR_se_mantiene_igual'
        return 'insercion_AR_cambia_en_refinamiento'
    return 'AR_con_otros_errores_y_refinado_insercion'


def token_transitions(ar: dict, ref: dict) -> list[dict]:
    # Solo compara los tokens de posiciones presentes antes de EOS en ambas etapas.
    # Una posicion numerica no equivale necesariamente a la misma letra real.
    out = []
    upto = min(len(ar['tokens']), len(ref['tokens']))
    for i in range(upto):
        a, b = ar['tokens'][i], ref['tokens'][i]
        if a == b:
            continue
        ca, cb = is_ascii_alnum(a), is_ascii_alnum(b)
        if not ca and cb:
            kind = 'simbolo_a_alfanumerico'
        elif ca and not cb:
            kind = 'alfanumerico_a_simbolo'
        elif ca and cb:
            kind = 'alfanumerico_a_alfanumerico'
        else:
            kind = 'simbolo_a_simbolo'
        out.append({'posicion_1based': i + 1, 'token_AR': a,
                    'token_refinado': b, 'tipo_transicion': kind})
    return out


def is_subsequence(sub: str, whole: str) -> bool:
    idx = 0
    for ch in whole:
        if idx < len(sub) and sub[idx] == ch:
            idx += 1
    return idx == len(sub)


def possible_inserted_positions(gt: str, pred: str) -> list[int]:
    """Indices 0-based que PODRIAN ser inserciones en una alineacion valida.

    En B + B + MW... ambos B son candidatos. Un unico alineamiento de
    SequenceMatcher elegiria solo uno, ocultando cambios '.' -> 'B'.
    """
    if len(pred) <= len(gt) or not is_subsequence(gt, pred):
        return []
    return [i for i in range(len(pred))
            if is_subsequence(gt, pred[:i] + pred[i + 1:])]


def event_inserted_tokens(gt: str, st: dict, other: dict | None = None) -> list[dict]:
    """Alineacion textual aproximada: ambigua en repeticiones consecutivas."""
    return [
        {'posicion_pred_norm_1based': j + 1,
         'posicion_logit_1based': st['char_to_position_0based'][j] + 1,
         'token': st['normalized'][j],
         'token_otro_en_misma_posicion': (
             (other['tokens'][st['char_to_position_0based'][j]] if
              other is not None and st['char_to_position_0based'][j] < len(other['tokens'])
              else '') if other is not None else ''),
        }
        for segment in align_insertions(gt, st['normalized'])
        for j in range(segment['pred_inicio_0based'], segment['pred_fin_exclusivo_0based'])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Validar los 127 casos sin GPU ni lectura de imagenes')
    args = parser.parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f'Falta el CSV de PARSeq ELPD V17: {args.csv}')
    records, selected, expected_counts = read_selection(args.csv)
    print('=== PASO 8 | PARSEQ ELPD V17 | 127 INSERCIONES PURAS ===', flush=True)
    print('Verificadas:', len(records), '| fallos: 216 | inserciones:', expected_counts, flush=True)
    print('Checkpoint: PARSeq preentrenado, AR (0) y AR + 1 refinamiento', flush=True)
    print('CSV V17 SHA256:', sha256(args.csv), flush=True)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'La carpeta de salida contiene resultados: {args.out}')
    if args.prepare_only:
        print('PREFLIGHT OK: seleccion de 127 casos validada; NO se ha ejecutado inferencia.', flush=True)
        return

    import torch
    from PIL import Image
    from torchvision import transforms as T
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible: ejecutar con GVIS: Ejecutar en GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True)
    model.eval().cuda()
    core = model.model
    if not bool(core.decode_ar) or int(core.refine_iters) != 1:
        raise RuntimeError(f'Se esperaba PARSeq AR + 1 refinamiento; '
                           f'decode_ar={core.decode_ar}, refine_iters={core.refine_iters}')
    if tuple(int(v) for v in model.hparams.img_size) != (32, 128):
        raise RuntimeError('El checkpoint no utiliza la resolucion esperada 32x128')
    tokenizer = model.tokenizer
    itos = list(tokenizer._itos)
    eos = int(tokenizer.eos_id)
    transform = T.Compose([
        T.Resize(tuple(model.hparams.img_size), T.InterpolationMode.BICUBIC),
        T.ToTensor(), T.Normalize(0.5, 0.5),
    ])
    print('GPU:', torch.cuda.get_device_name(0), '| imagenes:', len(selected), flush=True)
    started = time.perf_counter()
    stage_rows = []
    transitions_rows = []
    token_rows = []
    selections_rows = []
    total_reproduced = 0
    by_kind = defaultdict(Counter)
    transitions_count = Counter()
    eos_changes = Counter()
    pre_ref_raw_match = 0
    saved_iters = int(core.refine_iters)
    try:
        with torch.inference_mode():
            for n, row in enumerate(selected, 1):
                gt = row['ground_truth']
                rel = row['image_path']
                with Image.open(safe_image(args.image_root, rel)) as im:
                    x = transform(im.convert('RGB')).unsqueeze(0).cuda()
                probs_map = {}
                stage = {}
                for name, iterations in (('AR', 0), ('refinado', 1)):
                    core.refine_iters = iterations
                    logits = model(x)
                    probs = logits.softmax(-1).float().cpu()[0]
                    decoded = describe_logits(probs, itos, eos)
                    official_text, _ = tokenizer.decode(probs.unsqueeze(0))
                    if decoded['raw'] != official_text[0]:
                        raise RuntimeError(f'{rel}: diferencia entre tokens y decode() '
                                           f'en {name}: {decoded["raw"]!r} vs {official_text[0]!r}')
                    probs_map[name] = probs
                    stage[name] = decoded
                ar, ref = stage['AR'], stage['refinado']
                original_matches = ref['normalized'] == row['prediction_normalized']
                total_reproduced += int(original_matches)
                raw_match = ref['raw'] == row['prediction_raw']
                pre_ref_raw_match += int(raw_match)
                verdict = classify_temporal(gt, ar, ref)
                if not original_matches:
                    verdict = 'NO_REPRODUCIDO'
                by_kind[row['tipo_insercion_pura']][verdict] += 1
                changes = token_transitions(ar, ref)
                for event in changes:
                    transitions_count[event['tipo_transicion']] += int(original_matches)
                    transitions_rows.append({'image_path': rel,
                                             'tipo_insercion_pura': row['tipo_insercion_pura'],
                                             'coincide_V17': int(original_matches),
                                             **event})
                p_ar, p_ref = probs_map['AR'], probs_map['refinado']
                if ar['eos_position_1based'] is not None and ref['eos_position_1based'] is not None:
                    diff_eos = ref['eos_position_1based'] - ar['eos_position_1based']
                    eos_status = 'sin_cambio' if diff_eos == 0 else 'desplazado'
                else:
                    diff_eos = None
                    eos_status = 'eos_no_presente_en_alguna_etapa'
                if original_matches:
                    eos_changes[eos_status] += 1
                # Los tokens insertados se alinean sobre las cadenas normalizadas.
                inserted_ar = event_inserted_tokens(gt, ar)
                inserted_ref = event_inserted_tokens(gt, ref, ar)
                possible_ref = possible_inserted_positions(gt, ref['normalized'])
                # No atribuir una posicion univoca en repeticiones consecutivas.
                # La activacion de la segunda B de BBMW6220, por ejemplo, debe
                # reconocerse aunque SequenceMatcher elija la primera B.
                possible_logits_ref = {
                    ref['char_to_position_0based'][j] + 1 for j in possible_ref
                }
                activated = [ev for ev in changes if ev['tipo_transicion'] == 'simbolo_a_alfanumerico']
                newly_visible_inserted = [ev for ev in activated
                                          if ev['posicion_1based'] in possible_logits_ref]
                # Probabilidad EOS al finalizar GT: util solo en insercion final
                # y si GT es prefijo literal de la salida de esa etapa.
                p_eos_after_gt_ar = float(p_ar[len(gt), eos]) if (
                    ar['normalized'].startswith(gt) and len(p_ar) > len(gt)) else None
                p_eos_after_gt_ref = float(p_ref[len(gt), eos]) if (
                    ref['normalized'].startswith(gt) and len(p_ref) > len(gt)) else None

                stage_rows.append({
                    'image_path': rel,
                    'split': row.get('scenario', ''),
                    'tipo_insercion_pura': row['tipo_insercion_pura'],
                    'ground_truth': gt,
                    'prediccion_V17': row['prediction_normalized'],
                    'raw_V17': row['prediction_raw'],
                    'raw_AR': ar['raw'],
                    'normalizada_AR': ar['normalized'],
                    'raw_refinado': ref['raw'],
                    'normalizada_refinado': ref['normalized'],
                    'coincide_V17': int(original_matches),
                    'coincide_raw_V17': int(raw_match),
                    'clasificacion_temporal': verdict,
                    'tipo_AR_pura': insertion_type(gt, ar['normalized']) or '',
                    'n_inserciones_norm_AR': len(inserted_ar),
                    'n_inserciones_norm_refinado': len(inserted_ref),
                    'caracteres_insertados_AR': json.dumps(inserted_ar, ensure_ascii=False),
                    'caracteres_insertados_refinado': json.dumps(inserted_ref, ensure_ascii=False),
                    'posiciones_posibles_insertadas_refinado_1based':
                        json.dumps([j+1 for j in possible_ref], ensure_ascii=False),
                    'alineacion_de_insercion_ambigua': int(
                        len(possible_ref) > len(ref['normalized'])-len(gt)),
                    'activaciones_simbolo_a_alfanumerico': len(activated),
                    'activaciones_en_posiciones_insertadas_refinado': len(newly_visible_inserted),
                    'detalle_activaciones': json.dumps(activated, ensure_ascii=False),
                    'EOS_AR_1based': ar['eos_position_1based'] or '',
                    'EOS_refinado_1based': ref['eos_position_1based'] or '',
                    'diferencia_posicion_EOS': diff_eos if diff_eos is not None else '',
                    'estado_EOS': eos_status,
                    'n_logits_AR': ar['length_logits'],
                    'n_logits_refinado': ref['length_logits'],
                    'p_EOS_tras_GT_AR_si_prefijo': round(p_eos_after_gt_ar, 8) if p_eos_after_gt_ar is not None else '',
                    'p_EOS_tras_GT_refinado_si_prefijo': round(p_eos_after_gt_ref, 8) if p_eos_after_gt_ref is not None else '',
                })
                selections_rows.append({'image_path': rel, 'tipo': row['tipo_insercion_pura'],
                                        'ground_truth': gt,
                                        'prediction_normalized_V17': row['prediction_normalized'],
                                        'prediction_raw_V17': row['prediction_raw']})
                for name, d in (('AR', ar), ('refinado', ref)):
                    probs = probs_map[name]
                    for pos, dist in enumerate(probs):
                        values, indices = torch.topk(dist, k=2)
                        ids = indices.tolist()
                        token_rows.append({
                            'image_path': rel, 'etapa': name,
                            'posicion_1based': pos + 1,
                            'token_max': itos[ids[0]],
                            'p_token_max': round(float(values[0]), 8),
                            'token_segundo': itos[ids[1]],
                            'p_token_segundo': round(float(values[1]), 8),
                            'p_EOS': round(float(dist[eos]), 8),
                            'hasta_primer_EOS': int(pos < d['eos_position_1based']) if d['eos_position_1based'] else 1,
                        })
                if n <= 10 or n % 10 == 0 or n == len(selected):
                    print(f'[{n:03d}/{len(selected)}] {row["tipo_insercion_pura"]:<7} '
                          f'GT={gt} | AR={ar["raw"]} | REF={ref["raw"]} '
                          f'| V17={"OK" if original_matches else "NO"}', flush=True)
    finally:
        core.refine_iters = saved_iters

    args.out.mkdir(parents=True, exist_ok=False)
    dump_csv(args.out / 'seleccion_127.csv', selections_rows, list(selections_rows[0]))
    dump_csv(args.out / 'comparacion_127.csv', stage_rows, list(stage_rows[0]))
    dump_csv(args.out / 'tokens_por_posicion.csv', token_rows, list(token_rows[0]))
    dump_csv(args.out / 'transiciones_tokens.csv', transitions_rows,
             ['image_path', 'tipo_insercion_pura', 'coincide_V17',
              'posicion_1based', 'token_AR', 'token_refinado', 'tipo_transicion'])
    reproduced = [r for r in stage_rows if r['coincide_V17'] == 1]
    verdicts = Counter(r['clasificacion_temporal'] for r in reproduced)
    insertion_activation_cases = sum(r['activaciones_en_posiciones_insertadas_refinado'] > 0
                                     for r in reproduced)
    summary = {
        'paso': 8,
        'dataset': 'ELPD V17: 733 GT verificados',
        'modelo': 'PARSeq preentrenado (AR 0 refinamientos vs AR + 1 refinamiento)',
        'n_imagenes_verificadas': len(records),
        'n_fallos_originales': 216,
        'inserciones_puras_por_tipo': expected_counts,
        'inserciones_puras_seleccionadas': len(selected),
        'n_reproducidas_refinado_V17': total_reproduced,
        'n_con_raw_refinado_coincidente_V17': pre_ref_raw_match,
        'n_excluidas_por_no_reproduccion': len(selected) - total_reproduced,
        'clasificacion_temporal_solo_reproducidas': dict(verdicts),
        'clasificacion_temporal_por_tipo': {k: dict(v) for k, v in by_kind.items()},
        'casos_con_activacion_simbolo_a_alfanumerico_en_posicion_posiblemente_insertada': insertion_activation_cases,
        'transiciones_de_token_solo_reproducidas': dict(transitions_count),
        'estado_EOS_solo_reproducidas': dict(eos_changes),
        'cambio_de_posicion_EOS_solo_si_ambos_EOS': dict(Counter(
            str(r['diferencia_posicion_EOS']) for r in reproduced
            if r['diferencia_posicion_EOS'] != '')),
        'tiempo_inferencia_s': round(time.perf_counter() - started, 2),
        'checkpoint_modificado': False, 'GT_modificado': False,
        'csv_v17_sha256': sha256(args.csv),
        'metodologia': [
            'Seleccion identica al Paso 3 mediante SequenceMatcher: insercion pura normalizada.',
            'Cada imagen se infiere individualmente con misma entrada y preprocesamiento original.',
            'La etiqueta temporal se calcula solo para casos que reproducen el CSV V17.',
            'El token sin filtrar puede incluir signos que desaparecen al normalizar (A-Z0-9).',
            'Un simbolo convertido en alfanumerico es una posible causa de insercion visible.',
            'Las comparaciones de EOS se realizan sobre primeros EOS argmax de cada etapa.',
            'La probabilidad de EOS al final del GT solo se calcula cuando GT es prefijo literal.',
            'Una insercion puede tener varias posiciones validas en caracteres repetidos; se guardan posiciones posibles.',
            'Las clasificaciones son descriptivas y pueden solaparse con cambios de EOS o tokens.',
            'No se atribuye causalidad a una capa de PARSeq ni se estima prevalencia de otras clases de errores.',
        ],
        'archivos':['seleccion_127.csv', 'comparacion_127.csv', 'tokens_por_posicion.csv',
                    'transiciones_tokens.csv', 'resumen.json'],
    }
    (args.out / 'resumen.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Reproducidas V17:', total_reproduced, '/', len(selected), flush=True)
    print('Mecanismos temporales (solo reproducidas):', dict(verdicts), flush=True)
    print('EOS (solo reproducidas):', dict(eos_changes), flush=True)
    print('RESULTADO: 127 INSERCIONES ANALIZADAS; TOKENS CRUDOS, REFINAMIENTO Y EOS EXPORTADOS', flush=True)
    print('SALIDA:', args.out, flush=True)


if __name__ == '__main__':
    main()
