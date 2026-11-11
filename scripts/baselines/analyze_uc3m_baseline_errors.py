#!/usr/bin/env python3
"""Auditoria de errores y hojas de contacto de baselines OCR sobre UC3M-LP.

Ejecucion habitual (sin argumentos, desde una tarea CPU de VS Code):
    python scripts/baselines/analyze_uc3m_baseline_errors.py

Solo un modelo:
    python scripts/baselines/analyze_uc3m_baseline_errors.py --model parseq_pretrained

Todas las hojas (pueden ser mas de 100 entre los cuatro modelos):
    python scripts/baselines/analyze_uc3m_baseline_errors.py --all-sheets

ENTRADAS:
    results/baselines/UC3M-LP/<modelo>/predictions.csv y summary.json
    /home/miguel.sanchez/datasets/ocr_matriculas/processed/UC3M-LP_LPO/{manifest.csv,images/}

SALIDAS POR MODELO:
    results/baselines/UC3M-LP/<modelo>/error_analysis/
        errors.csv                 Discrepancias respecto al texto visible
        all_predictions_review.csv Todos los ejemplos con indicadores diagnosticos
        error_analysis.md          Informe, advertencias y ejemplos
        contact_sheet_errors_01.jpg, ... (hasta --max-sheets, por defecto 5)
        contact_sheet_high_confidence_errors.jpg
        contact_sheet_truncations.jpg
        contact_sheet_mask_ambiguity.jpg
        contact_sheet_low_confidence_visible_exact.jpg

IMPORTANTE:
  * El GT contiene '*' anonimizados. No reconstruimos ni inferimos esos caracteres.
  * 'Error visible' significa diferencia textual frente a GT sin '*'; NO prueba
    por si solo que el OCR sea incorrecto, pues puede producir caracteres en
    posiciones anonimizadas. Las ediciones no son causas visuales.
  * EM parcial estricto y distancia Levenshtein con mascara se mantienen como
    metricas originales del baseline, pero no son EM/CER completos.
  * Ni datasets, ni checkpoints ni predictions.csv se modifican.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATA = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DATASET_REL = Path('processed/UC3M-LP_LPO')
BASELINE_REL = Path('results/baselines/UC3M-LP')
EXPECTED_TEST = 491

OUTPUT_FIELDS = [
    'image_path', 'view_id', 'scenario', 'source_path', 'annotation_path',
    'ground_truth_masked', 'known_mask', 'ground_truth_visible',
    'prediction_raw', 'prediction_normalized', 'confidence',
    'gt_length', 'visible_gt_length', 'pred_length',
    'length_match', 'partial_exact_match', 'masked_edit_distance',
    'visible_exact_match', 'visible_edit_distance',
    'visible_error_category', 'visible_edit_operations', 'length_vs_visible',
    'visible_is_suffix', 'visible_is_substring', 'potential_mask_ambiguity',
]


def safe_rel_image(path_str: str) -> Path:
    """Impide que un CSV lea archivos arbitrarios fuera del dataset."""
    if not path_str or '\\' in path_str:
        raise ValueError(f'Ruta de imagen invalida: {path_str!r}')
    p = Path(path_str)
    if p.is_absolute() or '..' in p.parts or len(p.parts) != 2 or p.parts[0] != 'images':
        raise ValueError(f'Ruta insegura: {path_str!r}')
    return p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def visible_alignment(gt: str, pred: str) -> list[tuple[str, str]]:
    """Una alineacion Levenshtein optima determinista, desde GT visible a pred."""
    n, m = len(gt), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost,
                           dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1)
    changes: list[tuple[str, str]] = []
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (gt[i - 1] != pred[j - 1]):
            if gt[i - 1] != pred[j - 1]:
                changes.append(('substitution', f'{gt[i - 1]}>{pred[j - 1]}'))
            i -= 1
            j -= 1
        elif j and dp[i][j] == dp[i][j - 1] + 1:
            changes.append(('insertion', f'+{pred[j - 1]}'))
            j -= 1
        else:
            changes.append(('deletion', f'-{gt[i - 1]}'))
            i -= 1
    changes.reverse()
    assert len(changes) == dp[n][m]
    return changes


def wildcard_distance(masked_gt: str, pred: str) -> int:
    """Levenshtein conservador original: '*' sustitucion 0, indels coste 1."""
    prev = list(range(len(pred) + 1))
    for i, ch in enumerate(masked_gt, start=1):
        current = [i]
        for j, p in enumerate(pred, start=1):
            cost = 0 if ch == '*' or ch == p else 1
            current.append(min(current[-1] + 1, prev[j] + 1,
                               prev[j - 1] + cost))
        prev = current
    return prev[-1]


def category(changes: list[tuple[str, str]]) -> str:
    ops = {op for op, _ in changes}
    return 'correct' if not ops else next(iter(ops)) if len(ops) == 1 else 'mixed'


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'No esta el manifest de UC3M: {path}')
    rows = read_csv(path)
    res = {}
    for row in rows:
        if row.get('scenario') != 'test':
            continue
        key = row['image_path']
        safe_rel_image(key)
        if key in res:
            raise ValueError(f'Duplicado en manifest: {key}')
        res[key] = row
    if len(res) != EXPECTED_TEST:
        raise ValueError(f'Manifest: {len(res)} test, esperadas {EXPECTED_TEST}')
    return res


def read_and_score(model_dir: Path, manifest: dict[str, dict[str, str]],
                   manifest_hash: str) -> tuple[list[dict], dict]:
    path = model_dir / 'predictions.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_rows = read_csv(path)
    if len(raw_rows) != len(manifest):
        raise ValueError(f'{path}: {len(raw_rows)} filas, esperadas {len(manifest)}')
    required = {
        'image_path', 'scenario', 'ground_truth_masked', 'known_mask',
        'ground_truth_visible', 'prediction_normalized', 'partial_exact_match',
        'masked_edit_distance', 'confidence',
    }
    if raw_rows and required - set(raw_rows[0]):
        raise ValueError(f'{path}: faltan columnas {sorted(required - set(raw_rows[0]))}')
    summary_path = model_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    if summary:
        if summary.get('manifest_sha256') != manifest_hash:
            raise ValueError(f'{summary_path}: hash del manifest no coincide')
        if int(summary.get('dataset_size', -1)) != len(manifest):
            raise ValueError(f'{summary_path}: dataset_size diferente del manifest')
    seen = set()
    items = []
    for row in raw_rows:
        rel = row['image_path']
        safe_rel_image(rel)
        if rel in seen or rel not in manifest:
            raise ValueError(f'Imagen duplicada o no incluida en test: {rel}')
        seen.add(rel)
        orig = manifest[rel]
        gt = row['ground_truth_masked']
        mask = row['known_mask']
        visible = row['ground_truth_visible']
        if (row['scenario'] != 'test' or gt != orig['text_normalized']
                or visible != orig['text_visible'] or mask != orig['known_mask']):
            raise ValueError(f'Desacuerdo GT/predictions/manifest: {rel}')
        if not gt or '*' not in gt or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*' for c in gt):
            raise ValueError(f'GT parcial no valido: {rel} -> {gt}')
        if mask != ''.join('0' if c == '*' else '1' for c in gt) or visible != gt.replace('*', ''):
            raise ValueError(f'known_mask o GT visible incorrecto en {rel}')
        pred = row['prediction_normalized']
        if any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in pred):
            raise ValueError(f'Prediccion no normalizada en {rel}')
        conf = float(row['confidence']) if row['confidence'] else float('nan')
        length_match = int(len(pred) == len(gt))
        aligned_match = int(length_match and all(
            c == '*' or c == pred[i] for i, c in enumerate(gt)))
        mask_dist = wildcard_distance(gt, pred)
        if int(row['partial_exact_match']) != aligned_match:
            raise ValueError(f'EM parcial inconsistent en {rel}')
        if int(row['masked_edit_distance']) != mask_dist:
            raise ValueError(f'Distancia con mascara inconsistente en {rel}')
        changes = visible_alignment(visible, pred)
        is_exact = int(visible == pred)
        if len(pred) == len(visible):
            length_status = 'equal'
        elif len(pred) > len(visible):
            length_status = 'longer'
        else:
            length_status = 'shorter'
        item = dict(row)
        item.update(
            gt_length=len(gt), visible_gt_length=len(visible), pred_length=len(pred),
            length_match=length_match, partial_exact_match=aligned_match,
            masked_edit_distance=mask_dist,
            visible_exact_match=is_exact, visible_edit_distance=len(changes),
            visible_error_category=category(changes),
            visible_edit_operations=' '.join(s for _, s in changes),
            length_vs_visible=length_status,
            visible_is_suffix=int(bool(visible) and pred.endswith(visible)),
            visible_is_substring=int(bool(visible) and visible in pred),
            # Alto desacuerdo de metricas: transcripcion visible exacta pero falla EM parcial estricto.
            potential_mask_ambiguity=int(is_exact and not aligned_match),
            confidence_num=conf,
            ops=changes,
        )
        items.append(item)
    if set(manifest) != seen:
        raise ValueError(f'{path}: faltan imagenes de test: {len(set(manifest) - seen)}')
    total_known = sum(len(r['ground_truth_visible']) for r in items)
    total_masked_dist = sum(r['masked_edit_distance'] for r in items)
    strict = sum(r['partial_exact_match'] for r in items)
    if summary:
        expected = [('partial_exact_matches', strict),
                    ('masked_edit_errors', total_masked_dist),
                    ('known_gt_characters', total_known)]
        for key, computed in expected:
            if int(summary.get(key, -1)) != computed:
                raise ValueError(f'{model_dir.name} summary.{key} != valor recalculado: {computed}')
    return items, summary


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def safe_short(s: str, n: int = 43) -> str:
    return s if len(s) <= n else s[:n - 3] + '...'


def draw_sheet(rows: list[dict], path: Path, image_root: Path, title: str) -> list[str]:
    """Genera una hoja de 12 miniaturas y devuelve incidencias con las imagenes."""
    if not rows:
        return []
    from PIL import Image, ImageDraw, ImageOps
    cols, rows_per_page = 3, 4
    w, h = 510, 248
    canvas = Image.new('RGB', (cols * w + 12, rows_per_page * h + 92), '#f3f5f8')
    d = ImageDraw.Draw(canvas)
    d.text((16, 12), safe_short(title, 83), fill='#18212d', font=load_font(22, True))
    d.text((16, 45), 'GT * = posicion anonima; EXTRA vs visible NO implica necesariamente error OCR',
           fill='#505866', font=load_font(13))
    issues = []
    for i, row in enumerate(rows[:12]):
        x = 7 + (i % cols) * w
        y = 75 + (i // cols) * h
        d.rounded_rectangle((x, y, x + w - 9, y + h - 9), radius=9,
                            fill='white', outline='#cdd6e1', width=2)
        d.text((x + 11, y + 7), safe_short(Path(row['image_path']).name, 43),
               fill='#202a37', font=load_font(15, True))
        d.text((x + 11, y + 33), f'GT parcial: {safe_short(row["ground_truth_masked"], 30)}',
               fill='#3f5774', font=load_font(16, True))
        d.text((x + 11, y + 59), f'GT visible: {safe_short(row["ground_truth_visible"], 30)}',
               fill='#276448', font=load_font(16, True))
        d.text((x + 11, y + 85), f'Pred: {safe_short(row["prediction_normalized"], 37)}',
               fill='#a53f39', font=load_font(15, True))
        conf = row['confidence_num']
        conftext = 'n/d' if not math.isfinite(conf) else f'{conf:.3f}'
        d.text((x + 11, y + 113),
               f'Dvis={row["visible_edit_distance"]} | Dmask={row["masked_edit_distance"]} | conf={conftext}',
               fill='#454c58', font=load_font(13))
        source = image_root / safe_rel_image(row['image_path'])
        try:
            with Image.open(source) as im:
                im.load()
                cropped = ImageOps.contain(im.convert('RGB'), (w - 26, 93),
                                            method=Image.Resampling.LANCZOS)
            paste_x = x + (w - 9 - cropped.width) // 2
            paste_y = y + 138 + (93 - cropped.height) // 2
            canvas.paste(cropped, (paste_x, paste_y))
        except (OSError, ValueError) as e:
            issues.append(f'{row["image_path"]}: {type(e).__name__}: {e}')
            d.text((x + 11, y + 160), 'IMAGEN AUSENTE / NO LEGIBLE',
                   fill='#b33939', font=load_font(15, True))
    canvas.save(path, quality=88, optimize=True)
    return issues


def csv_save(path: Path, items: list[dict]):
    with path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(items)


def markdown_report(model_name: str, items: list[dict], summary: dict,
                    max_sheets: int, sheets_done: int) -> str:
    n = len(items)
    discrepancy = [r for r in items if not r['visible_exact_match']]
    exact = n - len(discrepancy)
    strict = sum(r['partial_exact_match'] for r in items)
    known = sum(len(r['ground_truth_visible']) for r in items)
    edits = sum(r['visible_edit_distance'] for r in items)
    masked_edits = sum(r['masked_edit_distance'] for r in items)
    equal_len = sum(r['length_match'] for r in items)
    mask_amb = sum(r['potential_mask_ambiguity'] for r in items)
    categories = Counter(r['visible_error_category'] for r in discrepancy)
    ops = Counter(op for r in discrepancy for op, _ in r['ops'])
    longer = sum(r['length_vs_visible'] == 'longer' for r in items)
    shorter = sum(r['length_vs_visible'] == 'shorter' for r in items)
    empty = sum(not r['prediction_normalized'] for r in items)
    suffix = sum(r['visible_is_suffix'] and r['length_vs_visible'] == 'longer' for r in items)
    lines = [
        f'# UC3M-LP — Revision visual del baseline: {model_name}', '',
        '## Alcance y validacion', '',
        f'- Conjunto de test oficial: **{n} recortes** (sin utilizar train).',
        '- Las cadenas GT contienen `*` en posiciones anonimizadas; el texto real de esas posiciones es desconocido.',
        f'- Checkpoint SHA-256: `{summary.get("checkpoint_sha256", "consultar summary.json")}`.',
        f'- Manifest SHA-256: `{summary.get("manifest_sha256", "consultar manifest.csv")}`.',
        '- Los resultados se obtienen de `predictions.csv`, no se ejecuta el modelo nuevamente.',
        '', '## Metricas originales del baseline', '',
        '| Metrica | Resultado |', '|---|---:|',
        f'| EM parcial estricto (requiere longitud original) | **{100 * strict / n:.2f}% ({strict}/{n})** |',
        f'| Distancia de edicion con mascara (proxy, no CER) | **{100 * masked_edits / known:.2f}% ({masked_edits}/{known})** |',
        f'| Predicciones de longitud original | {equal_len}/{n} |',
        '', '## Diagnosticos del texto visible (no son metricas de matricula completa)', '',
        '| Indicador | Resultado |', '|---|---:|',
        f'| Prediccion igual al GT visible | **{exact}/{n} ({100 * exact / n:.2f}%)** |',
        f'| Prediccion diferente del GT visible | **{len(discrepancy)}/{n}** |',
        f'| Tasa de edicion vs texto visible (diagnostico) | **{100 * edits / known:.2f}% ({edits}/{known})** |',
        f'| Prediccion mas larga que texto visible | {longer}/{n} |',
        f'| Prediccion mas corta que texto visible | {shorter}/{n} |',
        f'| GT visible es sufijo de una prediccion mas larga | {suffix}/{n} |',
        f'| GT visible exacto, pero no coincide EM parcial estricto | {mask_amb}/{n} |',
        f'| Salidas vacias | {empty}/{n} |',
        '', '## Clasificacion DESCRIPTIVA de discrepancias respecto al texto visible', '',
        '**Aviso:** una insercion frente a `ground_truth_visible` puede corresponder a una posicion',
        'anonimizada. Estas categorias no equivalen a errores OCR demostrados.', '',
        '| Categoria de alineacion | Imagenes |', '|---|---:|',
    ]
    for kind in ('insertion', 'substitution', 'deletion', 'mixed'):
        lines.append(f'| {kind} | {categories[kind]} |')
    lines += [
        '', f'Operaciones en una alineacion minima: inserciones={ops["insertion"]}, '
        f'sustituciones={ops["substitution"]}, eliminaciones={ops["deletion"]}.',
        '', '## Casos con mayor discrepancia de texto visible', '',
        '| Imagen | GT parcial | GT visible | Prediccion | Dvisible | Dmask | Confianza |',
        '|---|---|---|---|---:|---:|---:|',
    ]
    rank = sorted(discrepancy, key=lambda r: (-r['visible_edit_distance'], r['image_path']))
    for r in rank[:25]:
        conf = r['confidence_num']
        c = 'n/d' if not math.isfinite(conf) else f'{conf:.3f}'
        lines.append(f'| `{Path(r["image_path"]).name}` | `{r["ground_truth_masked"]}` | '
                     f'`{r["ground_truth_visible"]}` | `{r["prediction_normalized"]}` | '
                     f'{r["visible_edit_distance"]} | {r["masked_edit_distance"]} | {c} |')
    lines += [
        '', '## Hojas de contacto generadas', '',
        f'- Hojas principales: **{sheets_done}** (limite configurado: '
        + ('todas' if max_sheets < 0 else str(max_sheets)) + ').',
        '- `contact_sheet_errors_XX.jpg`: discrepancias respecto al GT visible ordenadas por distancia.',
        '- `contact_sheet_high_confidence_errors.jpg`: discrepancias de confianza elevada.',
        '- `contact_sheet_truncations.jpg`: predicciones mas cortas que el texto visible.',
        '- `contact_sheet_mask_ambiguity.jpg`: texto visible exacto pero EM parcial estricto fallido.',
        '- `contact_sheet_low_confidence_visible_exact.jpg`: coincidencias visibles con menor confianza.',
        '- `errors.csv`: **todas** las discrepancias visibles, no solo las mostradas en las hojas.',
        '- `all_predictions_review.csv`: los 491 ejemplos y sus indicadores.',
        '', '## Limitaciones y recomendaciones', '',
        '1. El `*` no es un caracter real: NUNCA se debe sustituir por un valor supuesto.',
        '2. El EM parcial estricto falla cuando la prediccion omite una posicion anonimizada, aunque lea todos los caracteres conocidos.',
        '3. La distancia con mascara usa sustitucion gratis en `*`, pero inserciones/eliminaciones cuestan 1. No es un CER convencional.',
        '4. La comparacion del GT visible eliminando `*` puede penalizar predicciones de caracteres en posiciones ocultas. Es un diagnostico, no una clasificacion concluyente de errores.',
        '5. El tipo de operacion proviene de una alineacion Levenshtein optima, que puede no ser unica; no informa del motivo visual del fallo.',
        '6. La puntuacion de confianza no esta calibrada como probabilidad de correccion.',
        '7. Revisar visualmente casos con prefijos, escudos, mascaras, repeticiones y truncamientos. Las miniaturas nunca alteran los PNG originales.',
        '8. Antes de comparar modelos de forma definitiva, establecer una alineacion validada que admita posiciones anonimizadas opcionalmente omitidas.',
        '',
    ]
    return '\n'.join(lines)


def analyze_one(model_dir: Path, dataset_root: Path, manifest: dict[str, dict[str, str]],
                manifest_hash: str, max_sheets: int) -> list[str]:
    items, summary = read_and_score(model_dir, manifest, manifest_hash)
    out = model_dir / 'error_analysis'
    out.mkdir(parents=True, exist_ok=True)
    # Evita hojas antiguas cuando se reduce --max-sheets en una ejecucion posterior.
    for obsolete in out.glob('contact_sheet_*.jpg'):
        obsolete.unlink()
    wrong = [r for r in items if not r['visible_exact_match']]
    correct_visible = [r for r in items if r['visible_exact_match']]
    wrong.sort(key=lambda r: (-r['visible_edit_distance'], r['image_path']))
    csv_save(out / 'errors.csv', wrong)
    csv_save(out / 'all_predictions_review.csv', items)
    issues = []
    max_count = len(wrong) if max_sheets < 0 else max_sheets * 12
    main_cases = wrong[:max_count]
    for k, offset in enumerate(range(0, len(main_cases), 12), 1):
        page = main_cases[offset:offset + 12]
        issues += draw_sheet(page, out / f'contact_sheet_errors_{k:02d}.jpg', dataset_root,
                             f'{model_dir.name} / UC3M-LP / discrepancias {offset+1}-{offset+len(page)}')
    highconf = sorted(wrong, key=lambda r: (-float(r['confidence_num']) if math.isfinite(r['confidence_num']) else 1.0,
                                            -r['visible_edit_distance'], r['image_path']))[:12]
    shorter = sorted((r for r in wrong if r['length_vs_visible'] == 'shorter'),
                     key=lambda r: (r['pred_length'] - r['visible_gt_length'], r['image_path']))[:12]
    ambig = sorted((r for r in items if r['potential_mask_ambiguity']),
                   key=lambda r: (-r['confidence_num'] if math.isfinite(r['confidence_num']) else 1.0,
                                  r['image_path']))[:12]
    lowconf = sorted(correct_visible, key=lambda r: (r['confidence_num'] if math.isfinite(r['confidence_num']) else 1.0,
                                                      r['image_path']))[:12]
    for label, rows, title in [
        ('contact_sheet_high_confidence_errors.jpg', highconf, 'Discrepancias visibles de alta confianza'),
        ('contact_sheet_truncations.jpg', shorter, 'Predicciones cortas: posibles truncamientos'),
        ('contact_sheet_mask_ambiguity.jpg', ambig, 'GT visible correcto, EM parcial estricto incorrecto'),
        ('contact_sheet_low_confidence_visible_exact.jpg', lowconf, 'GT visible coincidente, menor confianza'),
    ]:
        if rows:
            issues += draw_sheet(rows, out / label, dataset_root, f'{model_dir.name} / {title}')
    n_sheets = (len(main_cases) + 11) // 12
    (out / 'error_analysis.md').write_text(
        markdown_report(model_dir.name, items, summary, max_sheets, n_sheets), encoding='utf-8')
    if issues:
        (out / 'image_issues.txt').write_text('\n'.join(issues) + '\n', encoding='utf-8')
    elif (out / 'image_issues.txt').exists():
        (out / 'image_issues.txt').unlink()
    known = sum(len(r['ground_truth_visible']) for r in items)
    num_exact = sum(r['visible_exact_match'] for r in items)
    num_strict = sum(r['partial_exact_match'] for r in items)
    print(f'\n=== {model_dir.name} ===', flush=True)
    print(f'Test: {len(items)} | EM parcial estricto: {num_strict}/{len(items)}', flush=True)
    print(f'Coincidencia diagnostica visible: {num_exact}/{len(items)} ({100*num_exact/len(items):.2f}%)', flush=True)
    print(f'Edicion diagnostica visible: {sum(r["visible_edit_distance"] for r in items)}/{known}', flush=True)
    print(f'Discrepancias visibles: {len(wrong)} | hojas principales: {n_sheets}', flush=True)
    print(f'Incidencias con las imagenes: {len(issues)}', flush=True)
    print(f'Salida: {out}', flush=True)
    return issues


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=PROJECT)
    p.add_argument('--dataset-root', type=Path, default=DATA / DATASET_REL)
    p.add_argument('--model', help='Solo un modelo (ej.: svtrv2_tiny); por defecto todos.')
    p.add_argument('--max-sheets', type=int, default=5,
                   help='Hojas principales por modelo (12 imagenes/hoja), por defecto 5')
    p.add_argument('--all-sheets', action='store_true', help='Incluir todas las discrepancias en hojas')
    args = p.parse_args()
    if args.max_sheets <= 0:
        p.error('--max-sheets debe ser positivo')
    base = args.project / BASELINE_REL
    manifest_path = args.dataset_root / 'manifest.csv'
    if not base.is_dir():
        p.error(f'No existe el directorio de baselines: {base}')
    if not (args.dataset_root / 'images').is_dir():
        p.error(f'No existe la carpeta de imagenes: {args.dataset_root / "images"}')
    manifest = read_manifest(manifest_path)
    manifest_hash = sha256(manifest_path)
    if args.model:
        if Path(args.model).name != args.model or args.model in {'.', '..'}:
            p.error('--model debe ser el nombre de una carpeta, no una ruta')
        selected = [base / args.model]
    else:
        selected = sorted(d for d in base.iterdir()
                          if d.is_dir() and (d / 'predictions.csv').is_file())
    if not selected:
        p.error('No se encontraron modelos con predictions.csv')
    print('=== UC3M-LP: AUDITORIA VISUAL DE BASELINES (CPU) ===', flush=True)
    print(f'Manifest SHA256: {manifest_hash}', flush=True)
    print(f'Modelos encontrados: {len(selected)}', flush=True)
    issues = []
    for model_dir in selected:
        issues += analyze_one(model_dir, args.dataset_root, manifest, manifest_hash,
                              -1 if args.all_sheets else args.max_sheets)
    print('\nESTADO:', 'INFORMES GENERADOS; VALIDACION VISUAL PENDIENTE'
          if not issues else f'ADVERTENCIA: {len(issues)} imagenes ausentes o danadas', flush=True)
    return 1 if issues else 0


if __name__ == '__main__':
    raise SystemExit(main())
