"""Metricas reproducibles para OCR con caracteres anonimizados mediante '*'.

UC3M-LP aporta un texto de longitud conocida donde '*' denota una posicion
anonimizada. No existe GT para ese caracter: no se puede calcular CER/EM completo.

Definiciones:
  - partial_exact_match: longitud igual y todos los caracteres CONOCIDOS iguales
    en sus posiciones. Es un EM parcial estricto, no un EM de matricula completa.
  - masked_edit_distance: Levenshtein con sustitucion en '*' de coste 0;
    inserciones/eliminaciones cuestan 1 (incluida eliminacion de '*').
    Es una distancia optimista frente a un GT completo desconocido.
  - masked_error_rate: suma masked_edit_distance / suma de posiciones conocidas.
    Es una PROXY, no el CER convencional y puede superar el 100 %.
  - aligned_known_error_rate: errores entre posiciones conocidas exclusivamente
    en muestras de igual longitud; informar siempre su cobertura.

Nunca se reemplazan los caracteres anonimizados por una prediccion supuesta.
"""
from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

_CLEAN_PRED = re.compile(r'[^A-Z0-9]')
_VALID_GT = re.compile(r'[A-Z0-9*]+')


def normalize_prediction(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError('La prediccion debe ser una cadena')
    return _CLEAN_PRED.sub('', text.upper())


def validate_partial_gt(masked_gt: str, known_mask: str) -> tuple[int, int]:
    """Devuelve (longitud_total, caracteres_conocidos)."""
    if not isinstance(masked_gt, str) or not isinstance(known_mask, str):
        raise TypeError('GT y mascara deben ser cadenas')
    if not masked_gt or _VALID_GT.fullmatch(masked_gt) is None or '*' not in masked_gt:
        raise ValueError('GT parcial no valido; se espera A-Z/0-9 y al menos un *')
    expected_mask = ''.join('0' if c == '*' else '1' for c in masked_gt)
    if known_mask != expected_mask:
        raise ValueError('known_mask no corresponde a las posiciones del GT')
    known = known_mask.count('1')
    if not known:
        raise ValueError('GT sin ningun caracter verificable')
    return len(masked_gt), known


def masked_edit_distance(masked_gt: str, prediction_normalized: str) -> int:
    """Levenshtein con wildcard '*': sustitucion gratis, I/D con coste 1."""
    prev = list(range(len(prediction_normalized) + 1))
    for i, gt_char in enumerate(masked_gt, start=1):
        curr = [i]
        for j, pred_char in enumerate(prediction_normalized, start=1):
            substitute = 0 if gt_char == '*' or gt_char == pred_char else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + substitute))
        prev = curr
    return prev[-1]


def score_partial(masked_gt: str, known_mask: str, prediction: str) -> dict[str, Any]:
    length, known = validate_partial_gt(masked_gt, known_mask)
    pred = normalize_prediction(prediction)
    equal_length = len(pred) == length
    known_errors_aligned = (
        sum(1 for i, flag in enumerate(known_mask) if flag == '1' and masked_gt[i] != pred[i])
        if equal_length else None
    )
    strict_em = int(equal_length and known_errors_aligned == 0)
    dist = masked_edit_distance(masked_gt, pred)
    return {
        'ground_truth_masked': masked_gt,
        'known_mask': known_mask,
        'prediction_normalized': pred,
        'gt_length': length,
        'pred_length': len(pred),
        'known_characters': known,
        'masked_characters': length - known,
        'length_match': int(equal_length),
        'partial_exact_match': strict_em,
        'known_errors_if_aligned': known_errors_aligned,
        'masked_edit_distance': dist,
    }


def summarize_partial(pairs: Iterable[tuple[str, str, str]]) -> dict[str, int | float | None]:
    n = correct = edit_errors = known_total = length_match = 0
    known_err_aligned = known_total_aligned = 0
    for gt, mask, pred in pairs:
        r = score_partial(gt, mask, pred)
        n += 1
        correct += r['partial_exact_match']
        edit_errors += r['masked_edit_distance']
        known_total += r['known_characters']
        length_match += r['length_match']
        if r['length_match']:
            known_err_aligned += r['known_errors_if_aligned']
            known_total_aligned += r['known_characters']
    if not n:
        raise ValueError('No se pueden evaluar cero imagenes')
    return {
        'dataset_size': n,
        'partial_exact_matches': correct,
        'partial_EM': correct / n,
        'partial_EM_percent': 100 * correct / n,
        'masked_edit_errors': edit_errors,
        'known_gt_characters': known_total,
        'masked_error_rate_proxy': edit_errors / known_total,
        'masked_error_rate_proxy_percent': 100 * edit_errors / known_total,
        'equal_length_predictions': length_match,
        'equal_length_fraction': length_match / n,
        'aligned_known_errors': known_err_aligned,
        'aligned_known_characters': known_total_aligned,
        'aligned_known_error_rate': (
            known_err_aligned / known_total_aligned if known_total_aligned else None
        ),
    }
