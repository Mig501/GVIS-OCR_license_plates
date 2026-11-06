"""Metricas comunes para el reconocimiento OCR de matriculas con GT completo.

No aplica a datasets con caracteres ocultos (p. ej. UC3M-LP): esas
etiquetas necesitan un protocolo de evaluacion parcial independiente.

No modifica datasets ni realiza inferencia; solo utiliza Python estandar.
"""
from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

_ASCII_PLATE = re.compile(r"[^A-Z0-9]")


def normalize_plate(text: str) -> str:
    """Mayusculas y caracteres ASCII A-Z/0-9, como en el baseline PARSeq.

    ATENCION: no utilizar para GT con '*' de enmascaramiento, dado que
    eliminaria las posiciones ocultas.
    """
    if not isinstance(text, str):
        raise TypeError("El texto OCR debe ser una cadena")
    return _ASCII_PLATE.sub("", text.upper())


def edit_distance(a: str, b: str) -> int:
    """Distancia de Levenshtein, coste unitario para I/D/S."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def score_complete(ground_truth: str, prediction: str) -> dict[str, Any]:
    """Calcula EM y distancia de edicion para una matricula con GT completo.

    No emplea autocorreccion, reglas de formato de pais ni heuristicas.
    """
    if '*' in ground_truth:
        raise ValueError("GT parcial detectado: utilizar evaluacion con mascara, no EM/CER completos")
    gt = normalize_plate(ground_truth)
    pred = normalize_plate(prediction)
    if not gt:
        raise ValueError("El ground truth normalizado no puede estar vacio")
    dist = edit_distance(pred, gt)
    return {
        'ground_truth': gt,
        'prediction_normalized': pred,
        'exact_match': int(pred == gt),
        'edit_distance': dist,
        'gt_length': len(gt),
        'pred_length': len(pred),
    }


def summarize_complete(pairs: Iterable[tuple[str, str]]) -> dict[str, int | float]:
    """EM medio por imagen y CER micro: suma errores / suma caracteres GT.

    `pairs` contiene tuplas (ground_truth, prediction). Mantiene las
    definiciones del script run_parseq_elpd_baseline.py.
    """
    n = correct = errors = characters = 0
    for ground_truth, prediction in pairs:
        record = score_complete(ground_truth, prediction)
        n += 1
        correct += record['exact_match']
        errors += record['edit_distance']
        characters += record['gt_length']
    if not n:
        raise ValueError("No se pueden calcular metricas sobre un conjunto vacio")
    return {
        'dataset_size': n,
        'exact_matches': correct,
        'levenshtein_errors': errors,
        'gt_characters': characters,
        'EM': correct / n,
        'EM_percent': 100 * correct / n,
        'CER': errors / characters,
        'CER_percent': 100 * errors / characters,
    }
