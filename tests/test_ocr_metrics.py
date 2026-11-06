"""Pruebas unitarias de metricas para baselines con GT completo."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

# Permite ejecutar tests desde la raiz del repositorio, sin instalar paquete.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.ocr_metrics import (
    edit_distance,
    normalize_plate,
    score_complete,
    summarize_complete,
)


class TestOCRMetrics(unittest.TestCase):
    def test_normalization_matches_parseq(self):
        self.assertEqual(normalize_plate('  ab-1234 '), 'AB1234')
        self.assertEqual(normalize_plate('ioo8646'), 'IOO8646')

    def test_exact_match(self):
        s = score_complete('AB1234', 'ab1234')
        self.assertEqual(s['exact_match'], 1)
        self.assertEqual(s['edit_distance'], 0)

    def test_substitution(self):
        self.assertEqual(edit_distance('AB1234', 'AB1235'), 1)

    def test_insertion(self):
        self.assertEqual(edit_distance('AB1234X', 'AB1234'), 1)

    def test_deletion(self):
        self.assertEqual(edit_distance('AB123', 'AB1234'), 1)

    def test_empty_prediction(self):
        s = score_complete('AB1234', '')
        self.assertEqual(s['edit_distance'], 6)

    def test_micro_cer_and_em(self):
        s = summarize_complete([('AB1234', 'AB1234'), ('CD5678', 'CD567X')])
        self.assertEqual(s['exact_matches'], 1)
        self.assertEqual(s['levenshtein_errors'], 1)
        self.assertEqual(s['gt_characters'], 12)
        self.assertAlmostEqual(s['EM_percent'], 50.0)
        self.assertAlmostEqual(s['CER'], 1 / 12)

    def test_reject_masked_ground_truth(self):
        with self.assertRaisesRegex(ValueError, 'GT parcial'):
            score_complete('12*4*AB', '12345AB')

    def test_reject_empty_dataset(self):
        with self.assertRaises(ValueError):
            summarize_complete([])

    def test_reject_empty_ground_truth(self):
        with self.assertRaises(ValueError):
            score_complete('---', 'ABC')


if __name__ == '__main__':
    unittest.main()
