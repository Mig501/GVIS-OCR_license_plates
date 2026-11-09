"""Pruebas unitarias del protocolo de GT parcial UC3M-LP."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.partial_metrics import (
    normalize_prediction, validate_partial_gt, masked_edit_distance,
    score_partial, summarize_partial,
)

class TestPartialMetrics(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_prediction(' 12-3 ab '), '123AB')

    def test_mask_validate(self):
        self.assertEqual(validate_partial_gt('2*28*GG', '1011011'), (7, 5))

    def test_reject_mask_bad(self):
        with self.assertRaises(ValueError):
            validate_partial_gt('2*28*GG', '1111011')

    def test_reject_no_mask(self):
        with self.assertRaises(ValueError):
            validate_partial_gt('AB12345', '1111111')

    def test_reject_all_masked(self):
        with self.assertRaises(ValueError):
            validate_partial_gt('*******', '0000000')

    def test_mask_any_character(self):
        a = score_partial('2*28*GG', '1011011', '2A28BGG')
        self.assertEqual(a['partial_exact_match'], 1)
        self.assertEqual(a['masked_edit_distance'], 0)

    def test_wrong_visible_char(self):
        a = score_partial('2*28*GG', '1011011', '2A28BGH')
        self.assertEqual(a['partial_exact_match'], 0)
        self.assertEqual(a['known_errors_if_aligned'], 1)
        self.assertEqual(a['masked_edit_distance'], 1)

    def test_missing_char_length(self):
        a = score_partial('2*28*GG', '1011011', '228BGG')
        self.assertEqual(a['partial_exact_match'], 0)
        self.assertEqual(a['length_match'], 0)
        self.assertIsNone(a['known_errors_if_aligned'])
        self.assertEqual(a['masked_edit_distance'], 1)

    def test_extra_char_length(self):
        a = score_partial('2*28*GG', '1011011', '2A28BGGX')
        self.assertEqual(a['partial_exact_match'], 0)
        self.assertEqual(a['masked_edit_distance'], 1)

    def test_aggregate(self):
        r = summarize_partial([
            ('2*28*GG', '1011011', '2A28BGG'),
            ('2*28*GG', '1011011', '2A28BGH'),
        ])
        self.assertEqual(r['dataset_size'], 2)
        self.assertEqual(r['partial_exact_matches'], 1)
        self.assertEqual(r['known_gt_characters'], 10)
        self.assertEqual(r['masked_edit_errors'], 1)
        self.assertAlmostEqual(r['partial_EM'], .5)
        self.assertAlmostEqual(r['masked_error_rate_proxy'], .1)

    def test_empty(self):
        with self.assertRaises(ValueError):
            summarize_partial([])

if __name__ == '__main__':
    unittest.main()
