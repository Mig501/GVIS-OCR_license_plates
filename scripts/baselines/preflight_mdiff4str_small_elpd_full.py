#!/usr/bin/env python3
"""Preflight CPU de MDiff4STR-Small sobre el ELPD completo; sin inferencia."""
from __future__ import annotations

from run_mdiff4str_small_elpd_full_baseline import (
    load_rows_and_details,
    validate_assets,
    EXPECTED_FULL_MANIFEST_SHA256,
    EXPECTED_STRICT_MANIFEST_SHA256,
)


def main() -> None:
    print('=== GVIS: PREFLIGHT MDiff4STR-SMALL / ELPD COMPLETO ===', flush=True)
    rows, details = load_rows_and_details()
    assets = validate_assets()
    print(f'Dataset: ELPD completo, {len(rows)} imágenes, etiquetas candidatas', flush=True)
    print('Splits originales:', details['original_split_counts'], flush=True)
    print('Origen de GT:', details['gt_source_counts'], flush=True)
    print('GT auditados incorporados:', details['audited_rows_merged'], flush=True)
    print('GT corregidos por auditoría:', details['candidate_gt_changed_by_audit'], flush=True)
    print('SHA256 manifiesto completo:', EXPECTED_FULL_MANIFEST_SHA256, flush=True)
    print('SHA256 manifiesto estricto:', EXPECTED_STRICT_MANIFEST_SHA256, flush=True)
    print('OpenOCR commit:', assets['openocr_commit'], flush=True)
    print('Checkpoint SHA256:', assets['checkpoint_sha256'], flush=True)
    print('Config SHA256:', assets['config_sha256'], flush=True)
    print('Decoder: MDiffDecoder; sampler_step=3, sample_k=3, semi_autoregressive=True', flush=True)
    print('RESULTADO: PREFLIGHT MDiff4STR-SMALL CORRECTO; SIN INFERENCIA', flush=True)


if __name__ == '__main__':
    main()
