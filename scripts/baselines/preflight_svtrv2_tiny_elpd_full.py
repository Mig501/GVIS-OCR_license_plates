#!/usr/bin/env python3
"""Validación CPU de ELPD completo y recursos fijados de SVTRv2-Tiny."""
from __future__ import annotations

from run_parseq_elpd_full_baseline import (
    DEFAULT_IMAGE_ROOT, DEFAULT_MANIFEST, DEFAULT_STRICT_MANIFEST,
    load_manifest, sha256_file,
)
from run_svtrv2_tiny_elpd_full_baseline import validate_assets


def main() -> None:
    print('=== GVIS: PREFLIGHT SVTRv2-TINY / ELPD COMPLETO ===', flush=True)
    rows, details = load_manifest(DEFAULT_MANIFEST, DEFAULT_IMAGE_ROOT, DEFAULT_STRICT_MANIFEST)
    assets = validate_assets()
    print(f'Dataset: ELPD completo, {len(rows)} imágenes, etiquetas candidatas', flush=True)
    print('Splits originales:', details['original_split_counts'], flush=True)
    print('Origen de GT:', details['gt_source_counts'], flush=True)
    print('GT auditados incorporados:', details['audited_rows_merged'], flush=True)
    print('GT corregidos por auditoría:', details['candidate_gt_changed_by_audit'], flush=True)
    print('SHA256 manifiesto completo:', sha256_file(DEFAULT_MANIFEST), flush=True)
    print('SHA256 manifiesto estricto:', details['audited_manifest_sha256'], flush=True)
    print('OpenOCR commit:', assets['openocr_commit'], flush=True)
    print('Checkpoint SHA256:', assets['checkpoint_sha256'], flush=True)
    print('Config SHA256:', assets['config_sha256'], flush=True)
    print('RESULTADO: PREFLIGHT SVTRv2-TINY CORRECTO; SIN INFERENCIA', flush=True)


if __name__ == '__main__':
    main()
