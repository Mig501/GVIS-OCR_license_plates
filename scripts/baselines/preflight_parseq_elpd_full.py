#!/usr/bin/env python3
"""Preflight del ELPD completo para PARSeq (solo CPU, sin inferencia ni escrituras)."""
from __future__ import annotations

from run_parseq_elpd_full_baseline import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_STRICT_MANIFEST,
    load_manifest,
    sha256_file,
)


def main() -> None:
    print('=== GVIS: PARSEQ ZERO-SHOT / ELPD COMPLETO ===', flush=True)
    rows, details = load_manifest(
        DEFAULT_MANIFEST,
        DEFAULT_IMAGE_ROOT,
        DEFAULT_STRICT_MANIFEST,
    )
    print(f'Dataset: ELPD completo, {len(rows)} imágenes, etiquetas candidatas', flush=True)
    print('Splits originales:', details['original_split_counts'], flush=True)
    print('Origen de GT:', details['gt_source_counts'], flush=True)
    print('GT auditados incorporados:', details['audited_rows_merged'], flush=True)
    print('Etiquetas cambiadas tras la auditoría:', details['candidate_gt_changed_by_audit'], flush=True)
    print('SHA256 manifiesto completo:', sha256_file(DEFAULT_MANIFEST), flush=True)
    print('SHA256 manifiesto estricto:', details['audited_manifest_sha256'], flush=True)
    print('RESULTADO: MANIFIESTOS VALIDADOS, SIN INFERENCIA', flush=True)


if __name__ == '__main__':
    main()
