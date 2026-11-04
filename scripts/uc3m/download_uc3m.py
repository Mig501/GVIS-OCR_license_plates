#!/usr/bin/env python3
"""Descarga reanudable y verificación de UC3M-LP desde Zenodo (Legio2/SLURM).

Uso habitual: python scripts/uc3m/download_uc3m.py
No extrae el ZIP ni modifica otros datasets.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/17152029/files/UC3M-LP.zip?download=1"
EXPECTED_BYTES = 4_547_680_070
EXPECTED_MD5 = "496e3b5e66b78d231f71d85c0e5e11a3"
RAW_DIR = Path("/home/miguel.sanchez/datasets/ocr_matriculas/raw/UC3M-LP")
OUT_DIR = Path("/home/miguel.sanchez/ocr_matriculas/results/inspection/UC3M-LP")
ZIP_PATH = RAW_DIR / "UC3M-LP.zip"
PART_PATH = RAW_DIR / "UC3M-LP.zip.part"
CHUNK_BYTES = 4 * 1024 * 1024
PROGRESS_BYTES = 256 * 1024 * 1024
RETRIES = 5


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_once() -> None:
    offset = PART_PATH.stat().st_size if PART_PATH.exists() else 0
    if offset > EXPECTED_BYTES:
        raise ValueError(f"Archivo parcial más grande de lo previsto: {offset} bytes")
    if offset == EXPECTED_BYTES:
        print("Archivo parcial completo; pasando a la verificación.", flush=True)
        return

    headers = {"User-Agent": "GVIS-OCR-dataset-research/1.0", "Accept": "application/octet-stream, application/zip, */*"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Reanudando descarga desde el byte {offset:,}...", flush=True)
    else:
        print("Iniciando descarga desde Zenodo...", flush=True)

    request = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        ctype = response.headers.get("Content-Type", "").lower()
        if "text/html" in ctype or "application/json" in ctype:
            raise ValueError(f"Respuesta inesperada de Zenodo (HTTP {status}, {ctype}). No es el ZIP.")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range.strip())
            if not match or int(match.group(1)) != offset:
                raise ValueError(f"Respuesta Range incoherente: {content_range!r} (esperado desde {offset})")
            if match.group(3) != "*" and int(match.group(3)) != EXPECTED_BYTES:
                raise ValueError(f"Tamaño remoto inesperado: {content_range}")
            mode = "ab"
        elif status == 200:
            if offset:
                print("El servidor ignoró Range; reiniciando el archivo parcial sin mezclar datos.", flush=True)
            mode = "wb"
            offset = 0
            clen = response.headers.get("Content-Length")
            if clen and clen.isdigit() and int(clen) != EXPECTED_BYTES:
                raise ValueError(f"Content-Length no coincide: {clen}, esperado {EXPECTED_BYTES}")
        else:
            raise ValueError(f"Código HTTP no esperado: {status}")

        print(f"HTTP {status} | Content-Type: {ctype or 'no indicado'}", flush=True)
        total = offset
        last_report = (total // PROGRESS_BYTES) * PROGRESS_BYTES
        with PART_PATH.open(mode) as output:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                total += len(chunk)
                if total > EXPECTED_BYTES:
                    raise ValueError(f"Se recibieron más bytes de los esperados: {total}")
                if total - last_report >= PROGRESS_BYTES:
                    print(f"Descargados: {total / 1024**3:.2f} / {EXPECTED_BYTES / 1024**3:.2f} GiB", flush=True)
                    last_report = total
        print(f"Transferencia recibida hasta: {total:,} bytes", flush=True)
        if total != EXPECTED_BYTES:
            raise IOError(f"Descarga incompleta: {total:,}/{EXPECTED_BYTES:,} bytes. Se conserva .part para reanudar.")


def verify_zip(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(f"Tamaño incorrecto: {size:,} bytes, esperado {EXPECTED_BYTES:,}")
    print("Calculando MD5 del archivo completo...", flush=True)
    actual = md5_file(path)
    if actual != EXPECTED_MD5:
        raise ValueError(f"MD5 incorrecto: {actual}; esperado {EXPECTED_MD5}. No se extraerá ni renombrará el parcial.")
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        count = len(infos)
        unpacked = sum(info.file_size for info in infos)
    print(f"MD5 correcto: {actual}", flush=True)
    return count, unpacked


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== UC3M-LP: DESCARGA Y VERIFICACION ===", flush=True)
    print(f"Destino: {ZIP_PATH}", flush=True)
    print(f"Tamaño esperado: {EXPECTED_BYTES:,} bytes", flush=True)

    if ZIP_PATH.is_file():
        print("ZIP existente: comprobando antes de realizar otra descarga.", flush=True)
        count, unpacked = verify_zip(ZIP_PATH)
    else:
        remaining = EXPECTED_BYTES - (PART_PATH.stat().st_size if PART_PATH.exists() else 0)
        free = shutil.disk_usage(RAW_DIR).free
        if remaining + 512 * 1024 * 1024 > free:
            raise OSError(f"Espacio disponible insuficiente: {free:,} bytes; faltan {remaining:,} bytes más margen")

        for attempt in range(1, RETRIES + 1):
            try:
                print(f"Intento {attempt}/{RETRIES}", flush=True)
                download_once()
                break
            except (OSError, urllib.error.URLError) as exc:
                print(f"Error transitorio: {exc}", flush=True)
                if attempt == RETRIES:
                    raise
                time.sleep(min(10 * attempt, 40))
        count, unpacked = verify_zip(PART_PATH)
        os.replace(PART_PATH, ZIP_PATH)

    summary = "\n".join([
        "=== UC3M-LP: DESCARGA Y VERIFICACION ===",
        f"ZIP válido: {ZIP_PATH}",
        f"Tamaño ZIP: {ZIP_PATH.stat().st_size:,} bytes",
        f"MD5 (Zenodo): {EXPECTED_MD5} — correcto",
        f"Entradas en ZIP (sin directorios): {count}",
        f"Tamaño total descomprimido: {unpacked:,} bytes",
        "Extracción: NO realizada",
        "=== FINALIZADO ===",
    ])
    (OUT_DIR / "download_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"ERROR: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
