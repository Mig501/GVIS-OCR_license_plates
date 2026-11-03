#!/usr/bin/env python3
"""Descarga y verifica LRLPR-26 en Legio2 sin extraer archivos.

Lee el enlace autorizado desde scripts/lrlpr/.download_url (NO subir a Git).
Se ejecuta en un nodo de computo mediante SLURM.
"""
from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zipfile

DATA_DIR = Path('/home/miguel.sanchez/datasets/ocr_matriculas/raw/LRLPR-26')
DEFAULT_URL_FILE = Path(__file__).resolve().with_name('.download_url')
ZIP_NAME = 'LRLPR-26-5opEvJTW.zip'
CHUNK = 4 * 1024 * 1024
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; GVIS-LRLPR26-research/1.0)'}


class DropboxLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != 'a':
            return
        for key, value in attrs:
            if key == 'href' and value and 'dropbox.com/' in value:
                self.links.append(html.unescape(value))


def direct_dropbox(url: str) -> str:
    parts = urlsplit(url)
    if 'dropbox.com' not in (parts.hostname or ''):
        return url
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True) if key not in ('dl', 'raw')]
    query.append(('dl', '1'))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def resolve_source(url: str) -> str:
    """Resuelve la pagina oficial de UFPR, que suele contener un enlace Dropbox."""
    host = (urlsplit(url).hostname or '').lower()
    if 'dropbox.com' in host:
        return direct_dropbox(url)
    if host != 'www.inf.ufpr.br':
        raise ValueError('El enlace debe ser el proporcionado por UFPR o uno de Dropbox.')
    with urlopen(Request(url, headers=HEADERS), timeout=60) as response:
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            # En caso de redireccion HTTP directa a un fichero.
            return direct_dropbox(response.geturl())
        body = response.read(1_000_000).decode('utf-8', errors='replace')
    parser = DropboxLinks()
    parser.feed(body)
    if not parser.links:
        match = re.search(r'https?://(?:www\.)?dropbox\.com/[^\s"\'<>]+', body)
        if match:
            parser.links.append(html.unescape(match.group(0)))
    if not parser.links:
        raise RuntimeError('No se ha encontrado enlace Dropbox en la pagina oficial. Revisar el acceso.')
    return direct_dropbox(parser.links[0])


def verify_zip(path: Path) -> tuple[int, int]:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f'El archivo NO es un ZIP valido: {path}')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        invalid = archive.testzip()
        if invalid:
            raise RuntimeError(f'Archivo ZIP dañado en la entrada: {invalid}')
        return len(members), sum(item.file_size for item in members)


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + '.part')
    existing = partial.stat().st_size if partial.exists() else 0
    if existing:
        with partial.open('rb') as file:
            if not file.read(4).startswith(b'PK'):
                raise RuntimeError(f'Descarga parcial no parece ZIP: {partial}. Revisarla antes de continuar.')

    headers = dict(HEADERS)
    if existing:
        headers['Range'] = f'bytes={existing}-'
    print(f'Descarga en nodo SLURM: {target}', flush=True)
    if existing:
        print(f'Reanudando desde {existing:,} bytes', flush=True)

    with urlopen(Request(url, headers=headers), timeout=120) as response:
        status = response.status
        content_type = response.headers.get('Content-Type', 'desconocido')
        print(f'HTTP {status}; Content-Type: {content_type}', flush=True)
        # Si el servidor ignora Range (200), sustituimos el parcial, sin concatenar datos repetidos.
        resume = existing > 0 and status == 206
        if existing and status == 206:
            content_range = response.headers.get('Content-Range', '')
            if not content_range.startswith(f'bytes {existing}-'):
                raise RuntimeError(f'Content-Range inesperado: {content_range}')
        if existing and not resume:
            print('Servidor no permite reanudar; se reinicia la descarga.', flush=True)

        start = existing if resume else 0
        with partial.open('ab' if resume else 'wb') as output:
            if not resume:
                first = response.read(16)
                if not first.startswith((b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08')):
                    preview = first + response.read(96)
                    output.close()
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(
                        'La respuesta HTTP no contiene un ZIP (cabeceras pueden anunciar JSON). '
                        f'Primeros bytes: {preview[:112]!r}'
                    )
                output.write(first)
                start += len(first)
            last_report = start
            downloaded = start
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 64 * 1024 * 1024:
                    print(f'Recibidos {downloaded / (1024**2):.1f} MiB', flush=True)
                    last_report = downloaded

    print(f'Descarga finalizada: {partial.stat().st_size:,} bytes', flush=True)
    print('Comprobando estructura y CRC de todas las entradas...', flush=True)
    count, raw_bytes = verify_zip(partial)
    partial.replace(target)
    print('=== LRLPR-26 - DESCARGA Y VERIFICACION ===')
    print(f'ZIP valido: {target}')
    print(f'Tamano ZIP: {target.stat().st_size:,} bytes')
    print(f'Entradas en ZIP: {count}')
    print(f'Tamano total descomprimido: {raw_bytes:,} bytes')
    print('Integridad ZIP (CRC): correcta')
    print('Extraccion: NO realizada')
    print('=== FINALIZADO ===')


def main() -> None:
    parser = argparse.ArgumentParser(description='Descarga LRLPR-26 y comprueba ZIP sin extraerlo.')
    parser.add_argument('--url-file', type=Path, default=DEFAULT_URL_FILE)
    parser.add_argument('--dest', type=Path, default=DATA_DIR)
    args = parser.parse_args()
    if not args.url_file.is_file():
        raise FileNotFoundError(f'Falta archivo privado con URL autorizada: {args.url_file}')
    url = args.url_file.read_text(encoding='utf-8').strip()
    if not url.startswith('https://'):
        raise ValueError('El archivo con URL debe contener un enlace HTTPS valido en una sola linea.')
    args.dest.mkdir(parents=True, exist_ok=True)
    target = args.dest / ZIP_NAME
    if target.exists():
        print('El ZIP ya existe; comprobando integridad sin volver a descargar...', flush=True)
        count, size = verify_zip(target)
        print(f'ZIP valido: {target}\nEntradas: {count}\nTamano descomprimido: {size:,} bytes\n=== FINALIZADO ===')
        return
    resolved = resolve_source(url)
    download(resolved, target)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, HTTPError, URLError, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
