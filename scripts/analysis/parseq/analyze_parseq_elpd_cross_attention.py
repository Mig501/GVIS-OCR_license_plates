#!/usr/bin/env python3
"""Paso 4 · PARSeq x ELPD V17: mapas de atencion cruzada EN LA ETAPA AR.

Captura la MHA visual del decoder oficial (cross_attn) sin modificar sus
operaciones ni los pesos del modelo. Reconstruye las atenciones de cada cabeza
con las proyecciones del propio modulo y comprueba que su media coincide con
la atencion media devuelta por PyTorch. Crea overlays, tablas y tensores.

Ejecucion habitual (task GVIS: Ejecutar en GPU, sin argumentos):
  python scripts/analysis/parseq/analyze_parseq_elpd_cross_attention.py

Preflight sin GPU:
  python scripts/analysis/parseq/analyze_parseq_elpd_cross_attention.py --prepare-only

NOTA: atencion != causalidad. La alineacion de secuencias repetidas es ambigua.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
from difflib import SequenceMatcher
from typing import Any

PROJECT = Path('/home/miguel.sanchez/ocr_matriculas')
DATASETS = Path('/home/miguel.sanchez/datasets/ocr_matriculas')
DEFAULT_PREVIOUS = (PROJECT / 'results/analysis/PARSeq/ELPD/insertion_stages_v1/'
                    'comparacion_AR_refinamiento.csv')
DEFAULT_IMAGE_ROOT = DATASETS / 'processed/ELPD_LPO'
DEFAULT_OUT = PROJECT / 'results/analysis/PARSeq/ELPD/cross_attention_v1'
DEFAULT_TORCH_HOME = DATASETS / 'models/torch_hub'

# Una insercion de cada comportamiento; todos aparecieron ya en AR en el paso 3.
PREFERRED = [
    ('inicial', 'images/train__ABA1234.JPG'),
    ('interna', 'images/test__DANKE82.png'),
    ('repeticion', 'images/test__5888881.png'),
    ('final', 'images/test__CN498JC.png'),
]


def normalize_plate(x: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(x).upper())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_cases(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f'No se encuentra el CSV del paso 3: {csv_path}')
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        expected_cols = {'image_path', 'ground_truth', 'AR_sin_refinamiento',
                         'AR_mas_1_refinamiento', 'coincide_con_prediccion_guardada_v17'}
        if not expected_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError('El CSV no contiene las columnas de comparacion del paso 3')
        indexed = {r['image_path']: r for r in reader}
    cases = []
    for n, (kind, image_rel) in enumerate(PREFERRED, 1):
        if image_rel not in indexed:
            raise ValueError(f'Falta caso seleccionado del paso 3: {image_rel}')
        r = indexed[image_rel]
        if r['coincide_con_prediccion_guardada_v17'] != '1':
            raise ValueError(f'Prediccion del caso no reproducida en paso 3: {image_rel}')
        gt = r['ground_truth'].strip()
        prediction = r['AR_sin_refinamiento'].strip()
        opcodes = SequenceMatcher(None, gt, prediction, autojunk=False).get_opcodes()
        edits = [o for o in opcodes if o[0] != 'equal']
        if len(edits) != 1 or edits[0][0] != 'insert':
            raise ValueError(f'No es insercion pura unica: GT={gt}, AR={prediction}, {edits}')
        _, _, _, first_pred, last_pred = edits[0]
        if first_pred == last_pred or first_pred >= len(prediction):
            raise ValueError('Posiciones de insercion invalidas')
        is_repeated = kind == 'repeticion'
        cases.append({
            'numero': n, 'tipo': kind, 'image_path': image_rel,
            'gt': gt, 'ar_esperado': prediction,
            'refinado_esperado': r['AR_mas_1_refinamiento'].strip(),
            'posiciones_insertadas_0based': list(range(first_pred, last_pred)),
            'alineacion_ambigua': is_repeated,
            'nota': ('En una secuencia de digitos repetidos la posicion atribuida a la '
                     'insercion es una de varias alineaciones igualmente validas.'
                     if is_repeated else ''),
        })
    return cases


def safe_image(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or not p.parts or p.parts[0] != 'images' or '..' in p.parts:
        raise ValueError(f'Ruta de imagen insegura: {relative}')
    result = (root / p).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera del dataset: {result}')
    if not result.is_file():
        raise FileNotFoundError(f'Falta imagen ELPD: {result}')
    return result


def choose_font(size: int):
    from PIL import ImageFont
    candidate = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        return ImageFont.truetype(candidate, size)
    except OSError:
        return ImageFont.load_default()


def colour_overlay(img, weights, grid_hw, *, size=(1024, 256)):
    """Mapa medio relativo sobre la imagen RGB REALMENTE usada por la red."""
    import numpy as np
    from PIL import Image, ImageOps

    rows, cols = grid_hw
    raw = np.asarray(weights, dtype=np.float64).reshape(rows, cols)
    delta = float(raw.max() - raw.min())
    stretched = np.zeros_like(raw) if delta < 1e-12 else (raw - raw.min()) / delta
    intensity = Image.fromarray(np.uint8(np.round(255.0 * stretched)), mode='L')
    intensity = intensity.resize(size, resample=Image.Resampling.BILINEAR)
    colored = ImageOps.colorize(intensity, black=(3, 10, 32), white=(255, 85, 20))
    original = img.resize(size, resample=Image.Resampling.BICUBIC)
    return Image.blend(original, colored, 0.48)


def make_case_sheet(case, img, grid_hw, weights_mean, selected_positions, out_path):
    from PIL import Image, ImageDraw
    width, image_height, header_height, gap = 1100, 250, 58, 22
    blocks = [(None, 'Imagen preprocesada (32 x 128, o dimensiones del checkpoint)')]
    for pos in selected_positions:
        tok = case['ar_esperado'][pos] if pos < len(case['ar_esperado']) else 'EOS'
        is_insert = pos in case['posiciones_insertadas_0based']
        prefix = 'INSERCION' if is_insert else ('EOS' if tok == 'EOS' else 'VECINO')
        blocks.append((pos, f'{prefix} · paso AR {pos + 1} · token {tok}'))
    total_h = 115 + len(blocks) * (image_height + header_height + gap)
    page = Image.new('RGB', (width, total_h), 'white')
    d = ImageDraw.Draw(page)
    title_font = choose_font(27)
    body_font = choose_font(21)
    d.text((28, 12), f"{case['numero']:02d} · {case['tipo'].upper()} | GT: {case['gt']}",
           font=title_font, fill=(15, 27, 45))
    d.text((28, 52), f"AR: {case['ar_esperado']} | insercion en indice(s) "
           f"{','.join(str(i + 1) for i in case['posiciones_insertadas_0based'])}",
           font=body_font, fill=(95, 25, 25))
    if case['alineacion_ambigua']:
        d.text((28, 87), 'AVISO: posicion de insercion ambigua por repeticion',
               font=choose_font(18), fill=(163, 73, 11))
    y = 115
    for pos, heading in blocks:
        d.text((28, y + 12), heading, font=body_font, fill=(18, 55, 76))
        y += header_height
        if pos is None:
            panel = img.resize((1024, image_height), resample=Image.Resampling.BICUBIC)
        else:
            panel = colour_overlay(img, weights_mean[pos], grid_hw, size=(1024, image_height))
        page.paste(panel, ((width - 1024) // 2, y))
        y += image_height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def make_heads_sheet(case, img, grid_hw, weight12, pos, out_path):
    from PIL import Image, ImageDraw
    tile_w, tile_h = 480, 120
    margin, header, gutter = 24, 120, 20
    page = Image.new('RGB', (margin * 2 + tile_w * 3 + gutter * 2,
                             header + 4 * (tile_h + 45) + 20), 'white')
    d = ImageDraw.Draw(page)
    d.text((margin, 12), f"Atencion cruzada · 12 cabezas · {case['numero']:02d} {case['tipo']}",
           fill=(20, 35, 55), font=choose_font(25))
    d.text((margin, 55), f"GT={case['gt']} | AR={case['ar_esperado']} | "
           f"posicion AR={pos + 1} ({case['ar_esperado'][pos]})",
           fill=(40, 40, 40), font=choose_font(20))
    for h in range(12):
        row, col = divmod(h, 3)
        x = margin + col * (tile_w + gutter)
        y = header + row * (tile_h + 45)
        d.text((x, y), f'Cabeza {h + 1:02d}', font=choose_font(19), fill=(20, 60, 90))
        view = colour_overlay(img, weight12[h], grid_hw, size=(tile_w, tile_h))
        page.paste(view, (x, y + 29))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def write_csv(dest: Path, rows: list[dict[str, Any]], columns: list[str]):
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class AttnCollector:
    """Hook pasivo: no cambia ni el input ni el output de la MHA del modelo."""
    captures: list

    def __call__(self, layer, inputs, output):
        import torch
        import torch.nn.functional as F

        if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
            raise RuntimeError('PARSeq no devolvio los pesos de cross_attn (need_weights=True)')
        if len(inputs) < 3:
            raise RuntimeError('cross_attn no recibio query, key, value')
        q, k, _ = inputs[:3]
        if q.ndim != 3 or k.ndim != 3 or q.shape[0] != 1 or k.shape[0] != 1:
            raise RuntimeError(f'Forma inesperada de MHA: q={q.shape}, k={k.shape}')
        if not layer.batch_first or layer.in_proj_weight is None:
            raise RuntimeError('MHA diferente de la implementacion oficial estudiada')
        embed = layer.embed_dim
        heads = layer.num_heads
        d = embed // heads
        wq, wk, _ = layer.in_proj_weight.chunk(3, dim=0)
        bq, bk = None, None
        if layer.in_proj_bias is not None:
            bq, bk, _ = layer.in_proj_bias.chunk(3, dim=0)
        q_proj = F.linear(q, wq, bq).reshape(1, q.shape[1], heads, d).transpose(1, 2)
        k_proj = F.linear(k, wk, bk).reshape(1, k.shape[1], heads, d).transpose(1, 2)
        attn_heads = torch.softmax((q_proj @ k_proj.transpose(-2, -1)) / math.sqrt(d), dim=-1)
        mean_computed = attn_heads.mean(dim=1)
        mean_official = output[1]
        if mean_official.ndim != 3 or mean_official.shape != mean_computed.shape:
            raise RuntimeError(f'Forma atencion no coincide: {mean_official.shape} != {mean_computed.shape}')
        deviation = float((mean_computed - mean_official).abs().max().item())
        if deviation > 5e-5:
            raise RuntimeError(f'Atencion por cabeza no coincide con media oficial; delta={deviation}')
        # batch=1; query_len=1 en AR. No guardar referencias GPU tras la llamada.
        if q.shape[1] != 1:
            raise RuntimeError(f'Se esperaba una sola query por paso AR; recibidas {q.shape[1]}')
        self.captures.append({
            'heads': attn_heads[0, :, 0, :].detach().float().cpu().numpy(),
            'avg': mean_official[0, 0, :].detach().float().cpu().numpy(),
            'verification_delta': deviation,
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--comparison', type=Path, default=DEFAULT_PREVIOUS)
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()

    cases = read_cases(args.comparison)
    print('=== PASO 4 · PARSEQ / ELPD V17 · ATENCION CRUZADA EN AR ===', flush=True)
    print('CSV del paso 3:', args.comparison, flush=True)
    for c in cases:
        print(f"  {c['numero']:02d} {c['tipo']:11s} GT={c['gt']:8s} "
              f"AR={c['ar_esperado']:10s} "
              f"insertados={[p + 1 for p in c['posiciones_insertadas_0based']]}", flush=True)
    if args.prepare_only:
        print('RESULTADO: PREPARACION VALIDA; NO SE HA EJECUTADO INFERENCIA', flush=True)
        return

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'La carpeta de salida ya tiene archivos, no se sobrescribe: {args.out}')

    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms as T

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible; ejecutar en un nodo con GPU')
    os.environ.setdefault('TORCH_HOME', str(DEFAULT_TORCH_HOME))
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = torch.hub.load('baudm/parseq:main', 'parseq', pretrained=True, trust_repo=True).eval().cuda()
    core = model.model
    if not core.decode_ar or int(core.refine_iters) != 1:
        raise RuntimeError('El checkpoint no tiene la configuracion AR + 1 refinamiento de la baseline')
    if len(core.decoder.layers) != 1:
        raise RuntimeError('Se esperaba 1 capa de decoder PARSeq')
    cross_attn = core.decoder.layers[0].cross_attn
    grid_hw = tuple(int(n) for n in core.encoder.patch_embed.grid_size)
    n_patches = int(core.encoder.patch_embed.num_patches)
    if grid_hw[0] * grid_hw[1] != n_patches:
        raise RuntimeError('Grid size no coincide con el numero de parches del encoder')
    image_size = tuple(int(i) for i in model.hparams.img_size)
    transform = T.Compose([
        T.Resize(image_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ])
    args.out.mkdir(parents=True)
    (args.out / 'comparativas').mkdir()
    (args.out / 'cabezas').mkdir()
    (args.out / 'tensores').mkdir()
    records, top_rows = [], []
    validation_deltas = []
    model_path = Path(inspect.getfile(core.__class__))
    initial_refine = int(core.refine_iters)
    collector = AttnCollector([])
    handle = cross_attn.register_forward_hook(collector)
    try:
        core.refine_iters = 0  # SOLO la ejecucion AR; restaurado en finally.
        with torch.inference_mode():
            for case in cases:
                path = safe_image(args.image_root, case['image_path'])
                with Image.open(path) as orig:
                    img = orig.convert('RGB')
                x = transform(img).unsqueeze(0).cuda()
                collector.captures.clear()
                logits = model(x)
                probs = logits.softmax(-1)
                pred, _ = model.tokenizer.decode(probs)
                decoded = normalize_plate(pred[0])
                if decoded != case['ar_esperado']:
                    raise RuntimeError(f"PREDICCION AR NO REPRODUCIDA {case['image_path']}: "
                                       f"esperada={case['ar_esperado']} obtenida={decoded}")
                if len(collector.captures) != logits.shape[1]:
                    raise RuntimeError(f'Diferencia etapas/atenciones: {len(collector.captures)} '
                                       f'vs {logits.shape[1]}')
                heads = np.stack([t['heads'] for t in collector.captures], axis=0)
                means = np.stack([t['avg'] for t in collector.captures], axis=0)
                if heads.shape[1:] != (cross_attn.num_heads, n_patches):
                    raise RuntimeError(f'Dimensiones de atencion inesperadas: {heads.shape}')
                validation_deltas.extend(t['verification_delta'] for t in collector.captures)
                insert_pos = case['posiciones_insertadas_0based']
                selected_pos = sorted({p for ins in insert_pos for p in (ins - 1, ins, ins + 1)
                                       if 0 <= p < means.shape[0]})
                if not all(p < len(decoded) for p in insert_pos):
                    raise RuntimeError('Insercion fuera de la prediccion decodificada')
                tag = (f"{case['numero']:02d}_{case['tipo']}_" + case['gt']).replace('/', '_')
                np.savez_compressed(args.out / 'tensores' / f'{tag}.npz',
                                    heads=heads, mean=means,
                                    grid_hw=np.array(grid_hw, dtype=np.int16))
                model_resized = img.resize((image_size[1], image_size[0]),
                                           resample=Image.Resampling.BICUBIC)
                make_case_sheet(case, model_resized, grid_hw, means, selected_pos,
                                args.out / 'comparativas' / f'{tag}.png')
                for p in insert_pos:
                    make_heads_sheet(case, model_resized, grid_hw, heads[p], p,
                                     args.out / 'cabezas' / f'{tag}_pos{p+1:02d}.png')
                for p in selected_pos:
                    token = decoded[p] if p < len(decoded) else 'EOS'
                    type_pos = ('insertado' if p in insert_pos else
                                'eos' if p == len(decoded) else 'vecino')
                    arr = means[p].reshape(grid_hw)
                    top = np.argsort(means[p])[::-1][:5]
                    top_list = []
                    for rank, patch in enumerate(top, 1):
                        r, col = divmod(int(patch), grid_hw[1])
                        top_list.append(f'({r},{col}):{float(means[p,patch]):.5f}')
                        top_rows.append({
                            'imagen': case['image_path'], 'tipo_error': case['tipo'],
                            'posicion_AR_1based': p + 1, 'token': token,
                            'rol': type_pos, 'ranking': rank,
                            'fila_patch_0based': r, 'col_patch_0based': col,
                            'peso_medio': round(float(means[p,patch]), 8),
                        })
                    records.append({
                        'imagen': case['image_path'], 'tipo_error': case['tipo'],
                        'gt': case['gt'], 'prediccion_AR': decoded,
                        'posicion_AR_1based': p + 1, 'token': token,
                        'rol': type_pos, 'max_peso': round(float(arr.max()), 8),
                        'entropia_patches': round(float(-np.sum(arr * np.log(arr + 1e-12))), 6),
                        'top5_patches': ' | '.join(top_list),
                        'alineacion_ambigua': int(case['alineacion_ambigua']),
                    })
                print(f"[OK {case['numero']:02d}/04] {case['tipo']}: "
                      f"{case['gt']} -> {decoded} | "
                      f"pasos AR={len(collector.captures)} | "
                      f"patches={n_patches} | heads={cross_attn.num_heads}", flush=True)
    finally:
        handle.remove()
        core.refine_iters = initial_refine

    write_csv(args.out / 'atencion_por_caracter.csv', records,
              ['imagen', 'tipo_error', 'gt', 'prediccion_AR', 'posicion_AR_1based',
               'token', 'rol', 'max_peso', 'entropia_patches', 'top5_patches',
               'alineacion_ambigua'])
    write_csv(args.out / 'top5_patches.csv', top_rows,
              ['imagen', 'tipo_error', 'posicion_AR_1based', 'token', 'rol',
               'ranking', 'fila_patch_0based', 'col_patch_0based', 'peso_medio'])
    summary = {
        'paso': '4', 'modelo': 'PARSeq oficial, pretrained',
        'dataset': 'ELPD GT V17', 'etapa_observada': 'AR (sin refinamiento)',
        'casos': cases, 'casos_reproducidos': len(cases),
        'encoder_patches': n_patches, 'patch_grid_filas_columnas': list(grid_hw),
        'num_cabezas_decoder_cross_attn': int(cross_attn.num_heads),
        'image_size_H_W': list(image_size),
        'torch_version': torch.__version__,
        'sha256_comparacion_paso3': sha256(args.comparison),
        'sha256_implementacion_parseq': sha256(model_path),
        'max_desviacion_reconstruccion_de_atencion': max(validation_deltas),
        'checkpoint_y_pesos_modificados': False,
        'nota': ('Los mapas reflejan pesos de atencion, no atribucion causal. '
                 'La posicion de insercion en secuencias de caracteres repetidos '
                 'es ambigua. Para demostrar causalidad se requiere una intervencion '
                 'controlada sobre la imagen y replicacion de predicciones.'),
    }
    (args.out / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Reproducidos: 4/4 | pesos intactos | desviacion maxima:',
          summary['max_desviacion_reconstruccion_de_atencion'], flush=True)
    print('RESULTADO: MAPAS DE ATENCION CRUZADA GENERADOS', flush=True)
    print('SALIDA:', args.out, flush=True)


if __name__ == '__main__':
    main()
