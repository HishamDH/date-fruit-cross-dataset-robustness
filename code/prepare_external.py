"""Turn an external dataset of phone photographs into the two working copies used in the tests.

Usage:
    python prepare_external.py soda_hf       # ../_data/external/soda (Hugging Face copy, partial)
    python prepare_external.py soda          # ../_data/external/soda_kaggle (full set, zip)
    python prepare_external.py medina27      # ../_data/external/medina27 (zip)

For every original photograph (background-removed PNG copies are skipped) it writes
    ../data_ext/<name>/frame/<class>/<id>.jpg   central square of the whole photo, 256 x 256 (as for the main dataset)
    ../data_ext/<name>/fruit/<class>/<id>.jpg   square crop centred on the fruit, fruit = 29% of the frame, 256 x 256
    ../data_ext/<name>/plain/<class>/<id>.jpg   the same crop with the background replaced by the table colour of the main dataset
    ../data_ext/<name>/fruit_mask/<class>/<id>.png
and one line per image in ../data_ext/<name>/manifest.jsonl with the shape statistics of the located object
(area fraction in the photo, elongation), which are used later to separate whole fruits from pits.
The run can be interrupted and restarted; finished images are skipped.
"""
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps

import fruit_crop as FC

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = os.path.dirname(FC.HERE)
EXT = os.path.join(ROOT, '_data', 'external')
RAW_EXT = ('.jpg', '.jpeg', '.heic', '.bmp', '.webp')


def english(name):
    s = re.sub(r'[^A-Za-z0-9_ ]', '', name).strip()
    return re.sub(r'\s+', '_', s)


def soda_dir_items(root):
    """Folder per class; original photos are JPG, the PNG files are background-removed copies."""
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p) or d.startswith('.') or d == 'TabularDataset':
            continue
        for f in sorted(os.listdir(p)):
            if f.lower().endswith(RAW_EXT):
                yield english(d), os.path.splitext(f)[0], (lambda q=os.path.join(p, f): open(q, 'rb').read())


def zip_items(zpath, class_of, keep):
    """Generic reader for a zip archive. class_of(name) -> class or None; keep(name) -> bool."""
    z = zipfile.ZipFile(zpath)
    for n in sorted(z.namelist()):
        if n.endswith('/') or not keep(n):
            continue
        c = class_of(n)
        if c:
            yield c, os.path.splitext(os.path.basename(n))[0], (lambda q=n: z.read(q))


def sources(name):
    if name == 'soda_hf':
        return soda_dir_items(os.path.join(EXT, 'soda'))
    if name == 'soda':
        zp = os.path.join(EXT, 'soda_kaggle', 'soda_kaggle.zip')
        return zip_items(zp, lambda n: english(n.split('/')[-2]) if '/' in n else None,
                         lambda n: n.lower().endswith(RAW_EXT) and 'tabular' not in n.lower() and '__MACOSX' not in n)
    if name == 'medina27':
        zp = os.path.join(EXT, 'medina27', 'medina27.zip')
        cfg = json.load(open(os.path.join(FC.HERE, 'medina27_layout.json')))
        pat = re.compile(cfg['pattern'])

        def class_of(n):
            m = pat.search(n)
            return m.group(1) if m else None
        return zip_items(zp, class_of, lambda n: n.lower().endswith(tuple(cfg['extensions'])))
    raise SystemExit('unknown dataset ' + name)


def main():
    name = sys.argv[1]
    out = os.path.join(ROOT, 'data_ext', name)
    os.makedirs(out, exist_ok=True)
    man = os.path.join(out, 'manifest.jsonl')
    done = set()
    if os.path.exists(man):
        for ln in open(man, encoding='utf-8'):
            r = json.loads(ln)
            done.add((r['label'], r['id']))
    t0, n = time.time(), 0
    with open(man, 'a', encoding='utf-8') as fh:
        for label, sid, read in sources(name):
            sid = re.sub(r'[^A-Za-z0-9_.-]', '_', sid)
            if (label, sid) in done:
                continue
            try:
                raw = read()
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
                if img.mode in ('RGBA', 'LA', 'P'):
                    bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
                    img = Image.alpha_composite(bg, img.convert('RGBA'))
                img = img.convert('RGB')
                mask = FC.fruit_mask(img)
                crop, cm, st = FC.fruit_crop(img, mask)
                frame = FC.frame_crop(img)
            except Exception as e:  # noqa: BLE001
                print('SKIP', label, sid, repr(e)[:100], flush=True)
                continue
            for kind, arr in (('frame', frame), ('fruit', crop), ('plain', FC.on_plain_background(crop, cm))):
                d = os.path.join(out, kind, label)
                os.makedirs(d, exist_ok=True)
                Image.fromarray(arr).save(os.path.join(d, sid + '.jpg'), quality=95, subsampling=0)
            d = os.path.join(out, 'fruit_mask', label)
            os.makedirs(d, exist_ok=True)
            Image.fromarray((cm * 255).astype(np.uint8)).save(os.path.join(d, sid + '.png'))
            rec = {'label': label, 'id': sid, 'w': img.size[0], 'h': img.size[1], 'sha1': hashlib.sha1(raw).hexdigest(),
                   'area_frac': round(st['area_frac'], 5), 'elong': round(st['elong'], 3), 'pad': st['pad'], 'side': st['side']}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(name, n, 'images, %.1f s each' % ((time.time() - t0) / n), flush=True)
    print('PREPARE DONE', name, n, 'new images', flush=True)


if __name__ == '__main__':
    main()
