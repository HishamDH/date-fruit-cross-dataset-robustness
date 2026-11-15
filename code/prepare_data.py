"""Download the Kaggle date-fruit dataset and build a resized working copy.

Usage:
    python prepare_data.py download     # fetch the dataset archive from Kaggle into ../_data/dates.zip
    python prepare_data.py inspect      # print the folder layout, counts and image sizes
    python prepare_data.py build        # write ../data_256/<class>/<name>.jpg and ../data_256/manifest.csv

The archive is read in place (it is not extracted, the full-resolution photos take about 3 GB).
The working copy keeps the full frame: each photo is centre-cropped to a square and resized to
256 x 256 pixels. The manifest records, for every image, its class, original size, a SHA-1 of the
original file (exact duplicates) and a 64-bit difference hash (near duplicates).

Dataset: W. Alhamdan and J. Howe, "Date Fruit Image Dataset in Controlled Environment", Kaggle,
https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment
"""
import csv
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP = os.path.join(ROOT, '_data', 'dates.zip')
OUT = os.path.join(ROOT, 'data_256')
DATASET = 'wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment'
URL = 'https://www.kaggle.com/api/v1/datasets/download/' + DATASET
EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SIZE = 256


def download():
    os.makedirs(os.path.dirname(ZIP), exist_ok=True)
    req = urllib.request.Request(URL, headers={'User-Agent': 'dataset-downloader/1.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get('Content-Length') or 0)
        ctype = r.headers.get('Content-Type')
        print('status', r.status, 'type', ctype, 'bytes', total, flush=True)
        free = shutil.disk_usage(os.path.dirname(ZIP)).free
        if total and total + 1.5e9 > free:
            raise SystemExit('not enough free disk: need %d, free %d' % (total, free))
        done = 0
        step = 0
        with open(ZIP + '.part', 'wb') as fh:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if done // (200 << 20) > step:
                    step = done // (200 << 20)
                    print('  %d MB of %d MB' % (done >> 20, total >> 20), flush=True)
    os.replace(ZIP + '.part', ZIP)
    print('saved', ZIP, os.path.getsize(ZIP), 'bytes; is zip:', zipfile.is_zipfile(ZIP))


def members(z):
    return sorted((m for m in z.infolist() if m.filename.lower().endswith(EXT)), key=lambda m: m.filename)


def inspect():
    from PIL import Image
    with zipfile.ZipFile(ZIP) as z:
        ms = members(z)
        print('entries', len(z.infolist()), 'images', len(ms))
        by_dir = Counter(os.path.dirname(m.filename) for m in ms)
        for k, v in sorted(by_dir.items()):
            print('  %5d  %s' % (v, k))
        other = [m.filename for m in z.infolist() if not m.filename.lower().endswith(EXT) and not m.is_dir()]
        print('non-image files', other[:10])
        sizes = Counter()
        for m in ms[::25]:
            with Image.open(io.BytesIO(z.read(m))) as im:
                sizes[im.size] += 1
        print('sizes in a 1/25 sample:', sizes.most_common(8))
        print('uncompressed bytes', sum(m.file_size for m in ms))
        print('examples', [m.filename for m in ms[:4]])


def dhash(im, n=8):
    g = im.convert('L').resize((n + 1, n))
    px = list(g.getdata())
    bits = 0
    for r in range(n):
        for c in range(n):
            bits = (bits << 1) | (1 if px[r * (n + 1) + c] > px[r * (n + 1) + c + 1] else 0)
    return '%016x' % bits


def build():
    from PIL import Image, ImageOps
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        ms = members(z)
        for i, m in enumerate(ms):
            cls = os.path.basename(os.path.dirname(m.filename)).strip()
            name = os.path.splitext(os.path.basename(m.filename))[0]
            raw = z.read(m)
            sha = hashlib.sha1(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                w, h = im.size
                s = min(w, h)
                box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
                sq = im.crop(box).resize((SIZE, SIZE), Image.LANCZOS)
                dh = dhash(sq)
            os.makedirs(os.path.join(OUT, cls), exist_ok=True)
            rel = '%s/%s.jpg' % (cls, name)
            sq.save(os.path.join(OUT, rel), quality=95, subsampling=0)
            rows.append({'file': rel, 'label': cls, 'orig_w': w, 'orig_h': h, 'sha1': sha, 'dhash': dh,
                         'orig': m.filename})
            if i % 200 == 0:
                print(i, '/', len(ms), flush=True)
    with open(os.path.join(OUT, 'manifest.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('wrote', len(rows), 'images;', sorted(Counter(r['label'] for r in rows).items()))
    print('exact duplicate files:', len(rows) - len({r['sha1'] for r in rows}))
    print('distinct file names per class collide:', len(rows) - len({r['file'] for r in rows}))


if __name__ == '__main__':
    {'download': download, 'inspect': inspect, 'build': build}[sys.argv[1]]()
