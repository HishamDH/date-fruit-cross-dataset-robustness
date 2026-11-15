"""Contact sheets of the external datasets, to look at them before deciding how to use them.

Usage: python sheet_external.py
Writes ../_data/external/_sheets/<dataset>_<class>.jpg (random samples, jpg files first, then png files).
"""
import io
import os
import random
import re
import zipfile

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), '_data', 'external')
OUT = os.path.join(EXT, '_sheets')
os.makedirs(OUT, exist_ok=True)
random.seed(0)
T = 200


def thumb(im):
    im = im.convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 0, 255, 255))      # magenta shows where a png is transparent
    im = Image.alpha_composite(bg, im).convert('RGB')
    im.thumbnail((T, T))
    canvas = Image.new('RGB', (T, T), (40, 40, 40))
    canvas.paste(im, ((T - im.size[0]) // 2, (T - im.size[1]) // 2))
    return canvas


def sheet(items, path, cols=8):
    rows = (len(items) + cols - 1) // cols
    sh = Image.new('RGB', (cols * T, rows * (T + 14)), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    for i, (label, im) in enumerate(items):
        x, y = (i % cols) * T, (i // cols) * (T + 14)
        sh.paste(thumb(im), (x, y))
        dr.text((x + 2, y + T + 1), label[:30], fill=(255, 255, 255))
    sh.save(path, quality=85)


soda = os.path.join(EXT, 'soda')
for d in sorted(os.listdir(soda)) if os.path.isdir(soda) else []:
    p = os.path.join(soda, d)
    if not os.path.isdir(p) or d.startswith('.') or d == 'TabularDataset':
        continue
    eng = re.sub(r'[^A-Za-z ]', '', d).strip().replace(' ', '_')
    files = sorted(os.listdir(p))
    jpg = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
    png = [f for f in files if f.lower().endswith('.png')]
    pick = random.sample(jpg, min(16, len(jpg))) + random.sample(png, min(8, len(png)))
    items = []
    for f in pick:
        try:
            with Image.open(os.path.join(p, f)) as im:
                items.append(('%s %dx%d' % (os.path.splitext(f)[1], im.size[0], im.size[1]), im.copy()))
        except Exception as e:  # noqa: BLE001
            print('cannot open', d, f, repr(e)[:60])
    if items:
        sheet(items, os.path.join(OUT, 'soda_%s.jpg' % eng))
        print('soda', eng, 'jpg', len(jpg), 'png', len(png))

z = os.path.join(EXT, 'uci_ajwa_or_medjool', 'ajwa_or_medjool.zip')
if os.path.exists(z):
    with zipfile.ZipFile(z) as zf:
        for cls in ('Ajwa', 'Medjool'):
            names = sorted(n for n in zf.namelist() if n.startswith('ajwa+or+medjool/%s/' % cls) and n.lower().endswith(('.jpg', '.png')))
            jpg = [n for n in names if n.lower().endswith('.jpg')]
            png = [n for n in names if n.lower().endswith('.png')]
            items = []
            for n in jpg[:16] + png[:8]:
                with Image.open(io.BytesIO(zf.read(n))) as im:
                    items.append(('%s %dx%d' % (os.path.basename(n)[:14], im.size[0], im.size[1]), im.copy()))
            sheet(items, os.path.join(OUT, 'uci_%s.jpg' % cls))
            print('uci', cls, 'jpg', len(jpg), 'png', len(png), 'first names', [os.path.basename(n) for n in jpg[:12]])
