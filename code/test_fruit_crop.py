"""Try the fruit locator on a few external photos and save a sheet: photo, frame crop, fruit crop with stats."""
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import fruit_crop as FC

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
EXT = os.path.join(os.path.dirname(FC.HERE), '_data', 'external')
random.seed(1)
root = os.path.join(EXT, 'soda')
files = []
for d in sorted(os.listdir(root)):
    p = os.path.join(root, d)
    if os.path.isdir(p) and not d.startswith('.') and d != 'TabularDataset':
        js = [f for f in sorted(os.listdir(p)) if f.lower().endswith('.jpg')]
        files += [os.path.join(p, f) for f in random.sample(js, 3)]
T = 180
sheet = Image.new('RGB', (6 * T, ((len(files) + 1) // 2) * (T + 14)), (0, 0, 0))
dr = ImageDraw.Draw(sheet)
t0 = time.time()
for i, f in enumerate(files):
    img = ImageOps.exif_transpose(Image.open(f)).convert('RGB')
    crop, cm, st = FC.fruit_crop(img)
    fr = FC.frame_crop(img)
    x0, y0 = (i % 2) * 3 * T, (i // 2) * (T + 14)
    th = img.copy()
    th.thumbnail((T, T))
    sheet.paste(th, (x0, y0))
    sheet.paste(Image.fromarray(fr).resize((T, T)), (x0 + T, y0))
    sheet.paste(Image.fromarray(crop).resize((T, T)), (x0 + 2 * T, y0))
    dr.text((x0 + 2, y0 + T + 1), 'area %.3f elong %.2f pad %d' % (st['area_frac'], st['elong'], st['pad']), fill=(255, 255, 255))
print('%.1f s per image' % ((time.time() - t0) / len(files)))
sheet.save(os.path.join(EXT, '_sheets', 'fruit_crop_test.jpg'), quality=88)
print('saved')
