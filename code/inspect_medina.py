"""Layout of the Medina 27 archive and a first look at the fruit/pit statistics of the prepared SODA copy."""
import io
import json
import os
import sys
import zipfile
from collections import Counter

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
zp = os.path.join(ROOT, '_data', 'external', 'medina27', 'medina27.zip')
z = zipfile.ZipFile(zp)
names = [n for n in z.namelist() if not n.endswith('/')]
print('entries', len(names))
c = Counter(('/'.join(n.split('/')[:-2]), os.path.splitext(n)[1].lower()) for n in names)
for k, v in sorted(c.items()):
    print('   %5d  %s  %s' % (v, k[0], k[1]))
lvl = Counter('/'.join(n.split('/')[:-1]) for n in names)
print('folders (first 45):')
for k, v in sorted(lvl.items())[:45]:
    print('   %4d  %s' % (v, k))
other = [n for n in names if not n.lower().endswith(('.png', '.jpg', '.jpeg'))]
print('non-image files:', other[:20])
for n in other:
    if n.lower().endswith(('.txt', '.md', '.csv', '.json')) and z.getinfo(n).file_size < 6000:
        print('=====', n)
        print(z.read(n).decode('utf-8', 'replace')[:3000])
ex = [n for n in names if n.lower().endswith('.png')][:3] + [n for n in names if n.lower().endswith(('.jpg', '.jpeg'))][:3]
for n in ex:
    with Image.open(io.BytesIO(z.read(n))) as im:
        print('example', n, im.size, im.mode)

m = os.path.join(ROOT, 'data_ext', 'soda_hf', 'manifest.jsonl')
recs = [json.loads(ln) for ln in open(m, encoding='utf-8')]
print('\nSODA (HF) prepared:', len(recs))
for lab in sorted({r['label'] for r in recs}):
    rr = [r for r in recs if r['label'] == lab]
    el = np.array([r['elong'] for r in rr])
    ar = np.array([r['area_frac'] for r in rr])
    fruit = (ar >= 0.03) & (ar <= 0.4) & (el <= 2.0)
    print('  %-16s n %3d | rule says fruit %3d | elong quartiles %s | area quartiles %s' % (
        lab, len(rr), fruit.sum(), np.round(np.percentile(el, [10, 50, 90]), 2).tolist(), np.round(np.percentile(ar, [10, 50, 90]), 3).tolist()))
