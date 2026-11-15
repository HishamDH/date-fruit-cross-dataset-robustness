"""Layout of the full SODA archive from Kaggle, and a contact sheet of the prepared Medina 27 crops for the shared classes."""
import json
import os
import sys
import zipfile
from collections import Counter

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
z = zipfile.ZipFile(os.path.join(ROOT, '_data', 'external', 'soda_kaggle', 'soda_kaggle.zip'))
names = [n for n in z.namelist() if not n.endswith('/')]
print('SODA zip entries', len(names))
c = Counter(('/'.join(n.split('/')[:-1]), os.path.splitext(n)[1].lower()) for n in names)
for k, v in sorted(c.items()):
    print('   %5d  %s  %s' % (v, k[0], k[1]))

root = os.path.join(ROOT, 'data_ext', 'medina27')
recs = [json.loads(ln) for ln in open(os.path.join(root, 'manifest.jsonl'), encoding='utf-8')]
print('\nMedina labels:', sorted({r['label'] for r in recs}))
ids = sorted(r['id'] for r in recs if r['label'] == 'Ajwa')
print('Ajwa ids:', ids)
shared = ['Ajwa', 'Galaxy_Sukkari', 'Medjool', 'Nabt_Ali', 'Saggae', 'Sheshee', 'Sukkary_Ruttab', 'Meftal_Sukkari']
T = 110
shared = shared[:6]
for kind in ('fruit', 'plain'):
    sh = Image.new('RGB', (22 * T, len(shared) * 1 * (T + 2)), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    for ci, lab in enumerate(shared):
        rr = sorted((r for r in recs if r['label'] == lab), key=lambda r: r['id'])
        for j, r in enumerate(rr[:22]):
            x0, y0 = (j % 22) * T, ci * (T + 2)
            sh.paste(Image.open(os.path.join(root, kind, lab, r['id'] + '.jpg')).resize((T, T)), (x0, y0))
        dr.text((3, ci * (T + 2) + 2), lab, fill=(255, 255, 0))
    sh.save(os.path.join(root, '_sheet_%s.jpg' % kind), quality=80)
import numpy as np
ar = np.array([r['area_frac'] for r in recs])
print('Medina area fraction of the located object: 5%% %.3f median %.3f 95%% %.3f | pad>0: %d' % (np.percentile(ar, 5), np.median(ar), np.percentile(ar, 95), sum(r['pad'] > 0 for r in recs)))
print('sizes', Counter((r['w'], r['h']) for r in recs).most_common(5))
