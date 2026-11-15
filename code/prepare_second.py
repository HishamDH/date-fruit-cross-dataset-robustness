"""Build the working copy of SODA as a second, independent dataset for the model comparison.

Usage: python prepare_second.py
Reads data_ext/soda (prepare_external.py), keeps the whole-fruit photographs of all 13 classes (same rule and manual
exclusions as the external test) and writes data_soda_256/<class>/<id>.jpg, manifest.csv and masks.npy.
The images are the fruit-centred crops in which the fruit covers 29% of the frame, with their original background.
Afterwards every script runs on this dataset when the environment variable DATES_DATASET is set to "soda".
"""
import csv
import json
import os
import shutil

import numpy as np
from PIL import Image

import common as C

src = os.path.join(C.ROOT, 'data_ext', 'soda')
dst = os.path.join(C.ROOT, 'data_soda_256')
cfg = json.load(open(os.path.join(C.HERE, 'external_config.json')))['soda']
kind = json.load(open(os.path.join(src, 'kind.json')))
excl = set(cfg['exclude'])
recs = [json.loads(ln) for ln in open(os.path.join(src, 'manifest.jsonl'), encoding='utf-8')]
if os.path.isdir(dst):
    shutil.rmtree(dst)
rows, masks, drop = [], [], {'pit': 0, 'heap': 0, 'manual': 0}
for r in sorted(recs, key=lambda r: (r['label'], r['id'])):
    key = '%s/%s' % (r['label'], r['id'])
    if kind[key]['p_pit'] >= cfg['max_p_pit']:
        drop['pit'] += 1
        continue
    if r['area_frac'] > cfg['max_area']:
        drop['heap'] += 1
        continue
    if key in excl:
        drop['manual'] += 1
        continue
    os.makedirs(os.path.join(dst, r['label']), exist_ok=True)
    rel = '%s/%s.jpg' % (r['label'], r['id'])
    shutil.copy2(os.path.join(src, 'fruit', r['label'], r['id'] + '.jpg'), os.path.join(dst, rel))
    with Image.open(os.path.join(src, 'fruit_mask', r['label'], r['id'] + '.png')) as m:
        masks.append((np.asarray(m.convert('L')) > 127).astype(np.uint8))
    rows.append({'file': rel, 'label': r['label'], 'orig_w': r['w'], 'orig_h': r['h'], 'sha1': r['sha1'], 'dhash': '', 'orig': key})
with open(os.path.join(dst, 'manifest.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
np.save(os.path.join(dst, 'masks.npy'), np.stack(masks))
from collections import Counter
cnt = Counter(r['label'] for r in rows)
info = {'source_photos': len(recs), 'kept': len(rows), 'dropped': drop, 'per_class': dict(sorted(cnt.items())),
        'exact_duplicates': len(rows) - len({r['sha1'] for r in rows}), 'pad_used': sum(1 for r in recs if r['pad'] > 0)}
json.dump(info, open(os.path.join(dst, 'info.json'), 'w'), indent=1)
print(info)
