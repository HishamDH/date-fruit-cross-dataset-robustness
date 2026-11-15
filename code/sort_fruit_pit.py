"""Separate whole-fruit photos from pit photos in a prepared external dataset (SODA mixes both in one folder).

Usage: python sort_fruit_pit.py <dataset>
Step 1: confident labels from shape: pit = small and elongated object, fruit = large and compact object.
Step 2: a logistic regression on features of an ImageNet-pretrained network (computed on the fruit-centred crops)
        is fitted on the confident images and applied to all images.
Writes data_ext/<dataset>/kind.json  {"<label>/<id>": {"kind": "fruit"|"pit", "p_pit": ...}} and contact sheets
data_ext/<dataset>/_check_pits.jpg and _check_fruits_<class>.jpg for a visual check.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression

import common as C

name = sys.argv[1]
root = os.path.join(C.ROOT, 'data_ext', name)
recs = [json.loads(ln) for ln in open(os.path.join(root, 'manifest.jsonl'), encoding='utf-8')]
X = np.zeros((len(recs), 256, 256, 3), np.uint8)
for i, r in enumerate(recs):
    with Image.open(os.path.join(root, 'fruit', r['label'], r['id'] + '.jpg')) as im:
        X[i] = np.asarray(im.convert('RGB'))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import timm
model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, num_classes=0).to(dev).eval()
mean, std = model.pretrained_cfg['mean'], model.pretrained_cfg['std']
feats = []
with torch.no_grad():
    for i in range(0, len(X), 32):
        x = C.centre_crop(torch.from_numpy(X[i:i + 32]).to(dev).permute(0, 3, 1, 2).float() / 255)
        feats.append(model(C.normalise(x, mean, std)).float().cpu().numpy())
Fm = np.concatenate(feats)
el = np.array([r['elong'] for r in recs])
ar = np.array([r['area_frac'] for r in recs])
pit0 = (el >= 2.3) & (ar <= 0.035)
fruit0 = (ar >= 0.06) & (el <= 1.9)
print('confident pits', int(pit0.sum()), 'confident fruits', int(fruit0.sum()), 'undecided', int((~pit0 & ~fruit0).sum()), flush=True)
idx = np.where(pit0 | fruit0)[0]
clf = LogisticRegression(C=1.0, max_iter=2000).fit(Fm[idx], pit0[idx].astype(int))
p = clf.predict_proba(Fm)[:, 1]
kind = {('%s/%s' % (r['label'], r['id'])): {'kind': 'pit' if p[i] >= 0.5 else 'fruit', 'p_pit': round(float(p[i]), 4), 'elong': r['elong'], 'area_frac': r['area_frac']}
        for i, r in enumerate(recs)}
json.dump(kind, open(os.path.join(root, 'kind.json'), 'w'), indent=0)
labels = sorted({r['label'] for r in recs})
for lab in labels:
    ii = [i for i, r in enumerate(recs) if r['label'] == lab]
    print('  %-18s fruit %3d  pit %3d  uncertain (0.2-0.8) %d' % (lab, sum(p[i] < 0.5 for i in ii), sum(p[i] >= 0.5 for i in ii), sum(0.2 < p[i] < 0.8 for i in ii)), flush=True)
T = 96


def sheet(ids, path, cols=20):
    if not ids:
        return
    rows = (len(ids) + cols - 1) // cols
    sh = Image.new('RGB', (cols * T, rows * (T + 10)), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    for j, i in enumerate(ids):
        x0, y0 = (j % cols) * T, (j // cols) * (T + 10)
        sh.paste(Image.fromarray(X[i]).resize((T, T)), (x0, y0))
        dr.text((x0 + 1, y0 + T - 1), '%.2f %s' % (p[i], recs[i]['label'][:8]), fill=(255, 255, 0))
    sh.save(path, quality=80)


order = np.argsort(p)
sheet([i for i in order if p[i] >= 0.5], os.path.join(root, '_check_pits.jpg'))            # least certain pits first
fr = [i for i in order[::-1] if p[i] < 0.5]                                                 # least certain fruits first
sheet(fr[:300], os.path.join(root, '_check_fruits_least_certain.jpg'))
print('SORT DONE', name, 'fruits', int((p < 0.5).sum()), 'pits', int((p >= 0.5).sum()), flush=True)
