"""How much do repeat photographs of the same fruit affect the reported accuracy?

The 13 most similar same-class image pairs found by find_repeats.py (feature similarity >= 0.9365) were confirmed by eye
as two photographs of one fruit. A test image is "exposed" when its partner was in the training or validation part of
the same fold. This script recomputes the pooled test accuracy of every main run without the exposed images.
Writes results/repeats_effect.json.
"""
import glob
import json
import os

import numpy as np

import common as C

THRESH = 0.9365
rows, classes, y = C.load_manifest()
fold = C.get_folds(y, C.load_groups(rows))
cand = json.load(open(os.path.join(C.RESULTS, 'repeats_candidates.json')))
pairs = [(c['ia'], c['ib']) for c in cand if c['sim'] >= THRESH]
exposed = set()
for a, b in pairs:
    if fold[a] != fold[b]:
        exposed.update((a, b))
out = {'threshold': THRESH, 'pairs': len(pairs), 'images_in_pairs': len({i for p in pairs for i in p}), 'pairs_split_across_folds': int(sum(fold[a] != fold[b] for a, b in pairs)),
       'exposed_test_images': len(exposed), 'runs': {}}
for run in ['svm'] + list(C.MODELS):
    idx, ok = [], []
    for f in sorted(glob.glob(os.path.join(C.RESULTS, run, 'fold[0-9]_pred.npz'))):
        d = np.load(f)
        idx.append(d['idx'])
        ok.append(d['prob'].argmax(1) == d['y'])
    idx, ok = np.concatenate(idx), np.concatenate(ok)
    keep = ~np.isin(idx, list(exposed))
    out['runs'][run] = {'accuracy_all': float(ok.mean()), 'accuracy_without_exposed': float(ok[keep].mean()), 'n_without': int(keep.sum()),
                        'accuracy_on_exposed': float(ok[~keep].mean()) if (~keep).any() else None}
    print('%-15s all %.4f  without exposed %.4f  (on the %d exposed images: %s)' % (run, ok.mean(), ok[keep].mean(), (~keep).sum(), out['runs'][run]['accuracy_on_exposed']))
json.dump(out, open(os.path.join(C.RESULTS, 'repeats_effect.json'), 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'runs'})
