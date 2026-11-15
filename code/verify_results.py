"""Independent checks of the result files (does not use analyze.py).

1. Folds: every image is in exactly one test fold; train, validation and test indices of a fold are disjoint.
2. Every main run: the saved test predictions cover all images exactly once and belong to the right fold;
   accuracy recomputed from the predictions equals the value stored in fold<k>.json.
3. The numbers in summary.json equal the recomputed ones.
Prints PASS or FAIL per check and exits with code 1 on any failure.
"""
import glob
import json
import os
import sys

import numpy as np

import common as C

fails = 0


def check(name, ok, detail=''):
    global fails
    print('%-4s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
    fails += 0 if ok else 1


rows, classes, y = C.load_manifest()
groups = C.load_groups(rows)
fold = C.get_folds(y, groups)
n = len(y)
check('manifest has 1658 images in 9 classes', n == 1658 and len(classes) == 9, str(np.bincount(y).tolist()))
check('each image has one test fold', set(fold.tolist()) == set(range(5)) and len(fold) == n)
for k in range(5):
    tr, va, te = C.split_fold(y, fold, k, groups)
    ok = len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0 and len(tr) + len(va) + len(te) == n
    cls_ok = all(len(np.unique(y[s])) == 9 for s in (tr, va, te))
    check('fold %d: train/val/test disjoint, complete, all classes present' % k, ok and cls_ok, '%d/%d/%d' % (len(tr), len(va), len(te)))

S = json.load(open(os.path.join(C.RESULTS, 'summary.json')))
runs = [r for r in sorted(os.listdir(C.RESULTS)) if os.path.isdir(os.path.join(C.RESULTS, r)) and len(glob.glob(os.path.join(C.RESULTS, r, 'fold[0-9]_pred.npz'))) == 5]
print('five-fold runs:', runs)
PER_RUN = {}
for r in runs:
    idx_all, ok_all, accs, mca = [], [], [], []
    good = True
    for k in range(5):
        d = np.load(os.path.join(C.RESULTS, r, 'fold%d_pred.npz' % k))
        j = json.load(open(os.path.join(C.RESULTS, r, 'fold%d.json' % k)))
        good &= bool(np.all(fold[d['idx']] == k)) and bool(np.all(y[d['idx']] == d['y']))
        acc = float((d['prob'].argmax(1) == d['y']).mean())
        good &= abs(acc - j['test']['accuracy']) < 1e-9
        good &= abs(d['prob'].sum(1) - 1).max() < 1e-3
        accs.append(acc)
        mca.append(np.mean([v['accuracy'] for v in j['robustness'].values()]))
        good &= len(j['robustness']) == 24
        idx_all.append(d['idx'])
        ok_all.append(d['prob'].argmax(1) == d['y'])
    idx_all, ok_all = np.concatenate(idx_all), np.concatenate(ok_all)
    good &= sorted(idx_all.tolist()) == list(range(n))
    s = S['seed_runs'][r]
    PER_RUN[r] = (accs, mca)
    same = (abs(np.mean(accs) - s['accuracy']['mean']) < 1e-9 and abs(np.std(accs, ddof=1) - s['accuracy']['std']) < 1e-9
            and s['pooled']['correct'] == int(ok_all.sum()) and abs(np.mean(mca) - s['mean_corrupted_accuracy']['mean']) < 1e-9)
    check('%-22s predictions consistent; summary matches' % r, good and same,
          'acc %.2f +- %.2f, pooled %d/%d, mCA %.2f' % (100 * np.mean(accs), 100 * np.std(accs, ddof=1), ok_all.sum(), n, 100 * np.mean(mca)))
# the numbers reported per configuration are the mean and standard deviation over all folds of all its seeds
for name, g in S['runs'].items():
    accs = np.concatenate([PER_RUN[d][0] for d in g['dirs']])
    mca = np.concatenate([PER_RUN[d][1] for d in g['dirs']])
    ok = (abs(accs.mean() - g['accuracy']['mean']) < 1e-9 and abs(accs.std(ddof=1) - g['accuracy']['std']) < 1e-9
          and abs(mca.mean() - g['mean_corrupted_accuracy']['mean']) < 1e-9 and len(accs) == 5 * g['n_seeds'])
    check('%-22s reported mean over %d seeds matches' % (name, g['n_seeds']), ok, 'acc %.2f +- %.2f, mCA %.2f' % (100 * accs.mean(), 100 * accs.std(ddof=1), 100 * mca.mean()))
# the external test sets share no image with the main dataset (compared by the SHA-1 of the original files)
main_sha = {r['sha1'] for r in rows}
for name in ('medina27', 'soda'):
    m = os.path.join(C.ROOT, 'data_ext', name, 'manifest.jsonl')
    if os.path.exists(m):
        ext_sha = {json.loads(ln)['sha1'] for ln in open(m, encoding='utf-8')}
        check('external set %-9s shares no file with the main dataset' % name, len(main_sha & ext_sha) == 0, '%d files' % len(ext_sha))
# the learning-rate choice must follow validation accuracy (ties: validation loss), never the test fold
lr = json.load(open(os.path.join(C.RESULTS, 'lr_choice.json')))
for m, v in lr.items():
    cand = v['candidates']
    pick = max(cand, key=lambda q: (cand[q]['val_acc'], -cand[q]['val_loss']))
    check('learning rate of %-15s follows validation' % m, pick == v['lr'], '%s %s' % (v['lr'], {q: round(c['val_acc'], 4) for q, c in cand.items()}))
# a majority-class predictor, for reference
print('largest class share: %.2f%%; uniform chance: %.2f%%' % (100 * np.bincount(y).max() / n, 100 / 9))
print('RESULT:', 'ALL CHECKS PASSED' if fails == 0 else '%d CHECK(S) FAILED' % fails)
sys.exit(1 if fails else 0)
