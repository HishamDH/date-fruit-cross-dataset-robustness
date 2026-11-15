"""Traditional image-processing baseline: color histogram + HOG + LBP features with an RBF support vector machine.

Usage: python baseline_classical.py
Uses the same folds and the same corrupted test images as the deep models and writes
results/svm/fold<k>.json and fold<k>_pred.npz.
"""
import json
import os
import time

import cv2
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as C


def hog(gray, cell=16, bins=9):
    """Histogram of oriented gradients (Dalal-Triggs layout): 128 x 128 image, 16 x 16 cells, 2 x 2-cell
    blocks with a stride of one cell, 9 unsigned orientation bins, L2-Hys block normalisation -> 1764 values."""
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    pos = ang / (180.0 / bins) - 0.5
    lo = np.floor(pos).astype(int)
    w_hi = pos - lo
    n = 128 // cell
    hist = np.zeros((n, n, bins), np.float32)
    cy, cx = np.mgrid[0:128, 0:128]
    cy //= cell
    cx //= cell
    np.add.at(hist, (cy, cx, lo % bins), mag * (1 - w_hi))
    np.add.at(hist, (cy, cx, (lo + 1) % bins), mag * w_hi)
    blocks = np.concatenate([hist[:-1, :-1], hist[:-1, 1:], hist[1:, :-1], hist[1:, 1:]], axis=2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(2, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(2, keepdims=True) + 1e-6)
    return blocks.ravel()


def _uniform_map():
    m = np.zeros(256, np.int64)
    nxt = 0
    for v in range(256):
        b = [(v >> i) & 1 for i in range(8)]
        trans = sum(b[i] != b[(i + 1) % 8] for i in range(8))
        if trans <= 2:
            m[v] = nxt
            nxt += 1
        else:
            m[v] = 58
    return m


UNI = _uniform_map()


def lbp_hist(gray):
    """Uniform local binary patterns (8 neighbours, radius 1): 58 uniform codes + 1 bin for the rest."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.int64)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offs):
        code |= ((g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx] >= c).astype(np.int64) << i)
    h = np.bincount(UNI[code].ravel(), minlength=59).astype(np.float64)
    return h / h.sum()


def features(img):
    """img: 224 x 224 x 3 uint8 RGB -> 50 color + 1764 HOG + 59 LBP values."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hs = [np.histogram(hsv[..., 0], 18, (0, 180))[0], np.histogram(hsv[..., 1], 16, (0, 256))[0], np.histogram(hsv[..., 2], 16, (0, 256))[0]]
    col = np.concatenate([h / max(1, h.sum()) for h in hs])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.concatenate([col, hog(gray), lbp_hist(gray)])


def crop_uint8(X, idx, transform=None):
    x = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).float() / 255
    x = C.centre_crop(x)
    if transform is not None:
        x = transform(x, idx)
    return (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def feats(X, idx, transform=None):
    return np.stack([features(im) for im in crop_uint8(X, idx, transform)])


def main():
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    groups = C.load_groups(rows)
    fold = C.get_folds(y, groups)
    masks = np.load(os.path.join(C.DATA, 'masks.npy'))
    out = os.path.join(C.RESULTS, 'svm')
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    F_clean = feats(X, np.arange(len(y)))
    print('features', F_clean.shape, '%.0fs' % (time.time() - t0), flush=True)
    accs = []
    for k in range(C.K_FOLDS):
        tr, va, te = C.split_fold(y, fold, k, groups)
        trva = np.concatenate([tr, va])
        t1 = time.time()
        d = F_clean.shape[1]
        grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=0)),
                            {'svc__C': [1, 10, 100], 'svc__gamma': [0.3 / d, 1.0 / d, 3.0 / d]},
                            cv=StratifiedKFold(3, shuffle=True, random_state=k), n_jobs=4)
        grid.fit(F_clean[trva], y[trva])
        clf = grid.best_estimator_
        pt = clf.predict_proba(F_clean[te])
        res = {'model': 'svm', 'fold': k, 'best_params': grid.best_params_, 'inner_cv_acc': float(grid.best_score_),
               'n_train': len(trva), 'n_test': len(te), 'train_seconds': time.time() - t1, 'test': C.metrics(y[te], pt)}
        np.savez_compressed(os.path.join(out, 'fold%d_pred.npz' % k), idx=te, y=y[te], prob=pt)
        rob = {}
        for kind in C.CORRUPTIONS:
            for sev in (1, 2, 3):
                g = torch.Generator().manual_seed(7000 + 10 * k + sev)

                def tf(x, b, kind=kind, sev=sev, g=g):
                    m = None
                    if kind == 'background':
                        m = C.centre_crop(torch.from_numpy(masks[b]).float().unsqueeze(1))
                    return C.corrupt(x, kind, sev, mask=m, gen=g)

                rob['%s_%d' % (kind, sev)] = C.metrics(y[te], clf.predict_proba(feats(X, te, tf)))
        res['robustness'] = rob
        json.dump(res, open(os.path.join(out, 'fold%d.json' % k), 'w'), indent=1)
        accs.append(res['test']['accuracy'])
        print('fold %d  %s  test acc %.4f  f1 %.4f  auc %.4f  mean corrupted acc %.4f  (%.0fs)' % (
            k, grid.best_params_, res['test']['accuracy'], res['test']['f1'], res['test']['auc'],
            float(np.mean([v['accuracy'] for v in rob.values()])), time.time() - t1), flush=True)
    print('SUMMARY svm: test accuracy %.4f +/- %.4f' % (np.mean(accs), np.std(accs, ddof=1)))


if __name__ == '__main__':
    main()
