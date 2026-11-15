"""Two test-time remedies on the external photographs, with the networks trained on the main dataset (no retraining).

Usage: python external_tta.py <dataset>
For every network of the standard recipe (results/<model>_s<seed>/fold<k>.pt) and both forms of the photographs:
  base   the plain prediction (must reproduce external_test.py);
  tta    the mean prediction over the eight flips and quarter turns of the photograph;
  bn     networks with batch normalisation only: the stored statistics are replaced by the statistics of each
         batch of target photographs (64 per batch, fixed random order), which uses no labels.
Writes results/external_tta_<dataset>.json.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import external_test as E


@torch.no_grad()
def predict(model, X, mean, std, dev, mode, order=None):
    out = np.zeros((len(X), model.num_classes), np.float32)
    idx = np.arange(len(X)) if order is None else order
    for i in range(0, len(X), 64):
        b = idx[i:i + 64]
        if mode == 'bn' and len(b) < 64 <= len(X):
            b = idx[-64:]                                   # a full last batch, so that its statistics are not taken from a few photographs
        x = C.normalise(C.centre_crop(torch.from_numpy(X[b]).to(dev).permute(0, 3, 1, 2).float() / 255), mean, std)
        views = [x]
        if mode == 'tta':
            views = [torch.rot90(v, r, (2, 3)) for v in (x, torch.flip(x, (3,))) for r in range(4)]
        p = 0
        for v in views:
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                p = p + F.softmax(model(v).float(), 1)
        out[b] = (p / len(views)).cpu().numpy()
    return out


def main():
    name = sys.argv[1]
    cfg = json.load(open(os.path.join(C.HERE, 'external_config.json')))[name]
    rows, classes, _ = C.load_manifest()
    root, recs = E.load_external(name, cfg)
    y = np.array([classes.index(cfg['map'][r['label']]) for r in recs])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    order = np.random.default_rng(11).permutation(len(recs))          # mixed classes in every batch for the batch-statistics mode
    out = {'_info': {'n': len(recs), 'batch': 64}}
    for variant in ('frame', 'plain'):
        X = E.images(root, recs, variant)
        for key in C.MODELS:
            runs = E.run_dirs(key, 'standard')
            if not runs:
                continue
            model, mean, std = C.build_model(key, len(classes), pretrained=False)
            model.to(dev)
            has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
            acc = {'base': [], 'tta': []}
            if has_bn:
                acc['bn'] = []
            for run in runs:
                for k in range(5):
                    model.load_state_dict(torch.load(os.path.join(run, 'fold%d.pt' % k), map_location='cpu'))
                    model.eval()
                    for mode in ('base', 'tta'):
                        acc[mode].append(float((predict(model, X, mean, std, dev, mode).argmax(1) == y).mean()))
                    if has_bn:
                        for m in model.modules():
                            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                                m.train()                      # batch statistics; everything else stays in evaluation mode
                                m.momentum = 0.0               # the stored statistics are not overwritten
                        acc['bn'].append(float((predict(model, X, mean, std, dev, 'bn', order).argmax(1) == y).mean()))
            out['%s|%s' % (variant, key)] = {'n_models': len(acc['base']), 'has_bn': has_bn,
                                             **{m: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1))} for m, v in acc.items()}}
            print('%-22s %s' % ('%s|%s' % (variant, key), {m: round(100 * float(np.mean(v)), 1) for m, v in acc.items()}), flush=True)
            del model
            torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(C.RESULTS, 'external_tta_%s.json' % name), 'w'), indent=1)
    print('TTA DONE', name, flush=True)


if __name__ == '__main__':
    main()
