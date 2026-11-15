"""Does a second source of training photographs help on a third, unseen source?

Usage: python train_multisource.py --model convnext --extra soda --target medina27 --seed 0
       python train_multisource.py --model convnext --extra none --target medina27 --seed 0     (control)
Training set: 90% of the main dataset (all nine classes) plus, with --extra, every whole-fruit photograph of the
shared classes of the extra dataset (its scale- and background-normalised copy). The remaining 10% of the main
dataset is used to choose the epoch. The network is then tested once on the target dataset, which it has never
seen, in two forms: the photograph as it is (frame) and the normalised copy (plain).
Writes results/multisource/<model>_<extra>_to_<target>_s<seed>.json.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit

import common as C
import external_test as E
import train_cv as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(C.MODELS))
    ap.add_argument('--extra', default='none')
    ap.add_argument('--target', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=16)
    a = ap.parse_args()
    out_dir = os.path.join(C.RESULTS, 'multisource')
    os.makedirs(out_dir, exist_ok=True)
    targets = a.target.split(',')          # the control (--extra none) is trained once and tested on every target
    assert a.extra not in targets and (a.extra == 'none' or len(targets) == 1)
    paths = {t: os.path.join(out_dir, '%s_%s_to_%s_s%d.json' % (a.model, a.extra, t, a.seed)) for t in targets}
    if all(os.path.exists(p) for p in paths.values()):
        print('exists, skipping', list(paths.values()))
        return
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    cfg = json.load(open(os.path.join(C.HERE, 'external_config.json')))
    lr = float(json.load(open(os.path.join(C.RESULTS, 'lr_choice.json')))[a.model]['lr'])
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    tr, va = next(StratifiedShuffleSplit(1, test_size=0.10, random_state=C.FOLD_SEED).split(np.zeros(len(y)), y))
    Xtr, ytr = X[tr], y[tr]
    n_extra = 0
    if a.extra != 'none':
        root, recs = E.load_external(a.extra, cfg[a.extra])
        Xe = E.images(root, recs, 'plain')
        ye = np.array([classes.index(cfg[a.extra]['map'][r['label']]) for r in recs])
        Xtr, ytr = np.concatenate([Xtr, Xe]), np.concatenate([ytr, ye])
        n_extra = len(ye)
    seed = 5000 + 100 * a.seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model, mean, std = C.build_model(a.model, len(classes), pretrained=True)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = int(np.ceil(len(ytr) / a.bs))
    total, warm = a.epochs * steps, steps

    def lr_at(s):
        return (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    best = (-1.0, float('inf'), None, -1)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for b in T.batches(np.arange(len(ytr)), a.bs, True, rng):
            x = C.augment(torch.from_numpy(Xtr[b]).to(dev).permute(0, 3, 1, 2).float() / 255, gen)
            t = torch.from_numpy(ytr[b]).to(dev)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                loss = F.cross_entropy(model(C.normalise(x, mean, std)).float(), t, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        pv = T.predict(model, X, va, mean, std, dev)
        vl = float(-np.log(np.clip(pv[np.arange(len(va)), y[va]], 1e-12, 1)).mean())
        vacc = float((pv.argmax(1) == y[va]).mean())
        if (vacc, -vl) > (best[0], -best[1]):
            best = (vacc, vl, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), ep + 1)
    model.load_state_dict(best[2])
    train_seconds = time.time() - t0
    for target, out_path in paths.items():
        root_t, recs_t = E.load_external(target, cfg[target])
        yt = np.array([classes.index(cfg[target]['map'][r['label']]) for r in recs_t])
        shared = sorted(set(yt.tolist()))
        res = {'model': a.model, 'extra': a.extra, 'target': target, 'seed': a.seed, 'n_train_main': int(len(tr)), 'n_train_extra': int(n_extra),
               'n_target': int(len(yt)), 'shared': [classes[c] for c in shared], 'val_acc_main': best[0], 'best_epoch': best[3],
               'train_seconds': train_seconds}
        for variant in ('frame', 'plain'):
            Xt = E.images(root_t, recs_t, variant)
            p = T.predict(model, Xt, np.arange(len(Xt)), mean, std, dev)
            s = E.score(yt, p, shared)
            conf = np.zeros((len(classes), len(classes)), int)
            np.add.at(conf, (yt, p.argmax(1)), 1)
            res[variant] = {'accuracy': s['accuracy'], 'recall': {classes[c]: s['recall'][c] for c in shared}, 'confusion': conf.tolist()}
        json.dump(res, open(out_path, 'w'), indent=1)
        print('MULTI %s extra=%s -> %s seed %d | val(main) %.4f | target frame %.3f plain %.3f | %.0fs' % (
            a.model, a.extra, target, a.seed, best[0], res['frame']['accuracy'], res['plain']['accuracy'], train_seconds), flush=True)


if __name__ == '__main__':
    main()
