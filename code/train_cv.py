"""Fine-tune one model with stratified 5-fold cross-validation and test it on clean and corrupted images.

Usage:
    python train_cv.py --model convnext                      # all folds, standard recipe
    python train_cv.py --model convnext --folds 0 --lr 3e-4 --tag lr3e-4 --val-only   # learning-rate check
    python train_cv.py --model convnext --no-aug --tag noaug                           # ablation
    python train_cv.py --model convnext --scratch --epochs 60 --tag scratch            # ablation

Writes results/<model>[_<tag>]/fold<k>.json (history, metrics, robustness) and fold<k>_pred.npz
(test indices, labels, class probabilities). The checkpoint of fold 0 is kept for Grad-CAM.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def batches(idx, bs, shuffle, rng=None):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, X, idx, mean, std, dev, bs=64, transform=None):
    model.eval()
    out = []
    for b in batches(idx, bs, False):
        x = torch.from_numpy(X[b]).to(dev).permute(0, 3, 1, 2).float() / 255
        x = C.centre_crop(x)
        if transform is not None:
            x = transform(x, b)
        with torch.autocast('cuda', enabled=dev.type == 'cuda'):
            logits = model(C.normalise(x, mean, std))
        out.append(F.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out)


def run_fold(args, k, X, y, fold, groups, masks, classes, dev):
    tr, va, te = C.split_fold(y, fold, k, groups)
    seed = 1000 + k + 100 * args.seed      # --seed 0 reproduces the first set of runs
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    model, mean, std = C.build_model(args.model, len(classes), pretrained=not args.scratch)
    model.to(dev)
    opt =torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = int(np.ceil(len(tr) / args.bs))
    total, warm = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    hist = []
    best = (-1.0, float('inf'), None, -1)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for b in batches(tr, args.bs, True, rng):
            x = torch.from_numpy(X[b]).to(dev).permute(0, 3, 1, 2).float() / 255
            if args.no_aug:
                x = C.centre_crop(x)
            elif args.strong_aug:
                x = C.augment(x, gen, strong=True, mask=torch.from_numpy(masks[b]).to(dev).float().unsqueeze(1))
            else:
                x = C.augment(x, gen, rigid=args.rigid_aug)
            t = torch.from_numpy(y[b]).to(dev)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                logits = model(C.normalise(x, mean, std))
                loss = F.cross_entropy(logits.float(), t, label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(b)
            tc += (logits.argmax(1) == t).sum().item()
            tn += len(b)
        pv = predict(model, X, va, mean, std, dev)
        vl = float(-np.log(np.clip(pv[np.arange(len(va)), y[va]], 1e-12, 1)).mean())
        vacc = float((pv.argmax(1) == y[va]).mean())
        hist.append({'epoch': ep + 1, 'train_loss': tl / tn, 'train_acc': tc / tn, 'val_loss': vl, 'val_acc': vacc})
        if (vacc, -vl) > (best[0], -best[1]):
            best = (vacc, vl, copy.deepcopy({k_: v.detach().cpu() for k_, v in model.state_dict().items()}), ep + 1)
        print('  fold %d epoch %2d  train %.4f/%.4f  val %.4f/%.4f  (%.0fs)' % (
            k, ep + 1, tl / tn, tc / tn, vl, vacc, time.time() - t0), flush=True)
    train_time = time.time() - t0
    model.load_state_dict(best[2])
    res = {'model': args.model, 'tag': args.tag, 'fold': k, 'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
           'best_epoch': best[3], 'best_val_acc': best[0], 'train_seconds': train_time, 'history': hist,
           'args': vars(args), 'mean': mean, 'std': std}
    if not args.val_only:
        pt = predict(model, X, te, mean, std, dev)
        res['test'] = C.metrics(y[te], pt)
        np.savez_compressed(os.path.join(args.out, 'fold%d_pred.npz' % k), idx=te, y=y[te], prob=pt)
        rob = {}
        for kind in C.CORRUPTIONS:
            for sev in (1, 2, 3):
                g = torch.Generator(device=dev).manual_seed(7000 + 10 * k + sev)

                def tf(x, b, kind=kind, sev=sev, g=g):
                    m = None
                    if kind == 'background':
                        m = C.centre_crop(torch.from_numpy(masks[b]).to(dev).float().unsqueeze(1))
                    return C.corrupt(x, kind, sev, mask=m, gen=g)

                pc = predict(model, X, te, mean, std, dev, transform=tf)
                rob['%s_%d' % (kind, sev)] = C.metrics(y[te], pc)
        res['robustness'] = rob
        print('  fold %d TEST acc %.4f f1 %.4f auc %.4f | best epoch %d | mean corrupted acc %.4f' % (
            k, res['test']['accuracy'], res['test']['f1'], res['test']['auc'], best[3],
            float(np.mean([v['accuracy'] for v in rob.values()]))), flush=True)
        if (k == 0 and args.keep_ckpt) or args.keep_all_ckpt:
            torch.save(best[2], os.path.join(args.out, 'fold%d.pt' % k))
    json.dump(res, open(os.path.join(args.out, 'fold%d.json' % k), 'w'), indent=1)
    del model, opt, scaler
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(C.ALL_MODELS))
    ap.add_argument('--strong-aug', action='store_true', help='standard recipe plus background, colour, blur and noise randomisation')
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--ls', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--no-aug', action='store_true')
    ap.add_argument('--rigid-aug', action='store_true', help='shift, rotate, flip and jitter only: no change of scale or aspect ratio')
    ap.add_argument('--scratch', action='store_true')
    ap.add_argument('--val-only', action='store_true', help='stop after validation; the test fold is not touched')
    ap.add_argument('--keep-ckpt', action='store_true')
    ap.add_argument('--keep-all-ckpt', action='store_true', help='save the selected weights of every fold (needed for the external test)')
    ap.add_argument('--seed', type=int, default=0, help='training seed index; the folds stay the same')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False   # the autotuner costs about a minute per run on this GPU and gains nothing
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    groups = C.load_groups(rows)
    fold = C.get_folds(y, groups)
    masks = np.load(os.path.join(C.DATA, 'masks.npy'))
    args.out = os.path.join(C.RESULTS, args.model + ('_' + args.tag if args.tag else ''))
    os.makedirs(args.out, exist_ok=True)
    print('device', dev, torch.cuda.get_device_name(0) if dev.type == 'cuda' else '', '| images', X.shape, '| classes', classes, flush=True)
    done = []
    for k in [int(v) for v in args.folds.split(',')]:
        path = os.path.join(args.out, 'fold%d.json' % k)
        if os.path.exists(path):
            print('fold', k, 'already done, skipping')
            done.append(json.load(open(path)))
            continue
        done.append(run_fold(args, k, X, y, fold, groups, masks, classes, dev))
    if not args.val_only and len(done) > 1:
        acc = [d['test']['accuracy'] for d in done]
        print('SUMMARY %s%s: test accuracy %.4f +/- %.4f over %d folds' % (args.model, '_' + args.tag if args.tag else '', np.mean(acc), np.std(acc, ddof=1), len(acc)))
    elif args.val_only:
        print('SUMMARY val-only', args.model, args.tag, [round(d['best_val_acc'], 4) for d in done], [d['best_epoch'] for d in done])


if __name__ == '__main__':
    main()
