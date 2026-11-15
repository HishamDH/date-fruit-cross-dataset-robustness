"""Collect all fold results into tables and figures.

Usage: python analyze.py
Reads results/<run>/fold*.json and fold*_pred.npz, writes results/summary.json, results/tables/*.csv
and figures/*.png. Every number in the paper comes from summary.json.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import common as C

FIG = os.path.join(C.ROOT, 'figures')
TAB = os.path.join(C.RESULTS, 'tables')
ORDER = ['svm', 'resnet50', 'mobilenetv3', 'efficientnetv2', 'convnext', 'deit', 'swin']
NAMES = dict({k: v['name'] for k, v in C.MODELS.items()}, svm='HOG+LBP+color SVM')
SHORT = dict({k: v['name'] for k, v in C.MODELS.items()}, svm='SVM')
COL = {'svm': '#7f7f7f', 'resnet50': '#1f77b4', 'mobilenetv3': '#17becf', 'efficientnetv2': '#2ca02c',
       'convnext': '#d62728', 'deit': '#9467bd', 'swin': '#ff7f0e'}
CORR_NAMES = {'darker': 'Darker', 'brighter': 'Brighter', 'low_contrast': 'Low contrast', 'colour_cast': 'Color cast',
              'noise': 'Noise', 'blur': 'Blur', 'jpeg': 'JPEG', 'background': 'Background'}
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8, 'axes.linewidth': 0.6, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'legend.frameon': False})
W1 = 3.45   # single-column width in inches


def load_run(run, root=None):
    d = os.path.join(root or C.RESULTS, run)
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, 'fold[0-9].json')))]
    preds = [np.load(p) for p in sorted(glob.glob(os.path.join(d, 'fold[0-9]_pred.npz')))]
    return folds, preds


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar_exact(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    p = 1.0 if b + c == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))
    return b, c, p


def seed_dirs(name, root=None):
    """Run folders of one configuration: the first run and its repetitions with other seeds (five finished folds each)."""
    root = root or C.RESULTS
    out = []
    for d in [name] + ['%s_s%d' % (name, s) for s in range(1, 10)]:
        p = os.path.join(root, d)
        if len(glob.glob(os.path.join(p, 'fold[0-9].json'))) == 5 and len(glob.glob(os.path.join(p, 'fold[0-9]_pred.npz'))) == 5:
            out.append(d)
    return out


def aggregate(dirs, S, n_classes):
    """Statistics of one configuration over all its runs (seeds x folds)."""
    runs = [S[d] for d in dirs]
    out = {'dirs': dirs, 'n_seeds': len(dirs), 'n_folds': int(sum(r['n_folds'] for r in runs))}
    for m in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
        v = np.concatenate([r[m]['folds'] for r in runs])
        out[m] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)), 'seed_means': [float(np.mean(r[m]['folds'])) for r in runs]}
    rob = {}
    for key in runs[0]['robustness']:
        v = np.concatenate([r['robustness'][key]['folds'] for r in runs])
        rob[key] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1))}
    out['robustness'] = rob
    out['corruption_mean'] = {c: float(np.mean([rob['%s_%d' % (c, s)]['mean'] for s in (1, 2, 3)])) for c in C.CORRUPTIONS}
    v = np.concatenate([r['mca_folds'] for r in runs])
    out['mean_corrupted_accuracy'] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)), 'seed_means': [float(np.mean(r['mca_folds'])) for r in runs]}
    n = runs[0]['pooled']['n']
    out['pooled'] = {'n': n, 'errors': float(np.mean([r['pooled']['errors'] for r in runs])), 'accuracy': float(np.mean([r['pooled']['accuracy'] for r in runs])),
                     'errors_per_seed': [r['pooled']['errors'] for r in runs]}
    cm = np.zeros((n_classes, n_classes), int)
    frac = np.zeros(n)
    for r in runs:
        pred = r['_prob'].argmax(1)
        cm += confusion_matrix(r['_y'], pred, labels=range(n_classes))
        frac[r['_idx']] += (pred == r['_y'])
    out['confusion'] = cm.tolist()
    out['_frac'] = frac / len(runs)                     # per image: share of the seeds in which it was classified correctly
    out['best_epoch'] = [e for r in runs for e in r['best_epoch']]
    out['val_curve'] = np.mean([r['_val_curve'] for r in runs if r['_val_curve'] is not None], 0).tolist() if runs[0]['_val_curve'] is not None else None
    return out


def per_class(cm, classes, n_seeds):
    cm = np.array(cm, float)
    out = {}
    for i, c in enumerate(classes):
        p = cm[i, i] / max(cm[:, i].sum(), 1)
        r = cm[i, i] / max(cm[i].sum(), 1)
        out[c] = {'precision': p, 'recall': r, 'f1-score': 2 * p * r / max(p + r, 1e-12), 'support': cm[i].sum() / n_seeds}
    out['macro avg'] = {k: float(np.mean([out[c][k] for c in classes])) for k in ('precision', 'recall', 'f1-score')}
    return out


def paired_test(fa, fb):
    """Wilcoxon signed-rank test on the per-image share of correct predictions of two configurations (same images)."""
    d = fa - fb
    if not np.any(d != 0):
        return {'a_better': 0, 'b_better': 0, 'p': 1.0}
    return {'a_better': int((d > 0).sum()), 'b_better': int((d < 0).sum()), 'p': float(stats.wilcoxon(d, zero_method='wilcox').pvalue)}


def summarise(run, n_total, root=None):
    folds, preds = load_run(run, root)
    if not folds or 'test' not in folds[0]:
        return None
    out = {'n_folds': len(folds)}
    for m in ('accuracy', 'precision', 'recall', 'f1', 'auc'):
        v = np.array([f['test'][m] for f in folds])
        out[m] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0, 'folds': v.tolist()}
    idx = np.concatenate([p['idx'] for p in preds])
    yt = np.concatenate([p['y'] for p in preds])
    prob = np.concatenate([p['prob'] for p in preds])
    pred = prob.argmax(1)
    k, n = int((pred == yt).sum()), len(yt)
    lo, hi = wilson(k, n)
    out['pooled'] = {'correct': k, 'n': n, 'accuracy': k / n, 'ci95': [float(lo), float(hi)], 'errors': n - k,
                     'covers_all_images': bool(n == n_total and len(set(idx.tolist())) == n_total)}
    rob = {}
    for key in folds[0].get('robustness', {}):
        v = np.array([f['robustness'][key]['accuracy'] for f in folds])
        rob[key] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0, 'folds': v.tolist()}
    out['robustness'] = rob
    if rob:
        out['corruption_mean'] = {c: float(np.mean([rob['%s_%d' % (c, s)]['mean'] for s in (1, 2, 3)])) for c in C.CORRUPTIONS}
        per_fold = np.array([[f['robustness']['%s_%d' % (c, s)]['accuracy'] for c in C.CORRUPTIONS for s in (1, 2, 3)] for f in folds]).mean(1)
        out['mean_corrupted_accuracy'] = {'mean': float(per_fold.mean()), 'std': float(per_fold.std(ddof=1)) if len(per_fold) > 1 else 0.0}
        out['mca_folds'] = per_fold.tolist()
    out['best_epoch'] = [f.get('best_epoch') for f in folds]
    out['_val_curve'] = np.mean([[h['val_acc'] for h in f['history']] for f in folds], 0) if 'history' in folds[0] else None
    out['_hist'] = [f['history'] for f in folds] if 'history' in folds[0] else None
    out['train_seconds'] = {'mean': float(np.mean([f['train_seconds'] for f in folds])), 'total': float(np.sum([f['train_seconds'] for f in folds]))}
    out['_idx'], out['_y'], out['_prob'] = idx, yt, prob
    return out


def fig_samples(X, y, classes, rows):
    counts = np.bincount(y)
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(3, 3, figsize=(W1, W1 * 1.08))
    for c, a in enumerate(ax.ravel()):
        i = rng.choice(np.where(y == c)[0])
        a.imshow(X[i])
        a.set_title('%s (%d)' % (classes[c], counts[c]), fontsize=7.5, pad=2)
        a.axis('off')
    fig.subplots_adjust(wspace=0.04, hspace=0.16)
    fig.savefig(os.path.join(FIG, 'fig_samples.png'))
    plt.close(fig)


def fig_corruptions(X, masks, y, classes):
    i = int(np.where(y == classes.index(classes[0]))[0][5])
    rng = np.random.default_rng(11)
    i = int(rng.choice(len(y)))
    x = C.centre_crop(torch.from_numpy(X[[i]]).permute(0, 3, 1, 2).float() / 255)
    m = C.centre_crop(torch.from_numpy(masks[[i]]).float().unsqueeze(1))
    fig, ax = plt.subplots(3, 3, figsize=(W1, W1 * 1.08))
    items = [('Clean', x)] + [(CORR_NAMES[k], C.corrupt(x, k, 3, mask=m, gen=torch.Generator().manual_seed(1))) for k in C.CORRUPTIONS]
    for a, (t, im) in zip(ax.ravel(), items):
        a.imshow(im[0].permute(1, 2, 0).numpy())
        a.set_title(t, fontsize=7.5, pad=2)
        a.axis('off')
    fig.subplots_adjust(wspace=0.04, hspace=0.16)
    fig.savefig(os.path.join(FIG, 'fig_corruptions.png'))
    plt.close(fig)


def fig_curves(G, S, runs):
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.5), sharex=True)
    for r in runs:
        hist = [h for d in G[r]['dirs'] for h in S[d]['_hist']]          # every fold of every seed
        ep = np.arange(1, len(hist[0]) + 1)
        tl = np.mean([[e['train_loss'] for e in h] for h in hist], 0)
        vl = np.mean([[e['val_loss'] for e in h] for h in hist], 0)
        va = np.mean([[e['val_acc'] for e in h] for h in hist], 0)
        ax[0].plot(ep, vl, color=COL[r], lw=1.1, label=SHORT[r])
        ax[0].plot(ep, tl, color=COL[r], lw=0.8, ls=':')
        ax[1].plot(ep, 100 * va, color=COL[r], lw=1.1)
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Validation accuracy (%)')
    ax[1].set_xlabel('Epoch')
    ax[0].set_yscale('log')
    ax[0].legend(ncol=2, fontsize=6.5, loc='upper right', handlelength=1.4, columnspacing=0.8)
    for a in ax:
        a.grid(alpha=0.25, lw=0.4)
    fig.subplots_adjust(hspace=0.08)
    fig.savefig(os.path.join(FIG, 'fig_curves.png'))
    plt.close(fig)


def fig_confusion(S, runs, classes, name):
    fig, ax = plt.subplots(1, len(runs), figsize=(W1 * (2.02 if len(runs) > 1 else 1), W1 * 0.98), squeeze=False)
    for a, r in zip(ax[0], runs):
        cm = np.array(S[r]['confusion'])                               # summed over the seeds
        a.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    a.text(j, i, cm[i, j], ha='center', va='center', fontsize=6.5, color='white' if cm[i, j] > cm.max() * 0.55 else 'black')
        a.set_xticks(range(len(classes)), classes, rotation=55, ha='right', fontsize=6.5)
        a.set_yticks(range(len(classes)), classes, fontsize=6.5)
        a.set_xlabel('Predicted')
        a.set_title(NAMES[r], fontsize=8, pad=3)
    ax[0][0].set_ylabel('True')
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def fig_robust_heat(S, runs):
    M = np.array([[100 * S[r]['corruption_mean'][c] for c in C.CORRUPTIONS] for r in runs])
    fig, ax = plt.subplots(figsize=(W1, 2.25))
    im = ax.imshow(M, cmap='viridis', vmin=10, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, '%.1f' % M[i, j], ha='center', va='center', fontsize=6.3, color='white' if M[i, j] < 70 else 'black')
    ax.set_xticks(range(len(C.CORRUPTIONS)), [CORR_NAMES[c] for c in C.CORRUPTIONS], rotation=40, ha='right', fontsize=6.8)
    ax.set_yticks(range(len(runs)), [SHORT[r] for r in runs], fontsize=6.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(FIG, 'fig_robust_heat.png'))
    plt.close(fig)


def fig_severity(S, runs, kinds):
    MK = {'svm': 'x', 'resnet50': 'o', 'mobilenetv3': 'v', 'efficientnetv2': 's', 'convnext': 'D', 'deit': '^', 'swin': 'P'}
    fig, ax = plt.subplots(2, 2, figsize=(W1, 3.25), sharey=True, sharex=True)
    for a, c in zip(ax.ravel(), kinds):
        for r in runs:
            v = [100 * S[r]['accuracy']['mean']] + [100 * S[r]['robustness']['%s_%d' % (c, s)]['mean'] for s in (1, 2, 3)]
            a.plot(range(4), v, marker=MK[r], ms=2.8, lw=0.9, color=COL[r], label=SHORT[r])
        a.set_title(CORR_NAMES[c], fontsize=7.5, pad=2)
        a.set_xticks(range(4), ['clean', '1', '2', '3'])
        a.set_ylim(0, 103)
        a.grid(alpha=0.25, lw=0.4)
    for a in ax[1]:
        a.set_xlabel('Severity level')
    for a in ax[:, 0]:
        a.set_ylabel('Accuracy (%)')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, 1.07), handlelength=1.8, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.06, hspace=0.2)
    fig.savefig(os.path.join(FIG, 'fig_severity.png'))
    plt.close(fig)


def fig_tradeoff(S, eff, runs):
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    for r in runs:
        if r not in eff:
            continue
        x = eff[r]['gpu_ms_b1'][0] if 'gpu_ms_b1' in eff[r] else eff[r]['cpu_ms_b1'][0]
        ax.scatter(x, 100 * S[r]['mean_corrupted_accuracy']['mean'], s=18 + 6 * eff[r]['params_m'], color=COL[r], alpha=0.85, edgecolor='k', lw=0.4)
        off = {'deit': (7, 5, 'left'), 'convnext': (0, -13, 'center'), 'swin': (8, 3, 'left'), 'efficientnetv2': (-7, 7, 'right'),
               'resnet50': (7, 4, 'left'), 'mobilenetv3': (6, 3, 'left')}[r]
        ax.annotate(SHORT[r], (x, 100 * S[r]['mean_corrupted_accuracy']['mean']), textcoords='offset points', xytext=off[:2], ha=off[2], fontsize=6.5)
    ax.set_xlabel('GPU latency per image (ms), batch size 1')
    ax.set_ylabel('Mean accuracy on corrupted images (%)')
    ax.grid(alpha=0.25, lw=0.4)
    ax.margins(x=0.18, y=0.15)
    fig.savefig(os.path.join(FIG, 'fig_tradeoff.png'))
    plt.close(fig)


def fig_roc(S, r, classes):
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    for c, name in enumerate(classes):
        fpr, tpr, _ = roc_curve(S[r]['_y'] == c, S[r]['_prob'][:, c])
        ax.plot(fpr, tpr, lw=0.9, label='%s (%.3f)' % (name, auc(fpr, tpr)))
    ax.plot([0, 1], [0, 1], color='gray', lw=0.5, ls='--')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend(fontsize=6, title='Class (AUC)', title_fontsize=6.5, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(os.path.join(FIG, 'fig_roc_%s.png' % r))
    plt.close(fig)


def fig_gradcam(X, y, classes, keys):
    have = [k for k in keys if os.path.exists(os.path.join(C.RESULTS, 'gradcam_%s.npz' % k))]
    if not have:
        return
    data = {k: np.load(os.path.join(C.RESULTS, 'gradcam_%s.npz' % k)) for k in have}
    te = data[have[0]]['idx']
    rng = np.random.default_rng(5)
    pick = []
    for c in rng.permutation(len(classes))[:5]:
        cand = np.where(y[te] == c)[0]
        pick.append(int(rng.choice(cand)))
    fig, ax = plt.subplots(len(have) + 1, len(pick), figsize=(W1, W1 * (len(have) + 1) / len(pick) * 1.07))
    for j, p in enumerate(pick):
        img = X[te[p]][16:240, 16:240]
        ax[0, j].imshow(img)
        ax[0, j].set_title(classes[y[te[p]]], fontsize=6.5, pad=1.5)
        for i, k in enumerate(have):
            ax[i + 1, j].imshow(img)
            ax[i + 1, j].imshow(data[k]['cam'][p].astype(np.float32), cmap='jet', alpha=0.45, vmin=0, vmax=1)
    short = {'resnet50': 'ResNet-50', 'mobilenetv3': 'MobileNetV3', 'efficientnetv2': 'EffNetV2-S', 'convnext': 'ConvNeXt-T', 'deit': 'DeiT-S', 'swin': 'Swin-T'}
    for i, lab in enumerate(['Input'] + [short[k] for k in have]):
        ax[i, 0].text(-0.08, 0.5, lab, transform=ax[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=5.6)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(os.path.join(FIG, 'fig_gradcam.png'))
    plt.close(fig)


EXT_NAMES = {'medina27': 'Medina 27', 'soda': 'SODA'}


def extra_results(classes, G):
    """Second dataset, external tests, multi-source training and the repeat-photo check, if their result files exist."""
    out = {}
    # --- SODA as a second dataset (its own five-fold runs)
    root = os.path.join(C.ROOT, 'results_soda')
    info_p = os.path.join(C.ROOT, 'data_soda_256', 'info.json')
    if os.path.exists(info_p):
        out['soda_info'] = json.load(open(info_p))
    if os.path.isdir(root) and os.path.exists(info_p):
        info = json.load(open(info_p))
        runs = {}
        n_cls = len(info['per_class'])
        for name in ORDER:
            dirs = seed_dirs(name, root)
            if dirs:
                S2 = {d: summarise(d, info['kept'], root) for d in dirs}
                runs[name] = {k: v for k, v in aggregate(dirs, S2, n_cls).items() if not k.startswith('_')}
        out['second'] = {'name': 'SODA', 'info': info, 'runs': runs}
    # --- external tests: networks trained on the main dataset, tested on photographs from other datasets
    ext = {}
    for name in EXT_NAMES:
        p = os.path.join(C.RESULTS, 'external_%s.json' % name)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        info = R['_info']
        shared = [classes.index(c) for c in info['classes']]
        table, dest = {}, {}
        for tag, v in R.items():
            if tag == '_info' or 'confusion' not in v:
                continue
            variant, recipe, model = tag.split('|')
            cm = np.array(v['confusion'], float)
            keep = [c for c in shared if classes[c] != 'Galaxy']
            entry = {'accuracy': v['accuracy'], 'recall': v['recall'], 'n_models': v['n_models'], 'ensemble_accuracy': v['ensemble_accuracy'],
                     'accuracy_without_galaxy': float(sum(cm[c, c] for c in keep) / max(sum(cm[c].sum() for c in keep), 1)),
                     'accuracy_without_ajwa': float(sum(cm[c, c] for c in shared if classes[c] != 'Ajwa') / max(sum(cm[c].sum() for c in shared if classes[c] != 'Ajwa'), 1))}
            table.setdefault(recipe, {}).setdefault(variant, {})[model] = entry
            if recipe == 'standard' and model in C.MODELS:
                dest[variant] = dest.get(variant, 0) + cm
        ext[name] = {'info': info, 'table': table,
                     'destination': {v: (100 * d / np.maximum(d.sum(1, keepdims=True), 1)).round(1).tolist() for v, d in dest.items()}}
        # does the ranking under simulated corruption predict the ranking on real photographs?
        rank = {}
        for variant, tab in table.get('standard', {}).items():
            ms = [m for m in C.MODELS if m in tab and m in G]
            if len(ms) == len(C.MODELS):
                e = [tab[m]['accuracy']['mean'] for m in ms]
                rank[variant] = {'spearman_clean': float(stats.spearmanr([G[m]['accuracy']['mean'] for m in ms], e)[0]),
                                 'spearman_mca': float(stats.spearmanr([G[m]['mean_corrupted_accuracy']['mean'] for m in ms], e)[0])}
        ext[name]['rank'] = rank
        if name == 'medina27' and 'ids' in info:
            # the file names of Medina 27 encode the capture setting and the view (dataset manual)
            def setting(i):
                f = i.split('/')[1].split('.')
                code = f[1] if len(f) > 1 else ''
                return 'aged' if code.startswith('o') else ('ordinary' if code.startswith('r') else ('enclosure' if code.startswith('s') else 'other'))

            def view(i):
                last = i.split('/')[1].split('.')[-1]
                return 'stem end' if last.endswith('f') else ('apex' if last.endswith('b') else 'side or unmarked')
            groups = {}
            for fn, key in ((setting, 'by_setting'), (view, 'by_view')):
                lab = np.array([fn(i) for i in info['ids']])
                res = {}
                for variant, tab in table.get('standard', {}).items():
                    per = np.mean([R['%s|standard|%s' % (variant, m)]['per_image'] for m in C.MODELS if '%s|standard|%s' % (variant, m) in R
                                   and 'per_image' in R['%s|standard|%s' % (variant, m)]], 0)
                    if np.ndim(per) == 1:
                        res[variant] = {g: {'n': int((lab == g).sum()), 'accuracy': float(per[lab == g].mean())} for g in sorted(set(lab))}
                groups[key] = res
            ext[name].update(groups)
    out['external'] = ext
    # --- test-time remedies on the external photographs
    out['tta'] = {}
    for name in EXT_NAMES:
        p = os.path.join(C.RESULTS, 'external_tta_%s.json' % name)
        if os.path.exists(p):
            out['tta'][name] = json.load(open(p))
    # --- multi-source training
    ms = {}
    for p in sorted(glob.glob(os.path.join(C.RESULTS, 'multisource', '*.json'))):
        d = json.load(open(p))
        ms.setdefault('%s_to_%s' % (d['extra'], d['target']), {}).setdefault(d['model'], []).append(d)
    out['multisource'] = {}
    for key, per_model in ms.items():
        out['multisource'][key] = {}
        for model, runs in per_model.items():
            e = {'n': len(runs), 'n_train_extra': runs[0]['n_train_extra'], 'val_acc_main': float(np.mean([r['val_acc_main'] for r in runs]))}
            for variant in ('frame', 'plain'):
                a = np.array([r[variant]['accuracy'] for r in runs])
                e[variant] = {'mean': float(a.mean()), 'std': float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                              'recall': {c: float(np.mean([r[variant]['recall'][c] for r in runs])) for c in runs[0][variant]['recall']}}
            out['multisource'][key][model] = e
    rp = os.path.join(C.RESULTS, 'repeats_effect.json')
    out['repeats'] = json.load(open(rp)) if os.path.exists(rp) else {}
    return out


def fig_external_samples(X, y, classes, names=('Ajwa', 'Galaxy', 'Medjool', 'Shaishe', 'Sugaey')):
    """One photograph per shared class: main dataset, then each external dataset as photographed and after normalisation."""
    from PIL import Image
    import external_test as E
    cfg = json.load(open(os.path.join(C.HERE, 'external_config.json')))
    rows_img, labels = [], []
    rng = np.random.default_rng(7)
    rows_img.append([X[int(rng.choice(np.where(y == classes.index(c))[0]))] for c in names])
    labels.append('Main')
    for name in EXT_NAMES:
        if not os.path.exists(os.path.join(C.ROOT, 'data_ext', name, 'manifest.jsonl')):
            continue
        root, recs = E.load_external(name, cfg[name])
        inv = {v: k for k, v in cfg[name]['map'].items()}
        pick = []
        for c in names:
            rr = sorted((r for r in recs if r['label'] == inv[c]), key=lambda r: (r.get('area_frac', 0), r['id']))
            pick.append(rr[len(rr) // 2])          # the photograph with the median fruit size of its class
        for variant, lab in (('frame', 'photo'), ('plain', 'normalized')):
            rows_img.append([np.asarray(Image.open(os.path.join(root, variant, r['label'], r['id'] + '.jpg')).convert('RGB')) for r in pick])
            labels.append('%s\n(%s)' % (EXT_NAMES[name], lab))
    if len(rows_img) < 2:
        return
    fig, ax = plt.subplots(len(rows_img), len(names), figsize=(W1, W1 * len(rows_img) / len(names) * 1.04))
    for i, row in enumerate(rows_img):
        for j, im in enumerate(row):
            ax[i, j].imshow(im)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_visible(False)
            if i == 0:
                ax[i, j].set_title(names[j], fontsize=7, pad=2)
        ax[i, 0].text(-0.07, 0.5, labels[i], transform=ax[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=5.6, linespacing=0.9)
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    fig.savefig(os.path.join(FIG, 'fig_external_samples.png'))
    plt.close(fig)


def fig_external(G, ext, runs):
    """Accuracy of each network: own test folds, corrupted test folds, and photographs from the two other datasets."""
    series = [('Clean test folds', [100 * G[r]['accuracy']['mean'] for r in runs], '#4c72b0'),
              ('Corrupted test folds (mCA)', [100 * G[r]['mean_corrupted_accuracy']['mean'] for r in runs], '#8da0cb')]
    for name, col in (('medina27', '#dd8452'), ('soda', '#c44e52')):
        tab = ext.get(name, {}).get('table', {}).get('standard', {}).get('plain', {})
        if all(r in tab for r in runs):
            series.append(('%s photographs' % EXT_NAMES[name], [100 * tab[r]['accuracy']['mean'] for r in runs], col))
    if len(series) < 3:
        return
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    w = 0.8 / len(series)
    for i, (lab, v, col) in enumerate(series):
        ax.bar(np.arange(len(runs)) + (i - (len(series) - 1) / 2) * w, v, w, label=lab, color=col, edgecolor='k', lw=0.3)
    ax.set_xticks(range(len(runs)), [SHORT[r].replace('EfficientNetV2-S', 'Efficient-\nNetV2-S').replace('MobileNetV3-L', 'Mobile-\nNetV3-L') for r in runs],
                  fontsize=5.8, linespacing=0.95)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    ax.legend(ncol=2, fontsize=6, loc='upper center', handlelength=1.2, columnspacing=0.8)
    fig.savefig(os.path.join(FIG, 'fig_external.png'))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    masks = np.load(os.path.join(C.DATA, 'masks.npy'))
    S = {}
    for run in sorted(os.listdir(C.RESULTS)):
        if os.path.isdir(os.path.join(C.RESULTS, run)) and run != 'tables':
            s = summarise(run, len(y))
            if s:
                S[run] = s
    # every configuration is summarised over all its seeds: G[name] combines the run folders name, name_s1, name_s2, ...
    logical = (ORDER + [m + '_rigid' for m in C.MODELS] + ['convnext_noaug', 'convnext_scratch'] + list(C.EXTRA_MODELS)
               + [m + '_strong' for m in C.MODELS])
    G = {}
    for name in logical:
        dirs = [d for d in seed_dirs(name) if d in S]
        if dirs:
            G[name] = aggregate(dirs, S, len(classes))
    main_runs = [r for r in ORDER if r in G]
    deep = [r for r in main_runs if r != 'svm']
    best = max(deep, key=lambda r: (G[r]['accuracy']['mean'], G[r]['mean_corrupted_accuracy']['mean'])) if deep else None
    summary = {'classes': classes, 'class_counts': np.bincount(y).tolist(), 'n_images': int(len(y)), 'best_clean': best, 'runs': {}, 'seed_runs': {}}
    for r, s in S.items():
        summary['seed_runs'][r] = {k: v for k, v in s.items() if not k.startswith('_')}
    for r, g in G.items():
        summary['runs'][r] = {k: v for k, v in g.items() if not k.startswith('_')}
    summary['n_seeds'] = {r: G[r]['n_seeds'] for r in G}
    if best:
        # paired tests against the most accurate network: Wilcoxon signed-rank test on the per-image share of correct predictions
        summary['tests_vs_best'] = {r: paired_test(G[best]['_frac'], G[r]['_frac']) for r in main_runs if r != best}
        summary['per_class_best'] = per_class(G[best]['confusion'], classes, G[best]['n_seeds'])
        if 'svm' in G:
            summary['per_class_svm'] = per_class(G['svm']['confusion'], classes, G['svm']['n_seeds'])
        summary['ablation_tests'] = {k: paired_test(G[k]['_frac'], G['convnext']['_frac']) for k in ('convnext_noaug', 'convnext_rigid') if k in G and 'convnext' in G}
    summary.update(extra_results(classes, G))
    eff_path = os.path.join(C.RESULTS, 'efficiency.json')
    eff = json.load(open(eff_path)) if os.path.exists(eff_path) else {}
    summary['efficiency'] = eff
    gc_path = os.path.join(C.RESULTS, 'gradcam.json')
    summary['gradcam'] = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    summary['fruit_area_median'] = float(np.median(masks.reshape(len(masks), -1).mean(1)))
    if summary['gradcam']:
        summary['gradcam_area'] = float(np.mean([v['fruit_area_fraction_mean'] for v in summary['gradcam'].values()]))
    if deep:
        f0 = json.load(open(os.path.join(C.RESULTS, deep[0], 'fold0.json')))
        summary['epochs'] = f0['args']['epochs']
        summary['split_sizes'] = {'train': f0['n_train'], 'val': f0['n_val'], 'test': f0['n_test']}
    if '_machine' in eff:
        mch = dict(eff['_machine'])
        mch['cpu_short'] = mch['cpu'].replace('(R)', '').replace('(TM)', '').replace('  ', ' ').strip()
        summary['machine'] = mch
    lr_path = os.path.join(C.RESULTS, 'lr_choice.json')
    summary['lr_choice'] = json.load(open(lr_path)) if os.path.exists(lr_path) else {}
    json.dump(summary, open(os.path.join(C.RESULTS, 'summary.json'), 'w'), indent=1)
    # tables as CSV
    with open(os.path.join(TAB, 'main_results.csv'), 'w') as fh:
        fh.write('configuration,seeds,fold_runs,acc_mean,acc_std,prec,rec,f1,auc,errors_per_seed,mean_corrupted_acc,mean_corrupted_std\n')
        for r, s in G.items():
            fh.write('%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.4f,%.4f\n' % (
                r, s['n_seeds'], s['n_folds'], s['accuracy']['mean'], s['accuracy']['std'], s['precision']['mean'], s['recall']['mean'], s['f1']['mean'],
                s['auc']['mean'], s['pooled']['errors'], s['mean_corrupted_accuracy']['mean'], s['mean_corrupted_accuracy']['std']))
    with open(os.path.join(TAB, 'robustness.csv'), 'w') as fh:
        fh.write('configuration,' + ','.join('%s_%d' % (c, s) for c in C.CORRUPTIONS for s in (1, 2, 3)) + '\n')
        for r in G:
            fh.write(r + ',' + ','.join('%.4f' % G[r]['robustness']['%s_%d' % (c, s)]['mean'] for c in C.CORRUPTIONS for s in (1, 2, 3)) + '\n')
    # figures
    fig_samples(X, y, classes, rows)
    fig_corruptions(X, masks, y, classes)
    if deep:
        fig_curves(G, S, deep)
        fig_confusion(G, [best], classes, 'fig_confusion_best.png')
        fig_robust_heat(G, main_runs)
        fig_severity(G, main_runs, ['noise', 'blur', 'jpeg', 'darker'])
        if eff:
            fig_tradeoff(G, eff, deep)
        fig_gradcam(X, y, classes, deep)
        fig_external_samples(X, y, classes)
        fig_external(G, summary.get('external', {}), deep)
    for r, s in G.items():
        print('%-18s seeds %d  acc %.2f +/- %.2f  f1 %.2f  errors/seed %.1f  mCA %.2f +/- %.2f  seed means %s' % (
            r, s['n_seeds'], 100 * s['accuracy']['mean'], 100 * s['accuracy']['std'], 100 * s['f1']['mean'], s['pooled']['errors'],
            100 * s['mean_corrupted_accuracy']['mean'], 100 * s['mean_corrupted_accuracy']['std'], [round(100 * v, 2) for v in s['accuracy']['seed_means']]))
    print('best on clean images:', best, '| tests vs best:', {k: round(v['p'], 4) for k, v in summary.get('tests_vs_best', {}).items()})
    for name, e in summary.get('external', {}).items():
        print('external', name, e['rank'], {v: {m: round(100 * t['accuracy']['mean'], 1) for m, t in tab.items()} for v, tab in e['table'].get('standard', {}).items()})
    for key, e in summary.get('multisource', {}).items():
        print('multisource', key, {m: (v['n'], round(100 * v['frame']['mean'], 1), round(100 * v['plain']['mean'], 1)) for m, v in e.items()})
    if 'second' in summary:
        print('second dataset:', {m: (round(100 * v['accuracy']['mean'], 2), round(100 * v['mean_corrupted_accuracy']['mean'], 2)) for m, v in summary['second']['runs'].items()})


if __name__ == '__main__':
    main()
