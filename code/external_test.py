"""Test the networks trained on the main dataset on photographs from another dataset, without any retraining.

Usage: python external_test.py <dataset> [--recipes standard,rigid]
Uses the weights saved by run_seeds.py (results/<model>_s<seed>/fold<k>.pt and results/<model>_rigid_s<seed>/fold<k>.pt),
the working copies made by prepare_external.py, and the class mapping and fruit/pit rule in external_config.json.
Only the classes shared with the main dataset are tested; the networks still choose among their nine classes.
Writes results/external_<dataset>.json.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common as C


def load_external(name, cfg):
    root = os.path.join(C.ROOT, 'data_ext', name)
    recs = [json.loads(ln) for ln in open(os.path.join(root, 'manifest.jsonl'), encoding='utf-8')]
    excl = set(cfg.get('exclude', []))
    kind = json.load(open(os.path.join(root, 'kind.json'))) if cfg.get('kind_file') else None
    keep = []
    for r in recs:
        key = '%s/%s' % (r['label'], r['id'])
        if r['label'] not in cfg['map'] or key in excl:
            continue
        if kind is not None and (kind[key]['p_pit'] >= cfg.get('max_p_pit', 0.5) or r['area_frac'] > cfg.get('max_area', 1.0)):
            continue
        keep.append(r)
    return root, keep


def run_dirs(key, recipe):
    """Run folders of one model and recipe that hold the weights of all five folds."""
    import re
    mid = '' if recipe == 'standard' else '_' + recipe
    pat = re.compile(r'^%s%s_s\d+$' % (re.escape(key), mid))
    return [os.path.join(C.RESULTS, d) for d in sorted(os.listdir(C.RESULTS)) if pat.match(d) and os.path.isdir(os.path.join(C.RESULTS, d))
            and len(glob.glob(os.path.join(C.RESULTS, d, 'fold[0-9].pt'))) == 5]


def images(root, recs, variant):
    x = np.zeros((len(recs), C.SRC, C.SRC, 3), np.uint8)
    for i, r in enumerate(recs):
        with Image.open(os.path.join(root, variant, r['label'], r['id'] + '.jpg')) as im:
            x[i] = np.asarray(im.convert('RGB'))
    return x


@torch.no_grad()
def predict(model, X, mean, std, dev):
    out = []
    for i in range(0, len(X), 64):
        x = C.centre_crop(torch.from_numpy(X[i:i + 64]).to(dev).permute(0, 3, 1, 2).float() / 255)
        with torch.autocast('cuda', enabled=dev.type == 'cuda'):
            out.append(F.softmax(model(C.normalise(x, mean, std)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def score(y, prob, shared):
    pred = prob.argmax(1)
    rec = {int(c): float((pred[y == c] == c).mean()) for c in shared}
    return {'accuracy': float((pred == y).mean()), 'balanced_accuracy': float(np.mean(list(rec.values()))), 'recall': rec,
            'predicted_outside_shared': float((~np.isin(pred, shared)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--recipes', default='standard,rigid')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(C.HERE, 'external_config.json')))[a.dataset]
    rows, classes, y_main = C.load_manifest()
    root, recs = load_external(a.dataset, cfg)
    y = np.array([classes.index(cfg['map'][r['label']]) for r in recs])
    shared = sorted(set(y.tolist()))
    print(a.dataset, 'images', len(recs), 'shared classes', [classes[c] for c in shared], 'counts', np.bincount(y, minlength=len(classes)).tolist(), flush=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_path = os.path.join(C.RESULTS, 'external_%s.json' % a.dataset)
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out['_info'] = {'n': len(recs), 'classes': [classes[c] for c in shared], 'counts': {classes[c]: int((y == c).sum()) for c in shared},
                    'ext_labels': sorted({r['label'] for r in recs}), 'note': cfg.get('note'), 'ids': ['%s/%s' % (r['label'], r['id']) for r in recs]}
    for variant in ('frame', 'fruit', 'plain'):
        X = images(root, recs, variant)
        for recipe in a.recipes.split(','):
            # run folders: <model>_s<seed> (standard recipe), <model>_rigid_s<seed>, <model>_strong_s<seed>
            for key in (C.ALL_MODELS if recipe == 'standard' else C.MODELS):
                runs = sorted(d for d in run_dirs(key, recipe))
                tag = '%s|%s|%s' % (variant, recipe, key)
                if not runs or (tag in out and out[tag]['runs'] == [os.path.basename(r) for r in runs] and 'per_image' in out[tag]
                                and out[tag].get('n_images') == len(recs)):
                    continue
                model, mean, std = C.build_model(key, len(classes), pretrained=False)
                model.to(dev).eval()
                single, ens = [], []
                conf = np.zeros((len(classes), len(classes)), int)       # rows: true class, columns: predicted class, summed over models
                hits = np.zeros(len(recs))                               # per photograph: number of models that classify it correctly
                for run in runs:
                    probs = []
                    for k in range(5):
                        model.load_state_dict(torch.load(os.path.join(run, 'fold%d.pt' % k), map_location='cpu'))
                        p = predict(model, X, mean, std, dev)
                        probs.append(p)
                        single.append(score(y, p, shared))
                        np.add.at(conf, (y, p.argmax(1)), 1)
                        hits += (p.argmax(1) == y)
                    ens.append(score(y, np.mean(probs, 0), shared))
                acc = np.array([s['accuracy'] for s in single])
                bal = np.array([s['balanced_accuracy'] for s in single])
                out[tag] = {'runs': [os.path.basename(r) for r in runs], 'n_models': len(single), 'n_images': len(recs),
                            'accuracy': {'mean': float(acc.mean()), 'std': float(acc.std(ddof=1))},
                            'balanced_accuracy': {'mean': float(bal.mean()), 'std': float(bal.std(ddof=1))},
                            'recall': {classes[c]: float(np.mean([s['recall'][c] for s in single])) for c in shared},
                            'predicted_outside_shared': float(np.mean([s['predicted_outside_shared'] for s in single])),
                            'confusion': conf.tolist(), 'classes': classes, 'per_image': (hits / len(single)).round(4).tolist(),
                            'ensemble_accuracy': float(np.mean([e['accuracy'] for e in ens])),
                            'ensemble_balanced_accuracy': float(np.mean([e['balanced_accuracy'] for e in ens]))}
                print('%-28s acc %.3f +- %.3f  balanced %.3f  ensemble %.3f  recalls %s' % (
                    tag, acc.mean(), acc.std(ddof=1), bal.mean(), out[tag]['ensemble_accuracy'],
                    {k[:4]: round(v, 2) for k, v in out[tag]['recall'].items()}), flush=True)
                json.dump(out, open(out_path, 'w'), indent=1)
                del model
                torch.cuda.empty_cache()
    json.dump(out, open(out_path, 'w'), indent=1)
    print('EXTERNAL DONE', a.dataset, flush=True)


if __name__ == '__main__':
    main()
