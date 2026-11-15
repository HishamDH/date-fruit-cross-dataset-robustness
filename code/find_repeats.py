"""Look for repeat photographs of the same fruit in the main dataset.

Miski and Alshangiti (PLOS ONE, 2026) report 17 pairs of repeat photos in this dataset. The dataset has no
fruit identifiers, so this script ranks image pairs of the same class by the similarity of their features from
an ImageNet-pretrained network (no fine-tuning), writes the most similar pairs to a contact sheet for visual
checking, and saves the ranked list.

Usage: python find_repeats.py            # writes results/repeats_candidates.json and data_256/repeats_sheet.jpg
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import common as C

TOP = 60


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    import timm
    model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, num_classes=0).to(dev).eval()
    mean, std = model.pretrained_cfg['mean'], model.pretrained_cfg['std']
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            x = C.centre_crop(torch.from_numpy(X[i:i + 32]).to(dev).permute(0, 3, 1, 2).float() / 255)
            f = model(C.normalise(x, mean, std))
            # also the horizontally flipped view: a repeat photo may be mirrored
            feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
    Fm = np.concatenate(feats)
    S = Fm @ Fm.T
    np.fill_diagonal(S, -1)
    same = y[:, None] == y[None, :]
    S_same = np.where(same, S, -1)
    iu = np.triu_indices(len(y), 1)
    order = np.argsort(-S_same[iu])[:400]
    pairs = [(int(iu[0][o]), int(iu[1][o]), float(S_same[iu][o])) for o in order]
    fold = C.get_folds(y, C.load_groups(rows))
    out = [{'a': rows[a]['file'], 'b': rows[b]['file'], 'sim': s, 'ia': a, 'ib': b, 'same_fold': bool(fold[a] == fold[b])} for a, b, s in pairs]
    json.dump(out, open(os.path.join(C.RESULTS, 'repeats_candidates.json'), 'w'), indent=1)
    sims = np.array([p[2] for p in pairs])
    print('top similarities:', np.round(sims[:25], 4).tolist())
    print('pairs with sim >= 0.98: %d | >= 0.97: %d | >= 0.96: %d | >= 0.95: %d' % tuple(int((sims >= t).sum()) for t in (0.98, 0.97, 0.96, 0.95)))
    all_same = S_same[iu]
    print('distribution of same-class pair similarity: median %.3f, 99%% %.3f, 99.9%% %.3f, max %.3f' % (
        np.median(all_same[all_same > -1]), np.percentile(all_same[all_same > -1], 99), np.percentile(all_same[all_same > -1], 99.9), all_same.max()))
    T = 150
    sheet = Image.new('RGB', (6 * (2 * T + 8), (TOP // 6) * (T + 14)), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for i, (a, b, s) in enumerate(pairs[:TOP]):
        x0, y0 = (i % 6) * (2 * T + 8), (i // 6) * (T + 14)
        sheet.paste(Image.fromarray(X[a]).resize((T, T)), (x0, y0))
        sheet.paste(Image.fromarray(X[b]).resize((T, T)), (x0 + T, y0))
        dr.text((x0 + 2, y0 + T + 1), '#%d %.3f %s' % (i + 1, s, classes[y[a]]), fill=(255, 255, 255))
    sheet.save(os.path.join(C.DATA, 'repeats_sheet.jpg'), quality=88)
    print('saved sheet')


if __name__ == '__main__':
    main()
