"""Fruit / background masks for the background-replacement robustness test, and near-duplicate groups.

Usage: python make_masks.py
Writes data_256/masks.npy (N x 256 x 256 uint8, 1 = fruit), data_256/mask_check.jpg (a contact sheet)
and prints a near-duplicate report based on the difference hashes in the manifest.

The background is the part of the image that is close in colour to the image border and connected
to the border. Highlights inside the fruit are therefore not mistaken for background.
"""
import os

import cv2
import numpy as np

import common as C


def fruit_mask(img, tol=28):
    """img: 256 x 256 x 3 uint8 RGB. Returns uint8 mask, 1 on the fruit."""
    border = np.concatenate([img[:6].reshape(-1, 3), img[-6:].reshape(-1, 3), img[:, :6].reshape(-1, 3), img[:, -6:].reshape(-1, 3)])
    ref = np.median(border, axis=0)
    blur = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.int16)
    cand = (np.abs(blur - ref).max(axis=2) < tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=4)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge[edge != 0]) if (cand[0].any() or cand[-1].any() or cand[:, 0].any() or cand[:, -1].any()) else np.zeros_like(cand, bool)
    bg &= cand.astype(bool)
    fg = (~bg).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)
    # fill holes
    inv = (1 - fg).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    outside = np.isin(lab, edge[edge != 0]) & inv.astype(bool)
    fg = (~outside).astype(np.uint8)
    return cv2.dilate(fg, np.ones((3, 3), np.uint8))


def main():
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    masks = np.stack([fruit_mask(x) for x in X])
    np.save(os.path.join(C.DATA, 'masks.npy'), masks)
    area = masks.reshape(len(masks), -1).mean(1)
    print('fruit area fraction: min %.3f  5%% %.3f  median %.3f  95%% %.3f  max %.3f' % (
        area.min(), np.percentile(area, 5), np.median(area), np.percentile(area, 95), area.max()))
    for c, name in enumerate(classes):
        print('  %-12s n=%3d  median area %.3f' % (name, (y == c).sum(), np.median(area[y == c])))
    odd = np.where((area < 0.03) | (area > 0.6))[0]
    print('suspicious masks (area < 3%% or > 60%%): %d' % len(odd), [rows[i]['file'] for i in odd[:12]])
    # contact sheet: 4 images per class, original next to the image on a dark background
    rng = np.random.default_rng(0)
    tiles = []
    for c in range(len(classes)):
        pick = rng.choice(np.where(y == c)[0], 4, replace=False)
        row = []
        for i in pick:
            m = masks[i][..., None]
            row += [X[i], (X[i] * m + np.array([60, 60, 60]) * (1 - m)).astype(np.uint8)]
        tiles.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(tiles, axis=0)
    cv2.imwrite(os.path.join(C.DATA, 'mask_check.jpg'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # near duplicates: pairs whose 64-bit difference hashes differ in at most 4 bits
    h = np.array([int(r['dhash'], 16) for r in rows], dtype=np.uint64)
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1).astype(np.int16)
    d = (bits[:, None, :] != bits[None, :, :]).sum(2)
    np.fill_diagonal(d, 99)
    for thr in (0, 2, 4, 6):
        pairs = np.argwhere(np.triu(d <= thr))
        same = sum(y[a] == y[b] for a, b in pairs)
        print('hash distance <= %d: %d pairs (%d within one class)' % (thr, len(pairs), same))
    pairs = np.argwhere(np.triu(d <= 2))[:8]
    print('examples:', [(rows[a]['file'], rows[b]['file'], int(d[a, b])) for a, b in pairs])
    sha = [r['sha1'] for r in rows]
    print('exact duplicate files:', len(sha) - len(set(sha)))


if __name__ == '__main__':
    main()
