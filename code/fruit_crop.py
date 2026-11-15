"""Locate the fruit in a photograph and cut a square crop around it.

Used for the external datasets, whose phone photos show the fruit at any position and size.
The fruit is found with a salient-object segmentation network (U2-Net, through the rembg package).
`fruit_crop(img, area=0.29)` returns a square crop in which the fruit covers the given fraction of the
frame, which is the median value in the main dataset, together with the mask in the crop and some
shape statistics of the object.
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('U2NET_HOME', os.path.join(os.path.dirname(HERE), '_env', 'u2net'))    # keep model files on this drive
_session = None


def session(name='u2net'):
    global _session
    if _session is None:
        from rembg import new_session
        _session = new_session(name)
    return _session


def fruit_mask(img, work=640):
    """img: PIL RGB. Returns a boolean mask (H x W) of the largest salient object, computed on a reduced copy."""
    import cv2
    from rembg import remove
    w, h = img.size
    s = work / max(w, h)
    small = img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR) if s < 1 else img
    m = np.asarray(remove(small, session=session(), only_mask=True)) > 127
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n > 1:
        m = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def shape_stats(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return {'area_frac': 0.0, 'elong': 0.0, 'cx': 0.5, 'cy': 0.5, 'area_px': int(len(xs))}
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    ev = np.linalg.eigvalsh(np.cov((pts - c).T))
    return {'area_frac': float(mask.mean()), 'elong': float(np.sqrt(ev[1] / max(ev[0], 1e-9))), 'cx': float(c[0] / mask.shape[1]),
            'cy': float(c[1] / mask.shape[0]), 'area_px': int(len(xs))}


def fruit_crop(img, mask=None, area=0.29, out=256):
    """Square crop centred on the fruit, sized so that the fruit covers `area` of it. Regions outside the photo are
    filled by replicating the border. Returns (crop RGB uint8 out x out, crop mask uint8, stats)."""
    import cv2
    if mask is None:
        mask = fruit_mask(img)
    st = shape_stats(mask)
    a = np.asarray(img)
    h, w = mask.shape
    side = int(round(np.sqrt(max(st['area_px'], 1) / area)))
    side = max(side, 16)
    cx, cy = st['cx'] * w, st['cy'] * h
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        # the photo is extended by mirroring, which continues the real background instead of smearing its last row
        a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        mk = np.pad(mask.astype(np.uint8), ((pad, pad), (pad, pad)), mode='constant')
    else:
        mk = mask.astype(np.uint8)
    crop = a[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    cm = mk[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    interp = cv2.INTER_AREA if side > out else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (out, out), interpolation=interp)
    cm = cv2.resize(cm, (out, out), interpolation=cv2.INTER_NEAREST)
    st['pad'] = int(pad)
    st['side'] = int(side)
    return crop, cm, st


def on_plain_background(crop, cm, colour=(223, 223, 223)):      # median table colour of the main dataset (table_colour.py)
    """The fruit of a fruit-centred crop on a plain light background (the table colour of the main dataset).
    The mask edge is feathered over a few pixels to avoid a hard outline."""
    import cv2
    m = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 1.2)[..., None]
    bg = np.array(colour, np.float32).reshape(1, 1, 3)
    return (crop.astype(np.float32) * m + bg * (1 - m)).round().clip(0, 255).astype(np.uint8)


def frame_crop(img, out=256):
    """Central square of the whole photograph, as used for the main dataset."""
    w, h = img.size
    s = min(w, h)
    box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
    return np.asarray(img.crop(box).resize((out, out), Image.LANCZOS))
