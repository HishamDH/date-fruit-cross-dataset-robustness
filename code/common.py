"""Shared pieces of the date-fruit experiments: data, folds, augmentation, corruptions, models, metrics."""
import csv
import io
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# The main dataset lives in data_256/ and results/. Setting the environment variable DATES_DATASET
# (for example to "soda") switches every script to data_<name>_256/ and results_<name>/.
DATASET = os.environ.get('DATES_DATASET', '')
DATA = os.path.join(ROOT, 'data_256' if not DATASET else 'data_%s_256' % DATASET)
RESULTS = os.path.join(ROOT, 'results' if not DATASET else 'results_%s' % DATASET)
MAIN_RESULTS = os.path.join(ROOT, 'results')
SRC = 256          # side of the stored images
SIZE = 224         # network input side
K_FOLDS = 5
FOLD_SEED = 42

# ImageNet-1k weights only, so that no model has an advantage from larger pretraining data.
MODELS = {
    'resnet50': dict(tag='resnet50.tv_in1k', name='ResNet-50', family='CNN'),
    'mobilenetv3': dict(tag='mobilenetv3_large_100.ra_in1k', name='MobileNetV3-L', family='CNN'),
    'efficientnetv2': dict(tag='tf_efficientnetv2_s.in1k', name='EfficientNetV2-S', family='CNN'),
    'convnext': dict(tag='convnext_tiny.fb_in1k', name='ConvNeXt-T', family='CNN'),
    'deit': dict(tag='deit_small_patch16_224.fb_in1k', name='DeiT-S', family='Transformer'),
    'swin': dict(tag='swin_tiny_patch4_window7_224.ms_in1k', name='Swin-T', family='Transformer'),
}
# Checkpoints pretrained on more data than ImageNet-1k, for the question whether larger pretraining closes the cross-dataset gap.
# They are kept out of MODELS so that the main comparison stays a comparison of ImageNet-1k checkpoints.
EXTRA_MODELS = {
    'convnext_in22k': dict(tag='convnext_tiny.fb_in22k', name='ConvNeXt-T (IN-22k)', family='CNN', data='ImageNet-22k, supervised'),
    'vit_in21k': dict(tag='vit_small_patch16_224.augreg_in21k', name='ViT-S/16 (IN-21k)', family='Transformer', data='ImageNet-21k, supervised'),
    'dinov2': dict(tag='vit_small_patch14_dinov2.lvd142m', name='DINOv2 ViT-S/14', family='Transformer', data='LVD-142M, self-supervised',
                   kwargs=dict(img_size=224)),
    'clip': dict(tag='vit_base_patch16_clip_224.openai', name='CLIP ViT-B/16', family='Transformer', data='400M image-text pairs',
                 grad_checkpointing=True),
}
ALL_MODELS = dict(MODELS, **EXTRA_MODELS)


# ----------------------------------------------------------------------------- data
def load_manifest():
    rows = list(csv.DictReader(open(os.path.join(DATA, 'manifest.csv'), encoding='utf-8')))
    classes = sorted({r['label'] for r in rows})
    y = np.array([classes.index(r['label']) for r in rows])
    return rows, classes, y


def load_images(rows):
    """All images as one uint8 array, N x 256 x 256 x 3 (about 330 MB for this dataset)."""
    cache = os.path.join(DATA, 'images_uint8.npy')
    if os.path.exists(cache):
        x = np.load(cache)
        if len(x) == len(rows):
            return x
    x = np.zeros((len(rows), SRC, SRC, 3), np.uint8)
    for i, r in enumerate(rows):
        with Image.open(os.path.join(DATA, r['file'])) as im:
            x[i] = np.asarray(im.convert('RGB'))
    np.save(cache, x)
    return x


def get_folds(y, groups=None):
    """Stratified k-fold test assignment, stored once so every script uses the same folds.

    If `groups` is given (near-duplicate clusters), images of one group never straddle two folds.
    """
    path = os.path.join(RESULTS, 'folds.json')
    if os.path.exists(path):
        fold = np.array(json.load(open(path))['fold'])
        assert len(fold) == len(y)
        return fold
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    fold = np.zeros(len(y), int)
    if groups is None:
        sp = StratifiedKFold(K_FOLDS, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(y)), y)
    else:
        sp = StratifiedGroupKFold(K_FOLDS, shuffle=True, random_state=FOLD_SEED).split(np.zeros(len(y)), y, groups)
    for k, (_, te) in enumerate(sp):
        fold[te] = k
    os.makedirs(RESULTS, exist_ok=True)
    json.dump({'k': K_FOLDS, 'seed': FOLD_SEED, 'grouped': groups is not None, 'fold': fold.tolist()}, open(path, 'w'))
    return fold


def split_fold(y, fold, k, groups=None, val_frac=0.125):
    """Indices for fold k: test = fold k; the rest is split into train and validation (stratified)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
    test = np.where(fold == k)[0]
    rest = np.where(fold != k)[0]
    if groups is None:
        sp = StratifiedShuffleSplit(1, test_size=val_frac, random_state=FOLD_SEED + 1 + k)
        tr, va = next(sp.split(rest, y[rest]))
    else:
        sp = StratifiedGroupKFold(int(round(1 / val_frac)), shuffle=True, random_state=FOLD_SEED + 1 + k)
        tr, va = next(sp.split(rest, y[rest], groups[rest]))
    return rest[tr], rest[va], test


def load_groups(rows):
    if rows and 'group' in rows[0]:
        return np.array([int(r['group']) for r in rows])
    return None


# ----------------------------------------------------------------------------- augmentation (on the GPU, per sample)
def centre_crop(x):
    o = (SRC - SIZE) // 2
    return x[:, :, o:o + SIZE, o:o + SIZE]


def augment(x, gen, rot_deg=20.0, area=(0.6, 1.0), jitter=0.2, rigid=False, strong=False, mask=None):
    """Random resized crop, rotation, horizontal and vertical flip, brightness and contrast jitter.

    x: float tensor N x 3 x 256 x 256 in [0, 1]. Colour (hue, saturation) is left untouched because
    colour is a class cue for date varieties.
    rigid=True keeps the scale and the aspect ratio of the test crop (a 224 x 224 window that is only
    shifted, rotated and flipped), so the size and the shape of the fruit are preserved.
    strong=True adds what a change of camera and setting may bring: a random background colour behind the
    fruit (needs mask, N x 1 x 256 x 256, 1 on the fruit), wider brightness and contrast ranges, a random
    white-balance gain per channel, a random saturation, and blur or sensor noise on part of the batch.
    """
    n, dev = x.shape[0], x.device

    def u(lo, hi):
        return torch.rand(n, device=dev, generator=gen) * (hi - lo) + lo

    a = u(*area)
    logr = u(math.log(3 / 4), math.log(4 / 3))
    w = torch.sqrt(a * torch.exp(logr)).clamp(max=1.0)
    h = torch.sqrt(a / torch.exp(logr)).clamp(max=1.0)
    if rigid:
        w = torch.full_like(w, SIZE / SRC)
        h = torch.full_like(h, SIZE / SRC)
    cx = (torch.rand(n, device=dev, generator=gen) * 2 - 1) * (1 - w)
    cy = (torch.rand(n, device=dev, generator=gen) * 2 - 1) * (1 - h)
    th = u(-rot_deg, rot_deg) * math.pi / 180
    fx = torch.where(torch.rand(n, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(n, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(th), torch.sin(th)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = cos * w * fx
    theta[:, 0, 1] = -sin * h * fy
    theta[:, 1, 0] = sin * w * fx
    theta[:, 1, 1] = cos * h * fy
    theta[:, 0, 2] = cx
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (n, 3, SIZE, SIZE), align_corners=False)
    out = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    if strong:
        jitter = 0.4
        m = F.grid_sample(mask, grid, mode='bilinear', padding_mode='zeros', align_corners=False).clamp(0, 1)
        new_bg = (u(0.15, 0.95).view(n, 1, 1, 1) + (torch.rand(n, 3, 1, 1, device=dev, generator=gen) - 0.5) * 0.16).clamp(0, 1)
        swap = (torch.rand(n, 1, 1, 1, device=dev, generator=gen) < 0.5).float()
        out = out * (1 - swap) + (out * m + new_bg * (1 - m)) * swap
        out = out * (1 + (torch.rand(n, 3, 1, 1, device=dev, generator=gen) - 0.5) * 0.30)            # white balance
        grey = out.mean(dim=1, keepdim=True)
        out = (grey + (out - grey) * u(0.6, 1.4).view(n, 1, 1, 1)).clamp(0, 1)                         # saturation
        sel = torch.rand(n, device=dev, generator=gen) < 0.3
        if sel.any():
            out[sel] = _blur(out[sel], float(0.5 + 1.5 * torch.rand(1, device=dev, generator=gen)))
        sel = torch.rand(n, device=dev, generator=gen) < 0.3
        if sel.any():
            k = int(sel.sum())
            out[sel] = out[sel] + torch.randn((k,) + out.shape[1:], device=dev, generator=gen) * (0.01 + 0.05 * torch.rand(k, 1, 1, 1, device=dev, generator=gen))
        out = out.clamp(0, 1)
    if jitter > 0:
        b = u(1 - jitter, 1 + jitter).view(n, 1, 1, 1)
        c = u(1 - jitter, 1 + jitter).view(n, 1, 1, 1)
        out = out * b
        m = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - m) * c + m
    return out.clamp(0, 1)


# ----------------------------------------------------------------------------- corruptions for the robustness test
# Each function takes a float tensor N x 3 x H x W in [0, 1] and a severity 1..3 and returns the corrupted batch.
def _blur(x, sigma):
    r = int(math.ceil(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def _jpeg(x, quality):
    out = torch.empty_like(x)
    arr = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    for i in range(len(arr)):
        buf = io.BytesIO()
        Image.fromarray(arr[i]).save(buf, 'JPEG', quality=quality)
        out[i] = torch.from_numpy(np.asarray(Image.open(io.BytesIO(buf.getvalue())))).permute(2, 0, 1).to(x) / 255
    return out


def corrupt(x, kind, sev, mask=None, gen=None):
    """mask: N x 1 x H x W, 1 on the fruit and 0 on the background (needed for 'background')."""
    i = sev - 1
    if kind == 'darker':
        return x * (0.7, 0.5, 0.3)[i]
    if kind == 'brighter':
        return (x * (1.3, 1.6, 1.9)[i]).clamp(0, 1)
    if kind == 'low_contrast':
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - m) * (0.7, 0.5, 0.3)[i] + m).clamp(0, 1)
    if kind == 'noise':
        return (x + torch.randn(x.shape, device=x.device, generator=gen) * (0.04, 0.08, 0.12)[i]).clamp(0, 1)
    if kind == 'blur':
        return _blur(x, (1.0, 2.0, 3.0)[i])
    if kind == 'jpeg':
        return _jpeg(x, (30, 15, 8)[i])
    if kind == 'colour_cast':   # warm white-balance error: more red, less blue
        g = (0.10, 0.20, 0.30)[i]
        return (x * torch.tensor([1 + g, 1.0, 1 - g], device=x.device).view(1, 3, 1, 1)).clamp(0, 1)
    if kind == 'background':    # the white table is replaced by light grey, dark grey, or a tan surface
        col = ((0.70, 0.70, 0.70), (0.30, 0.30, 0.30), (0.76, 0.60, 0.42))[i]
        bg = torch.tensor(col, device=x.device).view(1, 3, 1, 1)
        return x * mask + bg * (1 - mask)
    raise ValueError(kind)


CORRUPTIONS = ['darker', 'brighter', 'low_contrast', 'colour_cast', 'noise', 'blur', 'jpeg', 'background']


# ----------------------------------------------------------------------------- models
def build_model(key, n_classes, pretrained=True):
    import timm
    spec = ALL_MODELS[key]
    m = timm.create_model(spec['tag'], pretrained=pretrained, num_classes=n_classes, **spec.get('kwargs', {}))
    if spec.get('grad_checkpointing'):
        m.set_grad_checkpointing(True)        # trades time for memory so that the ViT-B fits a 4 GB GPU at the common batch size
    cfg = m.pretrained_cfg
    return m, tuple(cfg['mean']), tuple(cfg['std'])


def normalise(x, mean, std):
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


# ----------------------------------------------------------------------------- metrics
def metrics(y_true, prob):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    pred = prob.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average='macro', zero_division=0)
    out = {'accuracy': float(accuracy_score(y_true, pred)), 'precision': float(p), 'recall': float(r), 'f1': float(f)}
    try:
        out['auc'] = float(roc_auc_score(y_true, prob, multi_class='ovr', average='macro', labels=list(range(prob.shape[1]))))
    except ValueError:
        out['auc'] = float('nan')
    return out
