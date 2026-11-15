"""Grad-CAM maps for the fold-0 checkpoints, and the share of the map that falls on the fruit.

Usage: python gradcam.py --models convnext,resnet50,swin
Writes results/gradcam_<model>.npz (maps for the fold-0 test images) and results/gradcam.json with,
per model, the mean fraction of Grad-CAM mass inside the fruit mask on clean test images.

The map is computed on the output of `forward_features` (the last feature map of a CNN, the last
token grid of a transformer), for the predicted class.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C


def cam_batch(model, key, x):
    if key == 'deit':
        # The head of a ViT reads only the class token, so the patch tokens of the last layer have no
        # gradient. The map is taken on the tokens that enter the last transformer block instead.
        t = model.patch_embed(x)
        t = model._pos_embed(t)
        t = model.patch_drop(t)
        t = model.norm_pre(t)
        for blk in model.blocks[:-1]:
            t = blk(t)
        feat = t
        feat.retain_grad()
        logits = model.forward_head(model.norm(model.blocks[-1](feat)))
    else:
        feat = model.forward_features(x)
        feat.retain_grad()
        logits = model.forward_head(feat)
    pred = logits.argmax(1)
    logits.gather(1, pred[:, None]).sum().backward()
    g, f = feat.grad, feat
    if key == 'deit':                       # N x (1 + 196) x C tokens
        g, f = g[:, 1:], f[:, 1:]
        s = int(g.shape[1] ** 0.5)
        g, f = g.reshape(-1, s, s, g.shape[-1]).permute(0, 3, 1, 2), f.reshape(-1, s, s, f.shape[-1]).permute(0, 3, 1, 2)
    elif key == 'swin':                     # N x H x W x C
        g, f = g.permute(0, 3, 1, 2), f.permute(0, 3, 1, 2)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(C.SIZE, C.SIZE), mode='bilinear', align_corners=False)
    cam = cam / cam.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    return cam[:, 0].detach(), pred.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='resnet50,convnext,swin')
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows, classes, y = C.load_manifest()
    X = C.load_images(rows)
    groups = C.load_groups(rows)
    fold = C.get_folds(y, groups)
    masks = np.load(os.path.join(C.DATA, 'masks.npy'))
    _, _, te = C.split_fold(y, fold, 0, groups)
    path = os.path.join(C.RESULTS, 'gradcam.json')
    summary = json.load(open(path)) if os.path.exists(path) else {}
    for key in args.models.split(','):
        model, mean, std = C.build_model(key, len(classes), pretrained=False)
        model.load_state_dict(torch.load(os.path.join(C.RESULTS, key, 'fold0.pt'), map_location='cpu'))
        model.to(dev).eval()
        cams, preds = [], []
        for i in range(0, len(te), 16):
            b = te[i:i + 16]
            x = C.centre_crop(torch.from_numpy(X[b]).to(dev).permute(0, 3, 1, 2).float() / 255)
            model.zero_grad(set_to_none=True)
            cam, pred = cam_batch(model, key, C.normalise(x, mean, std))
            cams.append(cam.cpu().numpy())
            preds.append(pred.cpu().numpy())
        cams, preds = np.concatenate(cams), np.concatenate(preds)
        m = masks[te][:, 16:240, 16:240].astype(np.float32)
        inside = (cams * m).sum((1, 2)) / np.maximum(cams.sum((1, 2)), 1e-8)
        area = m.mean((1, 2))
        summary[key] = {'cam_mass_on_fruit_mean': float(inside.mean()), 'cam_mass_on_fruit_median': float(np.median(inside)),
                        'fruit_area_fraction_mean': float(area.mean()), 'n': int(len(te)),
                        'accuracy_fold0': float((preds == y[te]).mean())}
        np.savez_compressed(os.path.join(C.RESULTS, 'gradcam_%s.npz' % key), idx=te, cam=cams.astype(np.float16), pred=preds)
        print(key, summary[key], flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(summary, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
