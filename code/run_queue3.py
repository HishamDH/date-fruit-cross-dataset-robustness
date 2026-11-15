"""Fourth batch of experiments (strengthening round); starts when run_queue2.py has finished. Every step skips finished work.

1. Test-time remedies on the external photographs (test-time augmentation, batch-statistics adaptation).
2. Checkpoints pretrained on more data than ImageNet-1k (C.EXTRA_MODELS): learning-rate check on the validation part of
   fold 0, then five folds with the chosen rate (seed index 1, weights of every fold kept).
3. The strong-augmentation recipe for the six main networks (seed index 1, weights kept).
4. External tests for the new runs.
"""
import json
import os
import time

import psutil

import common as C
from run_queue2 import run

EXTRA_LRS = ['1e-5', '3e-5', '1e-4']     # one step lower than for the ImageNet-1k checkpoints: large pretrained transformers need smaller rates


def busy(name):
    for p in psutil.process_iter(['cmdline']):
        try:
            cl = ' '.join(p.info['cmdline'] or [])
        except Exception:  # noqa: BLE001
            continue
        if 'MSCS824_AI_Project' in cl and name in cl:
            return True
    return False


while busy('run_queue2.py'):
    time.sleep(30)
print('queue 2 finished', flush=True)
for name in ('medina27', 'soda'):
    if not os.path.exists(os.path.join(C.RESULTS, 'external_tta_%s.json' % name)):
        run(['external_tta.py', name], 'external_tta.log')
print('TTA DONE', flush=True)
choice_path = os.path.join(C.RESULTS, 'lr_choice.json')
for m in C.EXTRA_MODELS:
    for lr in EXTRA_LRS:
        run(['train_cv.py', '--model', m, '--folds', '0', '--lr', lr, '--epochs', '12', '--tag', 'lr' + lr, '--val-only'], 'extra_models.log')
    choice = json.load(open(choice_path))
    if m not in choice:
        cand = {}
        for lr in EXTRA_LRS:
            p = os.path.join(C.RESULTS, '%s_lr%s' % (m, lr), 'fold0.json')
            if os.path.exists(p):
                d = json.load(open(p))
                best = d['history'][d['best_epoch'] - 1]
                cand[lr] = {'val_acc': best['val_acc'], 'val_loss': best['val_loss'], 'best_epoch': d['best_epoch']}
        if len(cand) < len(EXTRA_LRS):
            print('learning-rate check incomplete for', m, flush=True)
            continue
        pick = max(cand, key=lambda k: (cand[k]['val_acc'], -cand[k]['val_loss']))
        choice[m] = {'lr': pick, 'candidates': cand}
        json.dump(choice, open(choice_path, 'w'), indent=1)
        print('learning rate for %s: %s  %s' % (m, pick, {k: round(v['val_acc'], 4) for k, v in cand.items()}), flush=True)
    run(['train_cv.py', '--model', m, '--lr', choice[m]['lr'], '--epochs', '12', '--seed', '1', '--tag', 's1', '--keep-all-ckpt'], 'extra_models.log')
print('EXTRA MODELS DONE', flush=True)
choice = json.load(open(choice_path))
for m in C.MODELS:
    run(['train_cv.py', '--model', m, '--lr', choice[m]['lr'], '--epochs', '12', '--strong-aug', '--seed', '1', '--tag', 'strong_s1', '--keep-all-ckpt'], 'strong_aug.log')
print('STRONG AUGMENTATION DONE', flush=True)
for name in ('medina27', 'soda'):
    run(['external_test.py', name, '--recipes', 'standard,strong'], 'external_%s_final.log' % name)
print('QUEUE3 DONE', flush=True)
