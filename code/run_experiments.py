"""Run the whole experimental protocol, step by step, skipping whatever is already done.

Usage: python run_experiments.py [--epochs 12]

1. Learning-rate check: every model is trained on fold 0 with three learning rates. The rate with the
   best VALIDATION accuracy (ties: lower validation loss) is kept; test results play no part.
2. Main runs: every model, 5 folds, with its chosen learning rate (fold 0 is reused from step 1).
3. Classical baseline (HOG + LBP + colour histogram, RBF SVM).
4. Ablations on the reference model: no augmentation, and training from scratch.
5. Efficiency measurements, Grad-CAM, tables and figures.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import common as C

LRS = ['3e-5', '1e-4', '3e-4']
ABLATION_MODEL = 'convnext'


def run(args, log):
    cmd = [sys.executable] + args
    print('>>', ' '.join(args), flush=True)
    with open(os.path.join(C.ROOT, 'logs', log), 'a', encoding='utf-8') as fh:
        fh.write('\n>> %s\n' % ' '.join(args))
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=C.HERE)
    if r.returncode != 0:
        print('   FAILED (exit %d), see logs/%s' % (r.returncode, log), flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', default='12')
    ap.add_argument('--models', default=','.join(C.MODELS))
    a = ap.parse_args()
    models = a.models.split(',')
    os.makedirs(os.path.join(C.ROOT, 'logs'), exist_ok=True)
    # 1. learning-rate check on fold 0
    for m in models:
        for lr in LRS:
            run(['train_cv.py', '--model', m, '--folds', '0', '--lr', lr, '--epochs', a.epochs, '--tag', 'lr' + lr, '--keep-ckpt'], 'lr_check.log')
    choice_path = os.path.join(C.RESULTS, 'lr_choice.json')
    choice = json.load(open(choice_path)) if os.path.exists(choice_path) else {}
    for m in models:
        if m in choice:
            continue
        cand = {}
        for lr in LRS:
            p = os.path.join(C.RESULTS, '%s_lr%s' % (m, lr), 'fold0.json')
            if os.path.exists(p):
                d = json.load(open(p))
                best = d['history'][d['best_epoch'] - 1]
                cand[lr] = {'val_acc': best['val_acc'], 'val_loss': best['val_loss'], 'best_epoch': d['best_epoch']}
        if len(cand) < len(LRS):
            print('learning-rate check incomplete for', m)
            continue
        pick = max(cand, key=lambda k: (cand[k]['val_acc'], -cand[k]['val_loss']))
        choice[m] = {'lr': pick, 'candidates': cand}
        json.dump(choice, open(choice_path, 'w'), indent=1)
        print('learning rate for %s: %s  %s' % (m, pick, {k: round(v['val_acc'], 4) for k, v in cand.items()}), flush=True)
    # 2. main runs
    for m in models:
        if m not in choice:
            continue
        lr = choice[m]['lr']
        dst = os.path.join(C.RESULTS, m)
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(C.RESULTS, '%s_lr%s' % (m, lr))
        for f in ('fold0.json', 'fold0_pred.npz', 'fold0.pt'):
            if os.path.exists(os.path.join(src, f)) and not os.path.exists(os.path.join(dst, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        run(['train_cv.py', '--model', m, '--lr', lr, '--epochs', a.epochs, '--keep-ckpt'], 'main.log')
    # the checkpoints of the rejected learning rates are not needed any more
    for m in models:
        for lr in LRS:
            p = os.path.join(C.RESULTS, '%s_lr%s' % (m, lr), 'fold0.pt')
            if os.path.exists(p):
                os.remove(p)
    # 3. classical baseline
    if not os.path.exists(os.path.join(C.RESULTS, 'svm', 'fold4.json')):
        run(['baseline_classical.py'], 'svm.log')
    # 4. ablations
    if ABLATION_MODEL in choice:
        lr = choice[ABLATION_MODEL]['lr']
        run(['train_cv.py', '--model', ABLATION_MODEL, '--lr', lr, '--epochs', a.epochs, '--no-aug', '--tag', 'noaug'], 'ablation.log')
        run(['train_cv.py', '--model', ABLATION_MODEL, '--lr', lr, '--epochs', str(4 * int(a.epochs)), '--scratch', '--tag', 'scratch'], 'ablation.log')
    # 5. efficiency, Grad-CAM, summary
    run(['efficiency.py'], 'efficiency.log')
    run(['gradcam.py', '--models', ','.join(models)], 'gradcam.log')
    run(['analyze.py'], 'analyze.log')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
