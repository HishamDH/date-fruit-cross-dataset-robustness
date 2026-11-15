"""Repeat the five-fold runs of every network with further training seeds (same folds, same learning rates).

Usage: python run_seeds.py --recipe standard --seeds 1,2
       python run_seeds.py --recipe rigid --seeds 1,2
Writes results/<model>_s<seed>/ or results/<model>_rigid_s<seed>/, including the selected weights of every fold,
which the external test needs. Runs that already exist are skipped.
"""
import argparse
import json
import os
import subprocess
import sys

import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--recipe', choices=['standard', 'rigid'], default='standard')
ap.add_argument('--seeds', default='1,2')
ap.add_argument('--models', default=','.join(C.MODELS))
a = ap.parse_args()
lr = json.load(open(os.path.join(C.RESULTS, 'lr_choice.json')))
os.makedirs(os.path.join(C.ROOT, 'logs'), exist_ok=True)
for s in [int(v) for v in a.seeds.split(',')]:
    for m in a.models.split(','):
        tag = ('rigid_s%d' if a.recipe == 'rigid' else 's%d') % s
        args = ['train_cv.py', '--model', m, '--lr', lr[m]['lr'], '--epochs', '12', '--seed', str(s), '--tag', tag, '--keep-all-ckpt']
        if a.recipe == 'rigid':
            args.append('--rigid-aug')
        print('>>', ' '.join(args), flush=True)
        with open(os.path.join(C.ROOT, 'logs', 'seeds_%s.log' % a.recipe), 'a', encoding='utf-8') as fh:
            fh.write('\n>> %s\n' % ' '.join(args))
            fh.flush()
            r = subprocess.run([sys.executable] + args, stdout=fh, stderr=subprocess.STDOUT, cwd=C.HERE)
        if r.returncode != 0:
            print('   FAILED (exit %d)' % r.returncode, flush=True)
print('SEEDS DONE', a.recipe, a.seeds, flush=True)
