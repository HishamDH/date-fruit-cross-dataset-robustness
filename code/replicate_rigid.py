"""Robustness check of the model comparison: repeat the five-fold runs of every network with the rigid
augmentation (no change of scale or aspect ratio), using the learning rates already chosen.

Writes results/<model>_rigid/ and refreshes the summary. Runs that already exist are skipped.
"""
import json
import os
import subprocess
import sys

import common as C

lr = json.load(open(os.path.join(C.RESULTS, 'lr_choice.json')))
for m in C.MODELS:
    args = ['train_cv.py', '--model', m, '--lr', lr[m]['lr'], '--epochs', '12', '--rigid-aug', '--tag', 'rigid']
    print('>>', ' '.join(args), flush=True)
    with open(os.path.join(C.ROOT, 'logs', 'replicate_rigid.log'), 'a', encoding='utf-8') as fh:
        fh.write('\n>> %s\n' % ' '.join(args))
        fh.flush()
        r = subprocess.run([sys.executable] + args, stdout=fh, stderr=subprocess.STDOUT, cwd=C.HERE)
    if r.returncode != 0:
        print('   FAILED (exit %d)' % r.returncode, flush=True)
subprocess.run([sys.executable, 'analyze.py'], cwd=C.HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('REPLICATION DONE', flush=True)
