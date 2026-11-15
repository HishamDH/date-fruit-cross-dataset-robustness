"""Second batch of experiments, run one after another on the GPU once the first seed runs have finished.

1. SODA as a second dataset: the six networks (five folds, learning rates chosen on the main dataset) and the classical baseline.
2. Multi-source training: main dataset plus one external source, tested on the other external source, and a control
   trained on the main dataset alone (two seeds each).
3. External tests (Medina 27 and SODA photographs of the shared classes) with all trained networks.
4. Two further seeds of the no-augmentation ablation.
Every step skips work that already exists.
"""
import json
import os
import subprocess
import sys
import time

import psutil

import common as C

LOG = os.path.join(C.ROOT, 'logs')


def busy(names):
    for p in psutil.process_iter(['cmdline']):
        try:
            cl = ' '.join(p.info['cmdline'] or [])
        except Exception:  # noqa: BLE001
            continue
        if 'MSCS824_AI_Project' in cl and any(n in cl for n in names):
            return True
    return False


def run(args, log, env=None):
    print('>>', ' '.join(args), '' if not env else env, flush=True)
    e = dict(os.environ, **(env or {}))
    with open(os.path.join(LOG, log), 'a', encoding='utf-8') as fh:
        fh.write('\n>> %s\n' % ' '.join(args))
        fh.flush()
        r = subprocess.run([sys.executable] + args, stdout=fh, stderr=subprocess.STDOUT, cwd=C.HERE, env=e)
    if r.returncode != 0:
        print('   FAILED (exit %d), see logs/%s' % (r.returncode, log), flush=True)


while busy(['run_seeds.py']):
    time.sleep(30)
print('first seed runs finished', flush=True)
lr = json.load(open(os.path.join(C.MAIN_RESULTS, 'lr_choice.json')))
soda = {'DATES_DATASET': 'soda'}
for m in C.MODELS:
    run(['train_cv.py', '--model', m, '--lr', lr[m]['lr'], '--epochs', '12'], 'soda_runs.log', soda)
if not os.path.exists(os.path.join(C.ROOT, 'results_soda', 'svm', 'fold4.json')):
    run(['baseline_classical.py'], 'soda_runs.log', soda)
print('SODA DATASET DONE', flush=True)
# multi-source training with two seeds: a control on the main dataset alone (trained once, tested on both external sources),
# then main + one extra source, tested on the third source
for seed in (0, 1):
    for m in C.MODELS:
        run(['train_multisource.py', '--model', m, '--extra', 'none', '--target', 'medina27,soda', '--seed', str(seed)], 'multisource.log')
    for extra, target in (('soda', 'medina27'), ('medina27', 'soda')):
        for m in C.MODELS:
            run(['train_multisource.py', '--model', m, '--extra', extra, '--target', target, '--seed', str(seed)], 'multisource.log')
print('MULTISOURCE DONE', flush=True)
for name in ('medina27', 'soda'):
    run(['external_test.py', name, '--recipes', 'standard'], 'external_%s_final.log' % name)
print('CORE DONE', flush=True)
# two further seeds of the no-augmentation ablation, so that it is compared with the standard recipe on equal terms
for seed in (1, 2):
    run(['train_cv.py', '--model', 'convnext', '--lr', lr['convnext']['lr'], '--epochs', '12', '--no-aug', '--seed', str(seed), '--tag', 'noaug_s%d' % seed], 'ablation.log')
print('QUEUE DONE', flush=True)
# Optional and long (about four hours): two further seeds of the rigid recipe for the replication table.
#   python run_seeds.py --recipe rigid --seeds 1,2
