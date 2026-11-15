"""Third batch of experiments (strengthening round), run one after another on the GPU. Every step skips work that already exists.

1. A third seed of the multi-source experiment (control and both directions).
2. Two further seeds of the repetition on SODA (13 varieties).
Later steps are appended by run_queue3.py once their code exists.
"""
import json
import os
import subprocess
import sys

import common as C

LOG = os.path.join(C.ROOT, 'logs')


def run(args, log, env=None):
    print('>>', ' '.join(args), '' if not env else env, flush=True)
    e = dict(os.environ, **(env or {}))
    with open(os.path.join(LOG, log), 'a', encoding='utf-8') as fh:
        fh.write('\n>> %s\n' % ' '.join(args))
        fh.flush()
        r = subprocess.run([sys.executable] + args, stdout=fh, stderr=subprocess.STDOUT, cwd=C.HERE, env=e)
    if r.returncode != 0:
        print('   FAILED (exit %d), see logs/%s' % (r.returncode, log), flush=True)


if __name__ == '__main__':
    lr = json.load(open(os.path.join(C.MAIN_RESULTS, 'lr_choice.json')))
    for m in C.MODELS:
        run(['train_multisource.py', '--model', m, '--extra', 'none', '--target', 'medina27,soda', '--seed', '2'], 'multisource.log')
    for extra, target in (('soda', 'medina27'), ('medina27', 'soda')):
        for m in C.MODELS:
            run(['train_multisource.py', '--model', m, '--extra', extra, '--target', target, '--seed', '2'], 'multisource.log')
    print('MULTISOURCE SEED 2 DONE', flush=True)
    soda = {'DATES_DATASET': 'soda'}
    for seed in (1, 2):
        for m in C.MODELS:
            run(['train_cv.py', '--model', m, '--lr', lr[m]['lr'], '--epochs', '12', '--seed', str(seed), '--tag', 's%d' % seed], 'soda_runs.log', soda)
    print('SODA SEEDS DONE', flush=True)
    print('QUEUE2 DONE', flush=True)
