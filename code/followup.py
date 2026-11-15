"""Runs after run_experiments.py has finished: the third ablation (rigid augmentation), then the summary again.

The rigid augmentation keeps the scale and aspect ratio of the fruit (shift, rotation, flip, brightness and
contrast only). It was added after the no-augmentation ablation turned out more robust than the standard
recipe, to test whether the random change of scale and aspect ratio is the cause.
"""
import json
import os
import subprocess
import sys
import time

import psutil

import common as C


def pipeline_running():
    for p in psutil.process_iter(['cmdline']):
        try:
            cl = ' '.join(p.info['cmdline'] or [])
        except Exception:  # noqa: BLE001
            continue
        if 'run_experiments.py' in cl:
            return True
    return False


def run(args, log):
    print('>>', ' '.join(args), flush=True)
    with open(os.path.join(C.ROOT, 'logs', log), 'a', encoding='utf-8') as fh:
        fh.write('\n>> %s\n' % ' '.join(args))
        fh.flush()
        r = subprocess.run([sys.executable] + args, stdout=fh, stderr=subprocess.STDOUT, cwd=C.HERE)
    if r.returncode != 0:
        print('   FAILED (exit %d), see logs/%s' % (r.returncode, log), flush=True)


while pipeline_running():
    time.sleep(30)
print('pipeline finished, starting follow-up', flush=True)
lr = json.load(open(os.path.join(C.RESULTS, 'lr_choice.json')))['convnext']['lr']
run(['train_cv.py', '--model', 'convnext', '--lr', lr, '--epochs', '12', '--rigid-aug', '--tag', 'rigid'], 'ablation.log')
run(['analyze.py'], 'analyze.log')
print('FOLLOWUP DONE', flush=True)
