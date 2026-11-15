"""Rebuild the working copies of the external datasets from scratch (after a change in fruit_crop.py)."""
import os
import shutil
import subprocess
import sys

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in psutil.process_iter(['pid', 'cmdline']):
    try:
        cl = ' '.join(p.info['cmdline'] or [])
    except Exception:  # noqa: BLE001
        continue
    if 'prepare_external.py' in cl and 'MSCS824_AI_Project' in cl:
        p.kill()
        print('stopped', p.info['pid'], flush=True)
for name in ('soda_hf', 'soda', 'medina27'):
    d = os.path.join(ROOT, 'data_ext', name)
    if os.path.isdir(d):
        shutil.rmtree(d)
        print('removed', d, flush=True)
partial = os.path.join(ROOT, '_data', 'external', 'soda')      # partial Hugging Face copy, replaced by the full Kaggle archive
if os.path.isdir(partial):
    shutil.rmtree(partial, ignore_errors=True)
    print('removed partial SODA copy', flush=True)
for name in sys.argv[1:]:
    print('preparing', name, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, 'prepare_external.py'), name], cwd=HERE)
print('REDO DONE', flush=True)
