"""Restart run_queue.py (only while it is still waiting or between steps)."""
import os
import subprocess
import sys

import psutil

for p in psutil.process_iter(['pid', 'cmdline']):
    try:
        cl = ' '.join(p.info['cmdline'] or [])
    except Exception:  # noqa: BLE001
        continue
    if 'run_queue.py' in cl and 'MSCS824_AI_Project' in cl and 'restart_queue' not in cl and 'launch_detached' not in cl:
        p.kill()
        print('stopped', p.info['pid'])
here = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(here, 'launch_detached.py'), 'queue.log', 'run_queue.py'])
