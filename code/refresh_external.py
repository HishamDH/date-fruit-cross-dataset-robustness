"""Re-run the external tests with the networks trained so far, then refresh the summary and rebuild the paper (draft)."""
import os
import subprocess
import sys

import common as C

py = sys.executable
for name in ('medina27', 'soda'):
    subprocess.run([py, 'external_test.py', name, '--recipes', 'standard'], cwd=C.HERE)
subprocess.run([py, 'analyze.py'], cwd=C.HERE)
print('REFRESH DONE', flush=True)
