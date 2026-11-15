"""Model size and speed: parameters, FLOPs, GPU and CPU latency.

Usage: python efficiency.py
Writes results/efficiency.json. Timings are medians over repeated runs after a warm-up; the GPU is
synchronised before every reading. CPU latency uses 4 threads.
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import common as C


def median_ms(fn, n, warm):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        t.append((time.perf_counter() - a) * 1000)
    return float(np.median(t)), float(np.percentile(t, 25)), float(np.percentile(t, 75))


def main():
    out = {}
    cuda = torch.cuda.is_available()
    for key, info in C.MODELS.items():
        model, _, _ = C.build_model(key, 9, pretrained=False)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        x1 = torch.randn(1, 3, C.SIZE, C.SIZE)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x1)
        res = {'name': info['name'], 'family': info['family'], 'tag': info['tag'], 'params_m': params / 1e6,
               'gflops': fc.get_total_flops() / 1e9, 'size_mb_fp32': params * 4 / 2 ** 20}
        torch.set_num_threads(4)
        with torch.no_grad():
            res['cpu_ms_b1'] = median_ms(lambda: model(x1), 30, 5)
        if cuda:
            dev = torch.device('cuda')
            model.to(dev)
            g1 = x1.to(dev)
            g32 = torch.randn(32, 3, C.SIZE, C.SIZE, device=dev)

            def run(x):
                with torch.no_grad(), torch.autocast('cuda'):
                    model(x)
                torch.cuda.synchronize()

            res['gpu_ms_b1'] = median_ms(lambda: run(g1), 100, 20)
            b32 = median_ms(lambda: run(g32), 20, 5)
            res['gpu_ms_b32'] = b32
            res['gpu_img_per_s_b32'] = 32000.0 / b32[0]
            res['gpu'] = torch.cuda.get_device_name(0)
            model.cpu()
            torch.cuda.empty_cache()
        out[key] = res
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ('tag',)}, flush=True)
    import platform
    import subprocess
    try:
        cpu = subprocess.run(['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name'], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        cpu = platform.processor()
    out['_machine'] = {'cpu': cpu, 'gpu': torch.cuda.get_device_name(0) if cuda else None, 'torch': torch.__version__,
                       'python': platform.python_version(), 'os': platform.platform()}
    print(out['_machine'], flush=True)
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(C.RESULTS, 'efficiency.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
