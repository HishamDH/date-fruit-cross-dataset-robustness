"""Print the layout of the downloaded external datasets (file counts per folder, image sizes, examples)."""
import os
import sys
import zipfile
from collections import Counter

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), '_data', 'external')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

z = os.path.join(EXT, 'uci_ajwa_or_medjool', 'ajwa_or_medjool.zip')
if os.path.exists(z):
    with zipfile.ZipFile(z) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        print('UCI zip entries', len(names))
        c = Counter((os.path.dirname(n), os.path.splitext(n)[1].lower()) for n in names)
        for k, v in sorted(c.items()):
            print('   %5d  %s  %s' % (v, k[0], k[1]))
        print('   examples', names[:6])
        inner = [n for n in names if n.lower().endswith('.zip')]
        print('   inner zips', inner)

soda = os.path.join(EXT, 'soda')
if os.path.isdir(soda):
    tot = 0
    for d in sorted(os.listdir(soda)):
        p = os.path.join(soda, d)
        if not os.path.isdir(p) or d.startswith('.'):
            continue
        c = Counter()
        ex = {}
        for dd, _, files in os.walk(p):
            for f in files:
                key = (os.path.relpath(dd, p), os.path.splitext(f)[1].lower())
                c[key] += 1
                ex.setdefault(key, f)
        tot += sum(c.values())
        print('SODA', d)
        for k, v in sorted(c.items()):
            size = ''
            if k[1] in IMG:
                try:
                    with Image.open(os.path.join(p, k[0], ex[k])) as im:
                        size = '%dx%d %s' % (im.size[0], im.size[1], im.mode)
                except Exception as e:  # noqa: BLE001
                    size = repr(e)[:40]
            print('   %5d  %-28s %-6s e.g. %s  %s' % (v, k[0], k[1], ex[k], size))
    print('SODA files in class folders:', tot)
