"""Numbered contact sheets of the SODA photos kept as whole fruits, for a manual check.

Usage: python review_soda_fruits.py <class> [<class> ...]
A photo is kept when the fruit/pit classifier gives p_pit < 0.10 and the located object covers at most 40% of the
photo (larger objects are heaps of fruit). Writes data_ext/soda/_review_<class>.jpg; the number under each tile is
the position in the sorted list printed to data_ext/soda/_review_<class>.txt.
"""
import json
import os
import sys

from PIL import Image, ImageDraw

import common as C

root = os.path.join(C.ROOT, 'data_ext', 'soda')
recs = [json.loads(ln) for ln in open(os.path.join(root, 'manifest.jsonl'), encoding='utf-8')]
kind = json.load(open(os.path.join(root, 'kind.json')))
T = 120
for lab in sys.argv[1:]:
    rr = sorted((r for r in recs if r['label'] == lab and kind['%s/%s' % (lab, r['id'])]['p_pit'] < 0.10 and r['area_frac'] <= 0.40), key=lambda r: r['id'])
    cols = 16
    rows = (len(rr) + cols - 1) // cols
    sh = Image.new('RGB', (cols * T, rows * (T + 12)), (0, 0, 0))
    dr = ImageDraw.Draw(sh)
    with open(os.path.join(root, '_review_%s.txt' % lab), 'w', encoding='utf-8') as fh:
        for j, r in enumerate(rr):
            fh.write('%d\t%s\n' % (j, r['id']))
            x0, y0 = (j % cols) * T, (j // cols) * (T + 12)
            sh.paste(Image.open(os.path.join(root, 'fruit', lab, r['id'] + '.jpg')).resize((T, T)), (x0, y0))
            dr.text((x0 + 2, y0 + T), '%d' % j, fill=(255, 255, 0))
    sh.save(os.path.join(root, '_review_%s.jpg' % lab), quality=82)
    print(lab, 'kept', len(rr), 'of', sum(r['label'] == lab for r in recs))
