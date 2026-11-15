"""Print the external-test results: accuracy per variant and model, and where each shared class is sent."""
import json
import os
import sys

import numpy as np

import common as C

for name in sys.argv[1:]:
    p = os.path.join(C.RESULTS, 'external_%s.json' % name)
    if not os.path.exists(p):
        continue
    R = json.load(open(p))
    info = R['_info']
    print('=====', name, info['n'], 'images', info['counts'])
    tags = [t for t in R if t != '_info']
    for recipe in sorted({t.split('|')[1] for t in tags}):
        print('recipe', recipe, '(models per cell: %s)' % sorted({R[t]['n_models'] for t in tags if t.split('|')[1] == recipe}))
        print('%-16s %s' % ('', ''.join('%-22s' % v for v in ('frame', 'fruit', 'plain'))))
        for k in C.MODELS:
            cells = []
            for v in ('frame', 'fruit', 'plain'):
                t = '%s|%s|%s' % (v, recipe, k)
                cells.append('%5.1f +- %4.1f (ens %4.1f)' % (100 * R[t]['accuracy']['mean'], 100 * R[t]['accuracy']['std'], 100 * R[t]['ensemble_accuracy']) if t in R else '-')
            print('%-16s %s' % (k, ''.join('%-22s' % c for c in cells)))
    # where does each class go? (summed over all models of the standard recipe, plain variant, then frame)
    for v in ('frame', 'plain'):
        tot = None
        for k in C.MODELS:
            t = '%s|standard|%s' % (v, k)
            if t in R and 'confusion' in R[t]:
                c = np.array(R[t]['confusion'])
                tot = c if tot is None else tot + c
                cls = R[t]['classes']
        if tot is None:
            continue
        print('destination of each true class, %s variant, all models pooled (%%):' % v)
        print('%-11s' % 'true \\ pred', ' '.join('%6s' % c[:6] for c in cls))
        for i, c in enumerate(cls):
            if tot[i].sum():
                print('%-11s' % c, ' '.join('%6.1f' % (100 * x / tot[i].sum()) for x in tot[i]))
