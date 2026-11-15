"""Median colour of the table (background) in the main dataset, used for the plain-background variant of the external test."""
import os

import numpy as np

import common as C

rows, classes, y = C.load_manifest()
X = C.load_images(rows)
masks = np.load(os.path.join(C.DATA, 'masks.npy'))
rng = np.random.default_rng(0)
pick = rng.choice(len(X), 300, replace=False)
px = np.concatenate([X[i][masks[i] == 0] for i in pick])
print('median background colour (R, G, B):', np.median(px, axis=0).tolist(), '| mean', np.round(px.mean(0), 1).tolist(), '| 5-95% of gray', np.percentile(px.mean(1), [5, 95]).tolist())
