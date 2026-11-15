"""Record the result of the manual check of the SODA whole-fruit photos (shared classes) in external_config.json.

The numbers refer to the tiles of data_ext/soda/_review_<class>.jpg (review_soda_fruits.py). Removed photos:
heaps of fruit, pits (some held in a hand), and fruits that were cut or pulled open.
"""
import json
import os

import common as C

REMOVE = {
    'Ajwa': [2, 61],                      # heap of fruit; pit held in a hand
    'Medjool': [52],                      # pit held in a hand
    'Sukari_Galaxy': [0, 51, 52, 56, 71],  # heap; opened fruit in a hand; pit in a hand; pit; fruit pulled open
    'Shaishi': [0, 4, 5, 69, 70, 71],     # skin/pit in a hand; opened fruit in a hand; unclear small object; cut-open fruits
    'Sagai': [0, 55, 56, 72],             # flattened opened fruit; opened fruits held in a hand
    # classes that are not shared with the main dataset (used only when SODA serves as the second dataset)
    'Anbara': [2, 72, 74],                # close-up of a heap; small objects held in a hand
    'Khalas': [50, 57, 59, 60, 70],       # opened fruits and pits held in a hand
    'Raziz': [1, 4, 71, 72],              # heap; pit in a hand; opened fruits in a hand
    'Safawi': [0, 66, 69, 70, 72],        # heap; split fruit; opened fruit and pits in a hand
    'Shalabi': [11, 61, 62, 63, 64, 67, 69],   # pits in a hand; cut-open fruits
    'Sukari_Mofattal': [1],               # heap
    'Sukari_Rutab': [1, 69],              # heap; fruit pulled open
    'Wanan': [54, 55, 66, 69],            # small object and pit in a hand; fruits pulled open
}
root = os.path.join(C.ROOT, 'data_ext', 'soda')
excl = []
for lab, idxs in REMOVE.items():
    ids = dict(ln.rstrip('\n').split('\t') for ln in open(os.path.join(root, '_review_%s.txt' % lab), encoding='utf-8'))
    excl += ['%s/%s' % (lab, ids[str(i)]) for i in idxs]
path = os.path.join(C.HERE, 'external_config.json')
cfg = json.load(open(path))
cfg['soda']['exclude'] = sorted(excl)
cfg['soda']['kind_file'] = True
cfg['soda']['max_p_pit'] = 0.10
cfg['soda']['max_area'] = 0.40
cfg['soda']['note'] = ('Whole-fruit photographs only: fruit/pit classifier p_pit < 0.10, located object at most 40%% of the photo, '
                       'and %d photos removed after a manual check (heaps, pits, opened fruits). Sukari_Rutab and Sukari_Mofattal are not mapped.' % len(excl))
json.dump(cfg, open(path, 'w'), indent=1)
print('excluded', len(excl), excl)
