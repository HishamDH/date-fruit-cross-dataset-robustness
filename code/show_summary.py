"""Print the key numbers of results/summary.json in a compact form."""
import json
import os

import numpy as np

import common as C

S = json.load(open(os.path.join(C.RESULTS, 'summary.json')))
R = S['runs']
main = [r for r in ['svm', 'resnet50', 'mobilenetv3', 'efficientnetv2', 'convnext', 'deit', 'swin'] if r in R]
print('best clean:', S.get('best_clean'), '| worst corruptions:', S.get('worst_corruptions'))
print('%-15s %-14s %6s %6s %6s %7s  %-18s %4s  %s' % ('run', 'acc', 'P', 'R', 'F1', 'AUC', 'pooled [95% CI]', 'err', 'mCA'))
for r in main:
    s = R[r]
    print('%-15s %5.2f +- %4.2f %6.2f %6.2f %6.2f %7.4f  %5.2f [%5.2f, %5.2f] %4d  %5.2f +- %4.2f' % (
        r, 100 * s['accuracy']['mean'], 100 * s['accuracy']['std'], 100 * s['precision']['mean'], 100 * s['recall']['mean'], 100 * s['f1']['mean'],
        s['auc']['mean'], 100 * s['pooled']['accuracy'], 100 * s['pooled']['ci95'][0], 100 * s['pooled']['ci95'][1], s['pooled']['errors'],
        100 * s['mean_corrupted_accuracy']['mean'], 100 * s['mean_corrupted_accuracy']['std']))
print('\nfold accuracies:')
for r in main:
    print('  %-15s' % r, ' '.join('%.2f' % (100 * v) for v in R[r]['accuracy']['folds']), '| best epochs', R[r]['best_epoch'])
print('\nmean accuracy per corruption (over 3 severities):')
print('%-15s' % '', ' '.join('%9s' % c[:9] for c in C.CORRUPTIONS))
for r in main:
    print('%-15s' % r, ' '.join('%9.1f' % (100 * R[r]['corruption_mean'][c]) for c in C.CORRUPTIONS))
print('\naccuracy per severity:')
for c in C.CORRUPTIONS:
    print(' ', c)
    for r in main:
        print('    %-15s' % r, ' '.join('%6.1f' % (100 * R[r]['robustness']['%s_%d' % (c, s)]['mean']) for s in (1, 2, 3)))
print('\nMcNemar vs best:', json.dumps(S.get('mcnemar_vs_best'), indent=0))
if 'per_class_best' in S:
    print('\nper class (best):')
    for k in S['classes']:
        v = S['per_class_best'][k]
        print('  %-11s P %.3f R %.3f F1 %.3f n %d' % (k, v['precision'], v['recall'], v['f1-score'], v['support']))
    print('confusion (best):')
    for row in R[S['best_clean']]['confusion']:
        print('  ', row)
    print('confusion (svm):')
    for row in R['svm']['confusion']:
        print('  ', row)
for a in ('convnext_rigid', 'convnext_noaug', 'convnext_scratch'):
    if a in R:
        n = R[a]['n_folds']
        base = R['convnext']
        print('\n%s (%d folds): acc %s | mCA %.2f' % (a, n, ' '.join('%.2f' % (100 * v) for v in R[a]['accuracy']['folds']), 100 * R[a]['mean_corrupted_accuracy']['mean']))
        print('   convnext same folds: acc %s' % ' '.join('%.2f' % (100 * v) for v in base['accuracy']['folds'][:n]))
        print('   per corruption  %s' % ' '.join('%s %.1f' % (c[:6], 100 * R[a]['corruption_mean'][c]) for c in C.CORRUPTIONS))
        print('   convnext (5 f)  %s' % ' '.join('%s %.1f' % (c[:6], 100 * base['corruption_mean'][c]) for c in C.CORRUPTIONS))
print('\nlr choice:', {k: (v['lr'], {a: round(b['val_acc'], 4) for a, b in v['candidates'].items()}) for k, v in S.get('lr_choice', {}).items()})
print('gradcam:', S.get('gradcam'))
print('efficiency:', {k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a in ('params_m', 'gflops', 'gpu_ms_b1', 'cpu_ms_b1', 'gpu_img_per_s_b32')} for k, v in S.get('efficiency', {}).items() if not k.startswith('_')})
print('misc:', {k: S.get(k) for k in ('fruit_area_median', 'gradcam_area', 'epochs', 'split_sizes', 'machine')})
for r in ('convnext', 'convnext_rigid', 'convnext_noaug', 'convnext_scratch'):
    if r in R:
        s = R[r]
        print(r, 'acc %.2f +- %.2f f1 %.2f pooled %d/%d mCA %.2f +- %.2f best epochs %s' % (100 * s['accuracy']['mean'], 100 * s['accuracy']['std'], 100 * s['f1']['mean'],
              s['pooled']['correct'], s['pooled']['n'], 100 * s['mean_corrupted_accuracy']['mean'], 100 * s['mean_corrupted_accuracy']['std'], s['best_epoch']))
