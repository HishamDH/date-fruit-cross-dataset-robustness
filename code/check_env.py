"""Check the environment: GPU, library versions, and that every pretrained tag exists in timm."""
import sys

import numpy
import sklearn
import timm
import torch
import torchvision

import common as C

print('python', sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__, '| timm', timm.__version__,
      '| numpy', numpy.__version__, '| sklearn', sklearn.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-',
      '%.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 2 ** 30) if torch.cuda.is_available() else '')
for k, v in C.MODELS.items():
    found = timm.list_pretrained(v['tag'])
    print('%-16s %-40s %s' % (k, v['tag'], 'ok' if found else 'MISSING; candidates: %s' % timm.list_pretrained(v['tag'].split('.')[0] + '*')[:6]))
