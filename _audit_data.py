# -*- coding: utf-8 -*-
"""Ground truth for the audit: which clients the loader really builds, and the
descriptive statistics of the analysis window (13,896 aligned samples)."""
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asla_complete import load_pjm_multiple_files

data, names = load_pjm_multiple_files('.')
print()
print('CLIENTS LOADED = %d' % len(names))
print('names =', names)
print('shape =', data.shape)
print()
print('descriptive statistics of the ALIGNED ANALYSIS WINDOW (%d samples):' % data.shape[0])
print('%-10s %10s %10s %10s %10s' % ('client', 'mean', 'sd', 'min', 'max'))
for i, n in enumerate(names):
    col = data[:, i]
    print('%-10s %10.0f %10.0f %10.0f %10.0f'
          % (n, col.mean(), col.std(ddof=1), col.min(), col.max()))
