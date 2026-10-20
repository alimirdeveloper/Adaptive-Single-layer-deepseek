# -*- coding: utf-8 -*-
"""Per-client calendar window actually used by the loader (first 13,896 rows
after the ascending sort). Shows whether the 'clients' are time-aligned."""
import glob

import pandas as pd

TARGET = 13896
SELECTED = ['AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE', 'NI', 'PJME']
print('%-8s %-20s %-20s' % ('client', 'window start', 'window end'))
for name in SELECTED:
    df = pd.read_csv(glob.glob(name + '_hourly.csv')[0])
    dt = [c for c in df.columns if 'atetime' in c][0]
    k = pd.to_datetime(df[dt], errors='coerce').sort_values(kind='mergesort')
    print('%-8s %-20s %-20s' % (name, k.iloc[0], k.iloc[TARGET - 1]))
