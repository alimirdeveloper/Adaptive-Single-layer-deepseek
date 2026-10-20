# -*- coding: utf-8 -*-
"""Descriptive statistics of the ACTUAL analysis window (first 13,896 rows
after the loader's ascending sort), for the three clients in table 4-1."""
import glob
import os

import pandas as pd

TARGET = 13896
print('%-10s %10s %10s %10s %10s' % ('client', 'mean', 'sd', 'min', 'max'))
for name in ('AEP', 'COMED', 'DAYTON'):
    path = glob.glob(name + '_hourly.csv')[0]
    df = pd.read_csv(path)
    dt = [c for c in df.columns if 'atetime' in c or 'ate' in c][0]
    df = df.assign(_k=pd.to_datetime(df[dt], errors='coerce')) \
           .sort_values('_k', kind='mergesort').reset_index(drop=True)
    col = [c for c in df.columns if c.endswith('_MW')][0]
    v = pd.to_numeric(df[col], errors='coerce').dropna().values[:TARGET]
    print('%-10s %10.0f %10.0f %10.0f %10.0f'
          % (name, v.mean(), v.std(ddof=1), v.min(), v.max()))
    print('           window: %s .. %s  (n=%d)'
          % (df['_k'].iloc[0], df['_k'].iloc[TARGET - 1], len(v)))
