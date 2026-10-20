# -*- coding: utf-8 -*-
"""Can table 4-1 be reproduced from the shipped CSVs under ANY single window?

Tries the full series and several 13,896-sample slices. If none matches the
document's (mean, sd, min, max), the table is not reproducible from the data.
"""
import glob

import pandas as pd

TARGET = 13896
DOC = {  # table 4-1 as printed in the proposal
    'AEP':    (16067, 2549, 9823, 25695),
    'COMED':  (11521, 2399, 7237, 21753),
    'DAYTON': (2126, 405, 982, 3746),
}

for name, (dmean, dsd, dmin, dmax) in DOC.items():
    df = pd.read_csv(glob.glob(name + '_hourly.csv')[0])
    dt = [c for c in df.columns if 'atetime' in c][0]
    df = df.assign(_k=pd.to_datetime(df[dt], errors='coerce')) \
           .sort_values('_k', kind='mergesort').reset_index(drop=True)
    col = [c for c in df.columns if c.endswith('_MW')][0]
    v = pd.to_numeric(df[col], errors='coerce').dropna().values
    slices = {
        'full series': v,
        'first 13896': v[:TARGET],
        'last 13896': v[-TARGET:],
        'middle 13896': v[(len(v) - TARGET) // 2:(len(v) - TARGET) // 2 + TARGET],
    }
    print('=== %s   document says mean=%d sd=%d min=%d max=%d' % (name, dmean, dsd, dmin, dmax))
    for label, s in slices.items():
        hit = (round(s.mean()) == dmean and round(s.std(ddof=1)) == dsd
               and round(s.min()) == dmin and round(s.max()) == dmax)
        print('    %-14s mean=%6.0f sd=%5.0f min=%6.0f max=%6.0f  %s'
              % (label, s.mean(), s.std(ddof=1), s.min(), s.max(),
                 '<-- MATCHES' if hit else ''))
