# -*- coding: utf-8 -*-
"""Replicate load_pjm_multiple_files' file-selection logic WITHOUT printing
unicode, to see which CSVs become 'clients' and what the aligned window is."""
import glob
import os
import sys

import pandas as pd

rows = []
for path in sorted(glob.glob('*.csv')):
    try:
        df = pd.read_csv(path)
    except Exception as exc:                       # noqa: BLE001
        rows.append((os.path.basename(path), -1, 'UNREADABLE: %s' % exc))
        continue
    dt_col = None
    for col in df.columns:
        if pd.to_numeric(df[col], errors='coerce').notna().mean() > 0.5:
            continue
        if pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.9:
            dt_col = col
            break
    picked = None
    for col in df.columns:
        if col == dt_col:
            continue
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) > 1000:
            picked = (col, len(values))
            break
    rows.append((os.path.basename(path), len(df),
                 'CLIENT via %r (%d values)' % picked if picked else 'skipped'))

print('%-42s %8s  %s' % ('file', 'rows', 'loader verdict'))
for name, n, verdict in rows:
    print('%-42s %8d  %s' % (name, n, verdict))

clients = [r for r in rows if r[2].startswith('CLIENT')]
print()
print('files that become clients: %d' % len(clients))
print('aligned window = min(values) capped at 13896 = %d'
      % min(13896, min(int(r[2].split('(')[1].split()[0]) for r in clients)))
