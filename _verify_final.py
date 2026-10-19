# -*- coding: utf-8 -*-
"""Full verification of the rebuilt thesis copy."""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
COPY = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if '_' in os.path.basename(p) and os.path.getsize(p) > 900000][0]
ORIG = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == 975309 and '_' not in os.path.basename(p)][0]

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0

# ---------------------------------------------------------------- copy
doc = word.Documents.Open(COPY, ReadOnly=True)
print('COPY paragraphs = %d   tables = %d' % (doc.Paragraphs.Count, doc.Tables.Count))

CH3H = '\u0641\u0635\u0644 \u0633\u0648\u0645'
print('\n=== Heading outline of the whole document ===')
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    st = p.Style.NameLocal
    if st in ('Heading 1', 'Heading 2'):
        if p.Range.Tables.Count > 0:
            continue
        t = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
        print('p%-5d %s  %s' % (i, st, t[:72]))

doc.Close(False)

# ---------------------------------------------------------------- original
o = word.Documents.Open(ORIG, ReadOnly=True)
print('\n=== headings per source document (for comparison) ===')
for name, d in (('ORIGINAL', o),):
    counts = {}
    for i in range(1, d.Paragraphs.Count + 1):
        p = d.Paragraphs.Item(i)
        if p.Range.Tables.Count == 0:
            counts[p.Style.NameLocal] = counts.get(p.Style.NameLocal, 0) + 1
    print(name, 'paragraphs=%d tables=%d' % (d.Paragraphs.Count, d.Tables.Count))
    print('   ', {k: v for k, v in sorted(counts.items()) if k.startswith(('Heading', 'Caption', 'Normal'))})
o.Close(False)

word.Quit()
print('\nVERIFY DONE')
