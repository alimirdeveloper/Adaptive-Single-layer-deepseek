# -*- coding: utf-8 -*-
"""Dump raw offsets around the Ch3 heading in the pristine original."""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
ORIG = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == 975309 and '_' not in os.path.basename(p)][0]

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(ORIG, ReadOnly=True)

for i in range(744, 764):
    p = doc.Paragraphs.Item(i)
    print('p%-4d [%-10s] start=%-7d end=%-7d %r'
          % (i, p.Style.NameLocal, p.Range.Start, p.Range.End,
             p.Range.Text.replace('\r', '\\r').replace('\x0c', '\\f')[:50]))
print()
s = doc.Paragraphs.Item(759).Range.End
e = doc.Paragraphs.Item(760).Range.End
print('chars [759.end .. 760.end] =', repr(doc.Range(s, e).Text))
doc.Close(False)
word.Quit()
