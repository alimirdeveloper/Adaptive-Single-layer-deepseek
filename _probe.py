# -*- coding: utf-8 -*-
"""Probe: what exactly happens at offset ch4-1 ?"""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
PROBE = os.path.join(DIR, '_probe_test.docx')
ORIG = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == 975309 and '_' not in os.path.basename(p)][0]

import shutil
if os.path.exists(PROBE):
    os.remove(PROBE)
shutil.copyfile(ORIG, PROBE)

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(PROBE)

CH4H = '\u0641\u0635\u0644 \u0686\u0647\u0627\u0631\u0645'
ch4 = None
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    if p.Style.NameLocal == 'Heading 1' and CH4H in p.Range.Text:
        ch4 = p.Range.Start
        prev = doc.Paragraphs.Item(i - 1)
        print('Ch4 at p%d start=%d end=%d style=%s text=%r'
              % (i, p.Range.Start, p.Range.End, p.Style.NameLocal, p.Range.Text))
        print('prev p%d start=%d end=%d style=%s text=%r'
              % (i - 1, prev.Range.Start, prev.Range.End, prev.Style.NameLocal,
                 prev.Range.Text[-30:]))
        break

print('\nchar at (ch4-1):', repr(doc.Range(ch4 - 1, ch4).Text))
print('char at (ch4)  :', repr(doc.Range(ch4, ch4 + 1).Text))

for probe_off in (ch4 - 1, ch4):
    tst = doc.Range(probe_off, probe_off)
    print('\n-- Range(%d,%d) belongs to paragraph style %s, text/2=%r'
          % (probe_off, probe_off, tst.Paragraphs.Item(1).Style.NameLocal,
             tst.Paragraphs.Item(1).Range.Text[:25]))

doc.Close(False)
os.remove(PROBE)
word.Quit()
print('done')
