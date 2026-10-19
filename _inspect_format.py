# -*- coding: utf-8 -*-
"""Inspect the exact formatting of Chapter 1 & 2 headings + body paragraphs
so Chapter 3 can match them."""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
COPY = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.basename(p).endswith('.docx') and '_' in os.path.basename(p)
        and os.path.getsize(p) > 900000][0]
print('file =', COPY)

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(COPY, ReadOnly=True)

def describe(i):
    p = doc.Paragraphs.Item(i)
    pf = p.Range.ParagraphFormat
    f = p.Range.Font
    txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
    print('p%-4d [%-9s] al=%-2s lbl=%s out=%-4s rtl=%s spB=%s spA=%s spL=%s ls=%s '
          '| font=%s sz=%s b=%s | ind=%s | %s' % (
          i, p.Style.NameLocal, pf.Alignment, pf.FirstLineIndent, pf.OutlineLevel,
          pf.ReadingOrder, pf.SpaceBefore, pf.SpaceAfter, pf.LineSpacing,
          pf.LineSpacingRule, f.Name, f.Size, f.Bold,
          pf.LeftIndent, txt[:55]))

targets = {
    'H1 ch1': 135, 'H2 ch1': 137, 'H3 ch1': 140, 'H3 ch1 b': 152,
    'H1 ch2': 262, 'H2 ch2': 264, 'H3 ch2': 266, 'H3 ch2b': 276,
    'H1 ch4': 760,
}
print('\n=== headings ===')
for k, i in targets.items():
    print('%-10s ' % k, end='')
    describe(i)

print('\n=== body paragraphs ===')
for k, i in [('ch1 body', 141), ('ch1 body2', 142), ('ch2 body', 267),
             ('ch2 body2', 268), ('ch2 body3', 277)]:
    print('%-10s ' % k, end='')
    describe(i)

print('\n=== Caption paragraphs in ch1/ch2 ===')
for i in range(135, 760):
    p = doc.Paragraphs.Item(i)
    if p.Style.NameLocal == 'Caption':
        describe(i)
        if i > 300:
            break

print('\n=== all styles present ===')
names = set()
for i in range(1, doc.Paragraphs.Count + 1):
    if i > 760:
        break
    names.add(doc.Paragraphs.Item(i).Style.NameLocal)
print(sorted(names))

doc.Close(False)
word.Quit()
