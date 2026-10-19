# -*- coding: utf-8 -*-
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
ORIG = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == 975309 and '_' not in os.path.basename(p)][0]
COPY = ORIG[:-5] + '_' + '\u0641\u0635\u0644\u06F3' + '.docx'

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(COPY, ReadOnly=True)
print('paras =', doc.Paragraphs.Count)
print('--- all H1/H2 headings ---')
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    st = p.Style.NameLocal
    if st in ('Heading 1', 'Heading 2'):
        t = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
        print('p%-5d [%-10s] %r' % (i, st, t[:60]))
print('--- raw dump p755..p765 ---')
for i in range(755, 766):
    if i > doc.Paragraphs.Count:
        break
    p = doc.Paragraphs.Item(i)
    print('p%-5d [%-10s] %r' % (i, p.Style.NameLocal, p.Range.Text[:70]))
doc.Close(False)
word.Quit()
