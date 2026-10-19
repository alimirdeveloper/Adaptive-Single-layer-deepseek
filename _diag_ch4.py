# -*- coding: utf-8 -*-
"""Diagnose: where did chapter 4 go? Dump styles around the Ch3/Ch5 seam."""
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

def dump(path, lo, hi, show_outline=False):
    d = word.Documents.Open(path, ReadOnly=True)
    print('--- %s  (paras=%d) ---' % (os.path.basename(path), d.Paragraphs.Count))
    for i in range(lo, min(hi, d.Paragraphs.Count) + 1):
        p = d.Paragraphs.Item(i)
        t = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
        extra = ''
        if show_outline:
            extra = ' out=%s' % p.Range.ParagraphFormat.OutlineLevel
        print('p%-5d [%-10s]%s %s' % (i, p.Style.NameLocal, extra, t[:60]))
    d.Close(False)

# In the copy, Ch3 ends and Ch5 starts at p1096
dump(COPY, 1085, 1100, show_outline=True)
print()
# In the original, Ch4 body region
dump(ORIG, 854, 870, show_outline=True)
word.Quit()
