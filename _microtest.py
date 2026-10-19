# -*- coding: utf-8 -*-
"""Micro-experiment: which insertion recipe preserves the Ch4 Heading 1?"""
import os, sys, glob, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
ORIG = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == 975309 and '_' not in os.path.basename(p)][0]
CH4H = '\u0641\u0635\u0644 \u0686\u0647\u0627\u0631\u0645'

RECIPES = {
    'A_range_end_prev': 'range',      # Range.InsertAfter at prev.End
    'B_para_insertbefore': 'pib',     # Paragraph.Range.InsertBefore on Ch4 para
    'C_collapse_prev': 'collapse',    # prev.Range.Collapse(end) + TypeText
}


def run(name):
    mode = RECIPES[name]
    f = os.path.join(DIR, '_mx_%s.docx' % name)
    if os.path.exists(f):
        os.remove(f)
    shutil.copyfile(ORIG, f)

    word = win32.gencache.EnsureDispatch('Word.Application')
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(f)

    ch4p = None
    for i in range(1, doc.Paragraphs.Count + 1):
        p = doc.Paragraphs.Item(i)
        if p.Style.NameLocal == 'Heading 1' and CH4H in p.Range.Text:
            ch4p = p
            break

    if mode == 'range':
        prev = doc.Paragraphs.Item(ch4p.Range.Start and
                                   [i for i in range(1, doc.Paragraphs.Count + 1)
                                    if doc.Paragraphs.Item(i).Range.End == ch4p.Range.End][0] - 1)
        rng = doc.Range(ch4p.Start - 1, ch4p.Start - 1)
        rng.InsertAfter('\r')          # brand-new paragraph mark
        rng.InsertAfter('TESTHDR')
    elif mode == 'pib':
        ch4p.Range.InsertBefore('\r' + 'TESTHDR' + '\r')
    else:
        r = doc.Paragraphs.Item(
            [i for i in range(1, doc.Paragraphs.Count + 1)
             if doc.Paragraphs.Item(i).Range.End == ch4p.Range.End][0] - 1).Range
        r.Collapse(0)                  # wdCollapseEnd
        r.InsertAfter('TESTHDR')

    # now style the inserted paragraph, found by its text
    found = None
    for i in range(1, doc.Paragraphs.Count + 1):
        p = doc.Paragraphs.Item(i)
        if 'TESTHDR' in p.Range.Text:
            found = (i, p.Style.NameLocal)
            p.Style = doc.Styles('Heading 5')
            break
    # where is Ch4 now?
    ch4_now = [i for i in range(1, doc.Paragraphs.Count + 1)
               if doc.Paragraphs.Item(i).Style.NameLocal == 'Heading 1'
               and CH4H in doc.Paragraphs.Item(i).Range.Text]
    print('%-22s inserted-para=%s  ch4 H1 at %s' % (name, found, ch4_now))
    if ch4_now:
        p = doc.Paragraphs.Item(ch4_now[0])
        print('   ch4 text now = %r' % p.Range.Text[:40])
    doc.Close(False)
    word.Quit()
    try:
        os.remove(f)
    except Exception:
        pass


for n in RECIPES:
    try:
        run(n)
    except Exception as e:
        print('%-22s FAILED %s' % (n, e))
