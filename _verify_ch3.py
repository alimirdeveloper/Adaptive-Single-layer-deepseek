# -*- coding: utf-8 -*-
"""Verify the injected Chapter 3 inside the working copy."""
import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
cand = [p for p in glob.glob(os.path.join(DIR, '*.docx')) if os.path.getsize(p) == 986430]
if not cand:
    print('working copy not found'); sys.exit(1)
TARGET = cand[0]

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(TARGET, ReadOnly=True)

print('total paragraphs =', doc.Paragraphs.Count)
print('total tables     =', doc.Tables.Count)

CH3 = '\u0641\u0635\u0644 \u0633\u0648\u0645'      # فصل سوم
CH4 = '\u0641\u0635\u0644 \u0686\u0647\u0627\u0631\u0645'  # فصل چهارم

s = e = None
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    if p.Style.NameLocal == 'Heading 1':
        t = p.Range.Text.strip()
        print('  H1 p%-5d codepoints=%s' % (i, ' '.join('%04X' % ord(c) for c in t[:12])))

for i in list(range(755, 800)) + list(range(1090, 1105)):
    p = doc.Paragraphs.Item(i)
    st = p.Style.NameLocal
    txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
    print('p%-5d [%s] %s' % (i, st, txt[:95]))

    p = doc.Paragraphs.Item(i)
    st = p.Style.NameLocal
    if st.startswith('Heading'):
        txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
        lvl = st[-1]
        print('  ' * (int(lvl) - 1) + '[' + st + '] ' + txt[:80])

print('\n--- formulas / captions sanity ---')
for i in range(s, e):
    p = doc.Paragraphs.Item(i)
    txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
    if txt.startswith(('MAPE =', 'rel_L2', 'Pr[M', '\u0394w ', 'w\u0303(', 'clip:')) or 'جدول ' in txt[:10]:
        print('   p' + str(i) + ': ' + txt[:90])

doc.Close(False)
word.Quit()
print('\nVERIFY DONE')
