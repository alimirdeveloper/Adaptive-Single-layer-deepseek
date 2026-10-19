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

print('=== original: every heading (H1/H2/H3) with index ===')
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    st = p.Style.NameLocal
    if st in ('Heading 1', 'Heading 2', 'Heading 3'):
        txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
        print('p%-5d [%-10s] %s' % (i, st, txt[:75]))

doc.Close(False)
word.Quit()
