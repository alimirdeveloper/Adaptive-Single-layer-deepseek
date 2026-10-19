import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
cand = [p for p in glob.glob(os.path.join(DIR, '*.docx')) if os.path.getsize(p) == 986430]
TARGET = cand[0]

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(TARGET, ReadOnly=True)

print('=== ALL Heading 1 paragraphs (top-level) ===')
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    if p.Style.NameLocal == 'Heading 1':
        # skip paragraphs inside table cells
        in_tbl = p.Range.Tables.Count > 0
        txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
        print('p%-5d intable=%s  %s' % (i, in_tbl, txt[:70]))

print()
print('=== search for the literal text of chapter 4 heading ===')
for needle, label in [('\u0686\u0647\u0627\u0631\u0645', 'chaharom'),
                      ('\u067e\u0646\u062c\u0645', 'panjom')]:
    hits = []
    for i in range(1, doc.Paragraphs.Count + 1):
        p = doc.Paragraphs.Item(i)
        if p.Range.Tables.Count > 0:
            continue
        if needle in p.Range.Text:
            hits.append((i, p.Style.NameLocal,
                         p.Range.Text.replace('\r', '').replace('\x07', '').strip()[:60]))
    print(label, '->', hits[:10])

doc.Close(False)
word.Quit()
