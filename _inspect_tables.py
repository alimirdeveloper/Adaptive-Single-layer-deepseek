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

print('total tables:', doc.Tables.Count)
for ti in range(1, doc.Tables.Count + 1):
    t = doc.Tables.Item(ti)
    # find the paragraph index of table start
    st = t.Range.Start
    idx = None
    for i in range(1, doc.Paragraphs.Count + 1):
        if doc.Paragraphs.Item(i).Range.Start >= st:
            idx = i; break
    first = t.Range.Cells.Item(1) if t.Range.Cells.Count > 0 else None
    txt = ''
    try:
        txt = t.Range.Text.replace('\r', ' | ').replace('\x07','')[:100]
    except Exception as ex:
        txt = 'ERR ' + str(ex)
    print('Table %d: rows=%d cols=%d (~para %s) : %s' %
          (ti, t.Rows.Count, t.Columns.Count, idx, txt))

doc.Close(False)
word.Quit()
