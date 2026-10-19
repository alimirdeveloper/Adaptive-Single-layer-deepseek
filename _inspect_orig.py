import os, sys, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
# Inspect the ORIGINAL file
TARGET = os.path.join(DIR, [f for f in os.listdir(DIR) if f.endswith('.docx') and os.path.getsize(os.path.join(DIR, f)) == 975309][0])

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(TARGET, ReadOnly=True)

print('total paragraphs =', doc.Paragraphs.Count)
print()
print('=== p252..p262 of ORIGINAL ===')
for i in range(252, 263):
    p = doc.Paragraphs.Item(i)
    pf = p.Range.ParagraphFormat
    f = p.Range.Font
    txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
    print('p%-4d [%-12s] align=%s rtl=%s bold=%s size=%s name=%s' %
          (i, p.Style.NameLocal, pf.Alignment, pf.ReadingOrder, f.Bold, f.Size, f.Name))
    print('      %s' % txt[:100])

print()
print('=== p757..p765 of ORIGINAL (chapter 3 start) ===')
for i in range(757, 766):
    p = doc.Paragraphs.Item(i)
    txt = p.Range.Text.replace('\r', '').replace('\x07', '').strip()
    print('p%-4d [%-12s] %s' % (i, p.Style.NameLocal, txt[:80]))

doc.Close(False)
word.Quit()
