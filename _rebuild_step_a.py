# -*- coding: utf-8 -*-
"""
STEP A: create the working copy and delete ONLY the old Chapter 3 body.

Boundaries are the Heading 1 paragraphs of the chapter bodies:
    Heading1 "فصل سوم"  at p760  -> start
    Heading1 "فصل چهارم" at p858 -> end (exclusive)
The earlier attempt used the chapter-summary block (Normal style, p255)
which also contains the text "فصل چهارم", and therefore swallowed Ch4/Ch5.
"""
import os
import sys
import glob
import shutil

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import win32com.client as win32

DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
ORIG_SIZE = 975309

orig = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == ORIG_SIZE and '_' not in os.path.basename(p)]
if len(orig) != 1:
    print('ERROR: cannot uniquely identify the original'); sys.exit(1)
ORIG = orig[0]
COPY = ORIG[:-5] + '_' + '\u0641\u0635\u0644\u06F3' + '.docx'
shutil.copyfile(ORIG, COPY)
print('ORIG =', ORIG)
print('COPY =', COPY)

CH3H = '\u0641\u0635\u0644 \u0633\u0648\u0645'              # "فصل سوم"
CH4H = '\u0641\u0635\u0644 \u0686\u0647\u0627\u0631\u0645'  # "فصل چهارم"

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(COPY)
print('paragraphs =', doc.Paragraphs.Count)

b3 = b4 = None
for i in range(1, doc.Paragraphs.Count + 1):
    p = doc.Paragraphs.Item(i)
    if p.Style.NameLocal != 'Heading 1':
        continue
    t = p.Range.Text
    if CH3H in t and b3 is None:
        b3 = i
    elif CH4H in t and b3 is not None:
        b4 = i
        break

print('H1 Ch3 at p%s | H1 Ch4 at p%s' % (b3, b4))
if not b3 or not b4:
    print('ERROR: boundaries not found'); doc.Close(False); word.Quit(); sys.exit(1)

# Heading paragraphs are padded with trailing spaces before the mark, so we
# delete whole paragraph spans using offset 0 as a hard lower bound. This
# removes the Ch3 heading paragraph AND its body in one shot.
start = doc.Paragraphs.Item(b3 - 1).Range.End      # end of the paragraph before Ch3
end = doc.Paragraphs.Item(b4).Range.End            # end of the Ch4 heading paragraph
print('Ch3 heading span : [%d, %d)'
      % (doc.Paragraphs.Item(b3).Range.Start, doc.Paragraphs.Item(b3).Range.End))
print('delete span      : [%d, %d)' % (start, end))
print('deleting [%d,%d) = %d chars, %d paragraphs' % (start, end, end - start, b4 - b3))

probe = doc.Range(start, end)
print('  tables=%d inlineShapes=%d' % (probe.Tables.Count, probe.InlineShapes.Count))
assert probe.Tables.Count == 0 and probe.InlineShapes.Count == 0, 'unexpected objects in range'

probe.Delete()
print('paragraphs now =', doc.Paragraphs.Count)

# The Ch4 heading paragraph was consumed by the range above (its mark is the
# last character we deleted).  Re-create it verbatim as a Heading 1 paragraph
# at exactly the same position so Step B can anchor on it.
doc.Range(start, start).InsertAfter(CH4H + '\r')
newp = doc.Paragraphs.Item(b3)
print('after insert, p%s raw = %r  len=%d'
      % (b3, newp.Range.Text, len(newp.Range.Text)))
print('expected len =', len(CH4H) + 1, '| CH4H =', CH4H)
if CH4H not in newp.Range.Text:
    # InsertAfter on a collapsed range can split the text; use TypeText.
    doc.Range(start, start).Delete()
    r = doc.Range(start, start)
    r.InsertBefore(CH4H + '\r')
    newp = doc.Paragraphs.Item(b3)
    print('retry, p%s raw = %r' % (b3, newp.Range.Text))
newp.Style = doc.Styles('Heading 1')
print('recreated p%s: [%s] %r'
      % (b3, newp.Style.NameLocal, newp.Range.Text[:40]))
assert newp.Style.NameLocal == 'Heading 1' and CH4H in newp.Range.Text

doc.Save()
doc.Close(True)
word.Quit()
print('STEP A OK')
