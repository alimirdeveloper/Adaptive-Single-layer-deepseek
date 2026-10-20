# -*- coding: utf-8 -*-
"""Read-only structural dump of a .docx using only the standard library.

A .docx is a ZIP; word/document.xml holds the body. This prints, for every
top-level paragraph, its style id and its text, so the chapter structure can be
verified without Word, python-docx or pywin32.
"""
import sys
import zipfile
import xml.etree.ElementTree as ET

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PATH = sys.argv[1]
ONLY_HEADINGS = '--headings' in sys.argv
_pos = [a for a in sys.argv[2:] if not a.startswith('--')]
LO = int(_pos[0]) if len(_pos) > 0 else 1
HI = int(_pos[1]) if len(_pos) > 1 else 10 ** 9

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with zipfile.ZipFile(PATH) as z:
    xml = z.read('word/document.xml')
root = ET.fromstring(xml)
body = root.find(W + 'body')

# styleId -> human name (Heading 1 etc. are usually named exactly that)
names = {}
try:
    with zipfile.ZipFile(PATH) as z:
        sroot = ET.fromstring(z.read('word/styles.xml'))
    for st in sroot.iter(W + 'style'):
        sid = st.get(W + 'styleId')
        nm = st.find(W + 'name')
        if sid and nm is not None:
            names[sid] = nm.get(W + 'val')
except KeyError:
    pass


def para_text(p):
    out = []
    for node in p.iter():
        if node.tag == W + 't':
            out.append(node.text or '')
        elif node.tag == W + 'tab':
            out.append('\t')
        elif node.tag == W + 'br':
            out.append('\n')
    return ''.join(out)


def para_style(p):
    ppr = p.find(W + 'pPr')
    if ppr is None:
        return ''
    ps = ppr.find(W + 'pStyle')
    return ps.get(W + 'val') if ps is not None else ''


idx = 0
tables = 0
counts = {}
for child in body:
    if child.tag == W + 'tbl':
        tables += 1
        continue
    if child.tag != W + 'p':
        continue
    idx += 1
    sid = para_style(child)
    name = names.get(sid, sid) or 'Normal'
    counts[name] = counts.get(name, 0) + 1
    if ONLY_HEADINGS and not name.lower().startswith('heading'):
        continue
    if not (LO <= idx <= HI):
        continue
    txt = para_text(child).replace('\r', '').strip()
    print('p%-5d [%-12s] %s' % (idx, name, txt[:90]))

print()
print('paragraphs = %d   tables = %d' % (idx, tables))
print('style counts:', {k: v for k, v in sorted(counts.items())})
