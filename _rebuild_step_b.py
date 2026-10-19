# -*- coding: utf-8 -*-
"""
STEP B: insert Chapter 3 into the working copy, immediately before the
Heading 1 "f## #####" (chapter four), matching Chapters 1-2 formatting.

  Heading 1 : RIGHT(2),   space 18/4, line 13.9 exact, B Nazanin 20
  Heading 2 : RIGHT(2),   space  8/4, line 13.9 exact, B Nazanin 16
  Heading 3 : RIGHT(2),   space  8/4, line 13.9 exact, B Nazanin 14
  Heading 4 : RIGHT(2),   space  4/2, line 13.9 exact, B Nazanin 12
  Normal    : JUSTIFY(3), space 0/8, line 13.9 exact, size 12
  Caption   : CENTER(1),  space 0/10, line 12 single, size 9

All Persian strings are built from code points so the source file stays
pure ASCII -- writing Persian literals through the Windows shell corrupts
them.
"""
import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ch3_content import CH3, TABLES

import win32com.client as win32

# ---------------------------------------------------------------- literals
# All Persian strings are built from explicit code points so this source file
# stays pure ASCII (Persian literals written through this shell get corrupted).
def _w(*cps):
    return ''.join(chr(c) for c in cps)


CH3W = _w(0x641, 0x635, 0x644, 0x20, 0x633, 0x648, 0x645)                  # ### ###
CH4W = _w(0x641, 0x635, 0x644, 0x20, 0x686, 0x647, 0x627, 0x631, 0x645)   # ### #####
CH5W = _w(0x641, 0x635, 0x644, 0x20, 0x67E, 0x646, 0x62C, 0x645)          # ### ####
RESW = _w(0x641, 0x647, 0x631, 0x633, 0x62A, 0x20,
          0x645, 0x646, 0x627, 0x628, 0x639)                              # ##### #####

FARSI_LO, FARSI_HI = 0x600, 0x6FF


def heading_key(s):
    """Keep only Arabic/Persian letters and ASCII digits -- robust key."""
    return tuple(ord(c) for c in s
                 if FARSI_LO <= ord(c) <= FARSI_HI or c.isdigit())


CH3K, CH4K, CH5K, RESK = (heading_key(w) for w in (CH3W, CH4W, CH5W, RESW))

# ---------------------------------------------------------------- files
DIR = r'C:\Users\Asus\Desktop\uni\proposal git'
ORIG_SIZE = 975309
ORIG = [p for p in glob.glob(os.path.join(DIR, '*.docx'))
        if os.path.getsize(p) == ORIG_SIZE and '_' not in os.path.basename(p)]
if len(ORIG) != 1:
    print('ERROR: cannot uniquely identify the original:', ORIG); sys.exit(1)
COPY = ORIG[0][:-5] + '_' + chr(0x641) + chr(0x635) + chr(0x644) + \
    chr(0x6F3) + '.docx'
if not os.path.exists(COPY):
    print('ERROR: working copy missing, run _rebuild_step_a.py first'); sys.exit(1)
print('file =', COPY)

AL_RIGHT, AL_CENTER, AL_JUSTIFY = 2, 1, 3
LS_EXACT, LS_SINGLE = 4, 0
BODY_FONT = 'B Nazanin'

FMT = {
    'Heading 1': dict(align=AL_RIGHT,   before=18, after=4,  ls=LS_EXACT,  spacing=13.9, size=20),
    'Heading 2': dict(align=AL_RIGHT,   before=8,  after=4,  ls=LS_EXACT,  spacing=13.9, size=16),
    'Heading 3': dict(align=AL_RIGHT,   before=8,  after=4,  ls=LS_EXACT,  spacing=13.9, size=14),
    'Heading 4': dict(align=AL_RIGHT,   before=4,  after=2,  ls=LS_EXACT,  spacing=13.9, size=12),
    'Normal':    dict(align=AL_JUSTIFY, before=0,  after=8,  ls=LS_EXACT,  spacing=13.9, size=12),
    'Caption':   dict(align=AL_CENTER,  before=0,  after=10, ls=LS_SINGLE, spacing=12.0, size=9),
}

word = win32.gencache.EnsureDispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(COPY)
print('paragraphs before =', doc.Paragraphs.Count)


# ---------------------------------------------------------------- locating
def h1_map():
    """{index: heading_key(text)} for every Heading 1 paragraph."""
    out = []
    for i in range(1, doc.Paragraphs.Count + 1):
        p = doc.Paragraphs.Item(i)
        if p.Style.NameLocal == 'Heading 1':
            out.append((i, heading_key(p.Range.Text)))
    return out


hm = h1_map()
ch3_idx = next((i for i, k in hm if k.startswith(CH3K)), None)
ch4_idx = next((i for i, k in hm if k.startswith(CH4K)), None)
print('H1 keys:', [(i, k) for i, k in hm])
print('Ch3 @ p%s ; Ch4 @ p%s' % (ch3_idx, ch4_idx))
if not ch3_idx or not ch4_idx:
    print('ERROR: chapter headings not found'); doc.Close(False); word.Quit(); sys.exit(1)
assert ch4_idx == ch3_idx + 1, 'unexpected gap between Ch3 and Ch4'

sel = word.Selection
anchor = doc.Paragraphs.Item(ch3_idx).Range.End
sel.SetRange(anchor, anchor)
sel.Style = doc.Styles('Normal')
print('anchor = %d' % anchor)


# ---------------------------------------------------------------- helpers
def style_para(para, style_name):
    conf = FMT[style_name]
    pf = para.Range.ParagraphFormat
    pf.Alignment = conf['align']
    pf.SpaceBefore = conf['before']
    pf.SpaceAfter = conf['after']
    pf.LineSpacingRule = conf['ls']
    pf.LineSpacing = conf['spacing']
    pf.FirstLineIndent = 0
    pf.LeftIndent = 0
    pf.RightIndent = 0
    para.Range.Font.Size = conf['size']
    if style_name != 'Caption':
        para.Range.Font.Name = BODY_FONT
        para.Range.Font.Bold = 0


def new_paragraph(style_name):
    """Open a fresh empty paragraph at the cursor, then style it."""
    sel.TypeParagraph()
    p = sel.Paragraphs.Item(1)
    p.Style = doc.Styles(style_name)
    style_para(p, style_name)
    return p


def emit(text, style_name, bold=False):
    new_paragraph(style_name)
    sel.TypeText(text)
    if bold:
        sel.Font.Bold = True
    style_para(sel.Paragraphs.Item(1), style_name)


def emit_table(key):
    header, rows, widths = TABLES[key]
    ncols, nrows = len(header), len(rows) + 1
    new_paragraph('Normal')
    tbl = doc.Tables.Add(sel.Range, nrows, ncols)
    tbl.Borders.Enable = True
    tbl.Range.Font.Name = BODY_FONT
    tbl.Range.Font.Size = 10
    tbl.Range.ParagraphFormat.Alignment = AL_CENTER
    tbl.Range.ParagraphFormat.LineSpacingRule = LS_SINGLE
    tbl.Range.ParagraphFormat.SpaceBefore = 2
    tbl.Range.ParagraphFormat.SpaceAfter = 2

    for c, h in enumerate(header, start=1):
        cell = tbl.Cell(1, c)
        cell.Range.Text = h
        cell.Range.Font.Bold = True
        cell.Range.Font.Size = 10
        cell.Range.ParagraphFormat.Alignment = AL_CENTER
        cell.Shading.BackgroundPatternColor = 15921906

    for r, row in enumerate(rows, start=2):
        for c, val in enumerate(row, start=1):
            cell = tbl.Cell(r, c)
            cell.Range.Text = val
            cell.Range.Font.Bold = False
            cell.Range.Font.Size = 10
            cell.Range.ParagraphFormat.Alignment = AL_CENTER

    tbl.Rows.Item(1).HeadingFormat = True
    for c, w in enumerate(widths, start=1):
        tbl.Columns.Item(c).Width = w * 28.3465

    after = tbl.Range.End + 1
    sel.SetRange(after, after)
    sel.Style = doc.Styles('Normal')
    style_para(sel.Paragraphs.Item(1), 'Normal')


STYLE_MAP = {'h1': 'Heading 1', 'h2': 'Heading 2', 'h3': 'Heading 3',
             'h4': 'Heading 4', 'body': 'Normal', 'caption': 'Caption'}

print('emitting %d items ...' % len(CH3))
for n, (kind, text) in enumerate(CH3, start=1):
    if kind.startswith('table:'):
        emit_table(kind.split(':', 1)[1])
    elif kind == 'formula':
        emit(text, 'Normal', bold=True)
    else:
        emit(text, STYLE_MAP[kind])
    if n % 25 == 0:
        print('  ...', n, '/', len(CH3))

doc.Save()
print('paragraphs after =', doc.Paragraphs.Count)
print('tables after     =', doc.Tables.Count)

hm = h1_map()
print('H1 keys after:', hm)
for want, label in ((CH3K, 'Ch3'), (CH4K, 'Ch4'), (CH5K, 'Ch5'), (RESK, 'Refs')):
    assert any(k.startswith(want) for _, k in hm), 'missing H1: %s' % label
print('all chapter headings present')

# body sanity: the injected Ch3 body must not run into Ch4
idx3 = next(i for i, k in hm if k.startswith(CH3K))
idx4 = next(i for i, k in hm if k.startswith(CH4K))
print('Ch3 at p%d, Ch4 at p%d -> %d paragraphs injected'
      % (idx3, idx4, idx4 - idx3 - 1))

doc.Close(True)
word.Quit()
print('STEP B OK')
