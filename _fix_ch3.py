import io, re, sys

PATH = r'd:\proposal\Code\Adaptive Single-layer deepseek\_ch3_content.py'
src = io.open(PATH, encoding='utf-8').read()

# Repair any occurrences of a stray double backslash before the closing quote
bad = '\\"),'
if bad in src:
    src = src.replace(bad, '"),')
    io.open(PATH, 'w', encoding='utf-8').write(src)
    print('fixed stray backslash')
else:
    print('nothing to fix')

# Validate
import ast
ast.parse(io.open(PATH, encoding='utf-8').read())
print('SYNTAX OK')
