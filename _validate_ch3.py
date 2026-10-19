import io, ast

PATH = r'd:\proposal\Code\Adaptive Single-layer deepseek\_ch3_content.py'
src = io.open(PATH, encoding='utf-8').read()
ns = {}
exec(compile(src, PATH, 'exec'), ns)
CH3 = ns['CH3']
print('CH3 items:', len(CH3))
from collections import Counter
print(Counter(s for s, _ in CH3))
