import os
import re

GUARD = 'warnings.filterwarnings("ignore", category=RuntimeWarning)'
count = already = skipped = 0

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('docs', 'scripts')]
    for fn in files:
        if not fn.endswith('.py') or fn == 'main.py':
            continue
        p = os.path.join(root, fn)
        with open(p, encoding='utf-8') as fh:
            src = fh.read()
        if 'import warnings' in src:
            already += 1
            continue
        if 'import numpy' not in src:
            skipped += 1
            continue
        m = re.search(r'^import numpy[^\n]*$', src, re.M)
        if not m:
            skipped += 1
            continue
        insert = '\nimport warnings\n' + GUARD + '\n'
        src = src[:m.end()] + insert + src[m.end():]
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(src)
        count += 1

print('guarded:', count, '| already:', already, '| skipped(non-numpy):', skipped)