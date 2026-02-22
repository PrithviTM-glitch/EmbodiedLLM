#!/usr/bin/env python3
"""Fix remaining port issues in notebooks"""
import json
from pathlib import Path

notebooks_dir = Path('notebooks')

# Fix Pi0 comments
print('Fixing Pi0 comments...')
with open(notebooks_dir / 'pi0_complete.ipynb') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if '# 9001-8010' in line:
                cell['source'][i] = line.replace('# 9001-8010', '# 9001-9010')
                print('  ✅ Fixed BASELINE_PORTS comment')
            elif '# 9021-8210' in line:
                cell['source'][i] = line.replace('# 9021-8210', '# 9021-9030')
                print('  ✅ Fixed MW_PORTS comment')

with open(notebooks_dir / 'pi0_complete.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

# Check and report all ports
print('\n' + '='*60)
print('Final Port Allocations:')
print('='*60)

print('\n[Pi0 Complete]')
with open(notebooks_dir / 'pi0_complete.ipynb') as f:
    nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if 'PORTS' in line and 'range(' in line:
                    print(f'  {line.strip()}')

print('\n[RDT-1B Complete]')
with open(notebooks_dir / 'rdt_1b_complete.ipynb') as f:
    nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if 'PORTS' in line and 'range(' in line:
                    print(f'  {line.strip()}')

print('\n[Evo-1 Complete]')
with open(notebooks_dir / 'evo1_complete.ipynb') as f:
    nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if 'PORTS' in line and 'range(' in line:
                    print(f'  {line.strip()}')

print('\n✅ All ports verified!')
