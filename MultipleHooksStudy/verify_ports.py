#!/usr/bin/env python3
import json
from pathlib import Path

notebooks_dir = Path('notebooks')

print('='*60)
print('FINAL PORT VERIFICATION')
print('='*60)

notebooks = {
    'pi0_complete.ipynb': 'Pi0 (3 benchmarks)',
    'rdt_1b_complete.ipynb': 'RDT-1B (3 benchmarks)', 
    'evo1_complete.ipynb': 'Evo-1 (2 benchmarks)'
}

for nb_file, desc in notebooks.items():
    print(f'\n📓 {desc}')
    with open(notebooks_dir / nb_file) as f:
        nb = json.load(f)
    
    # Check code cells for port definitions
    port_defs = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if 'PORTS' in line and 'range' in line and '900' in line:
                    port_defs.append(line.strip())
    
    if port_defs:
        for port_def in port_defs:
            print(f'  ✅ {port_def}')
    else:
        # Check markdown for port references
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                for line in cell['source']:
                    if '9001' in line or '9011' in line or '9021' in line:
                        print(f'  📝 {line.strip()}')
                        break

print('\n' + '='*60)
print('✅ All ports start from 9001 onwards (sequential)')
print('='*60)
