#!/usr/bin/env python3
"""Update port allocations in all notebooks to start from 9001 sequentially"""
import json
from pathlib import Path

def update_pi0_rdt_ports(notebook_path):
    """Update Pi0/RDT notebooks (3 benchmarks)"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    changes = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                # Update LIBERO baseline ports (8001 -> 9001)
                if 'BASELINE_PORTS = range(8001' in line:
                    old_line = cell['source'][i]
                    cell['source'][i] = line.replace('8001', '9001').replace('8001-8010', '9001-9010')
                    changes.append(f'LIBERO baseline: 8001 -> 9001')
                
                # Update VLABench ports (8101 -> 9011)
                elif 'VLA_PORTS = range(8101' in line:
                    cell['source'][i] = line.replace('8101', '9011').replace('8101-8110', '9011-9020')
                    changes.append(f'VLABench baseline: 8101 -> 9011')
                
                # Update MetaWorld baseline ports (8201 -> 9021)
                elif 'MW_PORTS = range(8201' in line:
                    cell['source'][i] = line.replace('8201', '9021').replace('8201-8210', '9021-9030')
                    changes.append(f'MetaWorld baseline: 8201 -> 9021')
                
                # Update LIBERO ablation ports (9001 -> 9031)
                elif 'ABLATION_PORTS_LIBERO = range(9001' in line:
                    cell['source'][i] = line.replace('9001', '9031')
                    changes.append(f'LIBERO ablation: 9001 -> 9031')
                
                # Update VLABench ablation ports (9101 -> 9041)
                elif 'ABLATION_PORTS_VLA = range(9101' in line:
                    cell['source'][i] = line.replace('9101', '9041')
                    changes.append(f'VLABench ablation: 9101 -> 9041')
                
                # Update MetaWorld ablation ports (9201 -> 9051)
                elif 'ABLATION_PORTS_MW = range(9201' in line:
                    cell['source'][i] = line.replace('9201', '9051')
                    changes.append(f'MetaWorld ablation: 9201 -> 9051')
                
                # Update default port in argparse (9001 -> 9031 for ablation)
                elif 'parser.add_argument("--port", type=int, default=9001)' in line:
                    cell['source'][i] = line.replace('default=9001', 'default=9031')
                    changes.append(f'Default ablation port: 9001 -> 9031')
        
        # Update markdown descriptions
        elif cell['cell_type'] == 'markdown':
            for i, line in enumerate(cell['source']):
                if '8001-8010 (LIBERO), 8101-8110 (VLA), 8201-8210 (MW)' in line:
                    cell['source'][i] = line.replace(
                        '8001-8010 (LIBERO), 8101-8110 (VLA), 8201-8210 (MW)',
                        '9001-9010 (LIBERO), 9011-9020 (VLA), 9021-9030 (MW)'
                    )
                    changes.append(f'Markdown baseline description updated')
                elif '9001-9010 (LIBERO), 9101-9110 (VLA), 9201-9210 (MW)' in line:
                    cell['source'][i] = line.replace(
                        '9001-9010 (LIBERO), 9101-9110 (VLA), 9201-9210 (MW)',
                        '9031-9040 (LIBERO), 9041-9050 (VLA), 9051-9060 (MW)'
                    )
                    changes.append(f'Markdown ablation description updated')
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    return changes

def update_evo1_ports(notebook_path):
    """Update Evo-1 notebook (2 benchmarks)"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    changes = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                # Update LIBERO baseline ports (8001 -> 9001)
                if 'BASELINE_PORTS = range(8001' in line:
                    cell['source'][i] = line.replace('8001', '9001').replace('8001-8010', '9001-9010')
                    changes.append(f'LIBERO baseline: 8001 -> 9001')
                
                # Update MetaWorld baseline ports (8101 -> 9011)
                elif 'MW_BASELINE_PORTS = range(8101' in line:
                    cell['source'][i] = line.replace('8101', '9011').replace('8101-8110', '9011-9020')
                    changes.append(f'MetaWorld baseline: 8101 -> 9011')
        
        # Update markdown descriptions
        elif cell['cell_type'] == 'markdown':
            for i, line in enumerate(cell['source']):
                if 'ports 9001-9010' in line and 'LIBERO ablation' in line:
                    cell['source'][i] = line.replace('9001-9010', '9021-9030')
                    changes.append(f'LIBERO ablation description: 9001-9010 -> 9021-9030')
                elif 'ports 9101-9110' in line and 'MetaWorld ablation' in line:
                    cell['source'][i] = line.replace('9101-9110', '9031-9040')
                    changes.append(f'MetaWorld ablation description: 9101-9110 -> 9031-9040')
    
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=2)
    
    return changes

# Update all notebooks
notebooks_dir = Path(__file__).parent / 'notebooks'

print('='*60)
print('Updating port allocations to 9001 onwards (sequential)')
print('='*60)

# Pi0 notebook
print('\n[1/3] Pi0 Complete Notebook')
print('  LIBERO: 9001-9010, VLABench: 9011-9020, MetaWorld: 9021-9030')
print('  (Ablation: LIBERO 9031-9040, VLA 9041-9050, MW 9051-9060)')
pi0_changes = update_pi0_rdt_ports(notebooks_dir / 'pi0_complete.ipynb')
for change in pi0_changes:
    print(f'  ✅ {change}')

# RDT notebook
print('\n[2/3] RDT-1B Complete Notebook')
print('  LIBERO: 9001-9010, VLABench: 9011-9020, MetaWorld: 9021-9030')
print('  (Ablation: LIBERO 9031-9040, VLA 9041-9050, MW 9051-9060)')
rdt_changes = update_pi0_rdt_ports(notebooks_dir / 'rdt_1b_complete.ipynb')
for change in rdt_changes:
    print(f'  ✅ {change}')

# Evo-1 notebook
print('\n[3/3] Evo-1 Complete Notebook')
print('  LIBERO: 9001-9010, MetaWorld: 9011-9020')
print('  (Ablation: LIBERO 9021-9030, MetaWorld 9031-9040)')
evo1_changes = update_evo1_ports(notebooks_dir / 'evo1_complete.ipynb')
for change in evo1_changes:
    print(f'  ✅ {change}')

print('\n' + '='*60)
print('✅ All notebooks updated!')
print('='*60)
