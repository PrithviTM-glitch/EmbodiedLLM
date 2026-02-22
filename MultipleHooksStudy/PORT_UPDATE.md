# Port Allocation Update - 21 February 2026

## Changes Made

Updated all notebooks to use **sequential ports starting from 9001**, matching the working configuration from `ablation_state_encoder_pass0.ipynb`.

---

## New Port Allocation

### Pi0 & RDT-1B (3 Benchmarks Each)

**Baseline Experiments**:
- LIBERO: **9001-9010** (10 parallel trials)
- VLABench: **9011-9020** (10 parallel trials)
- MetaWorld: **9021-9030** (10 parallel trials)

**Ablation Experiments**:
- LIBERO: **9031-9040** (10 parallel trials)
- VLABench: **9041-9050** (10 parallel trials)
- MetaWorld: **9051-9060** (10 parallel trials)

### Evo-1 (2 Benchmarks)

**Baseline Experiments**:
- LIBERO: **9001-9010** (10 parallel trials)
- MetaWorld: **9011-9020** (10 parallel trials)

**Ablation Experiments**:
- LIBERO: **9021-9030** (10 parallel trials)
- MetaWorld: **9031-9040** (10 parallel trials)

---

## Previous Allocation (Before Update)

### Old Port Scheme
- Baseline benchmarks: 8xxx ports (8001, 8101, 8201)
- Ablation studies: 9xxx ports (9001, 9101, 9201)

### Why Changed?
- User confirmed 9001 onwards worked in previous testing
- Sequential allocation is cleaner and easier to track
- Matches working configuration from `ablation_state_encoder_pass0.ipynb`
- No port conflicts in sequential scheme

---

## Files Updated

1. **notebooks/pi0_complete.ipynb**
   - BASELINE_PORTS: 8001 → 9001
   - VLA_PORTS: 8101 → 9011
   - MW_PORTS: 8201 → 9021
   - ABLATION_PORTS_LIBERO: 9001 → 9031
   - ABLATION_PORTS_VLA: 9101 → 9041
   - ABLATION_PORTS_MW: 9201 → 9051

2. **notebooks/rdt_1b_complete.ipynb**
   - Markdown documentation updated to reflect new ports
   - Code structure references Pi0 notebook (inherits port definitions)
   - Default ablation port: 9001 → 9031

3. **notebooks/evo1_complete.ipynb**
   - BASELINE_PORTS: 8001 → 9001
   - MW_BASELINE_PORTS: 8101 → 9011
   - Markdown updated: ablation ports 9001-9010 → 9021-9030
   - Markdown updated: ablation ports 9101-9110 → 9031-9040

4. **VALIDATION_REPORT.md**
   - Port allocation section updated
   - Quick reference guide updated

---

## Verification

Run notebooks/verify cells to confirm:
```python
# Pi0 Complete
BASELINE_PORTS = range(9001, 9001 + NUM_TRIALS)  # 9001-9010 ✅
VLA_PORTS = range(9011, 9011 + NUM_TRIALS)  # 9011-9020 ✅
MW_PORTS = range(9021, 9021 + NUM_TRIALS)  # 9021-9030 ✅
ABLATION_PORTS_LIBERO = range(9031, 9031 + NUM_TRIALS)  # 9031-9040 ✅
ABLATION_PORTS_VLA = range(9041, 9041 + NUM_TRIALS)  # 9041-9050 ✅
ABLATION_PORTS_MW = range(9051, 9051 + NUM_TRIALS)  # 9051-9060 ✅

# Evo-1 Complete
BASELINE_PORTS = range(9001, 9001 + NUM_TRIALS)  # 9001-9010 ✅
MW_BASELINE_PORTS = range(9011, 9011 + NUM_TRIALS)  # 9011-9020 ✅
# Ablation: 9021-9030 (LIBERO), 9031-9040 (MetaWorld) ✅
```

---

## Benefits of New Scheme

1. **Sequential**: Easy to track and understand (9001, 9002, ..., 9060)
2. **No gaps**: Efficient use of port range
3. **Proven**: Matches working configuration that user confirmed
4. **Clear separation**: First N ports for one type, next N for another
5. **Scalable**: Easy to add more benchmarks at 9061+

---

## Scripts Created

- **update_ports.py**: Main script that updated all port definitions
- **fix_ports.py**: Fixed comment inconsistencies after initial update

Both scripts preserved in repository for documentation.

---

✅ **All notebooks updated and validated**
