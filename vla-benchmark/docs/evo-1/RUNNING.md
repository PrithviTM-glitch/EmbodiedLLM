# Running LIBERO Evaluation with Evo-1

## Environment Setup Summary

✅ **LIBERO Environment** (`libero`)
- Python: 3.8.13
- PyTorch: 2.4.1 (with MPS support)
- Packages: libero, robosuite 1.4.0, robomimic 0.3.0, websockets, pyyaml, bddl 3.6.0
- Config: `~/.libero/config.yaml` created with default paths
- Status: All dependencies installed and verified

✅ **Evo-1 Environment** (`Evo1`)
- Python: 3.10
- PyTorch: 2.5.1 (with MPS support)
- Packages: transformers, timm, accelerate, diffusers, websockets, websocket-client
- Checkpoint: Downloaded (1.4GB) at `checkpoints/libero/`
- Status: All dependencies installed and verified

## Running the Evaluation

### Step 1: Start Evo-1 Server (Terminal 1)

```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark
./scripts/run_evo1_server.sh
```

The server will:
- Load the Evo-1 model checkpoint (1.4GB)
- Start a WebSocket server on port 9000
- Wait for client connections

**Expected output:**
```
🚀 Starting Evo-1 Server...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Environment: Evo1 (Python 3.10)
Server Port: 9000
Checkpoint: checkpoints/libero/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loading model...
Server started on port 9000
```

### Step 2: Start LIBERO Client (Terminal 2)

In a **new terminal window**:

```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark
./scripts/run_libero_client.sh
```

The client will:
- Connect to the Evo-1 server via WebSocket
- Run evaluation on 4 LIBERO task suites
- Display progress and results

**Expected output:**
```
🔬 Starting LIBERO Benchmark Client...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Environment: libero (Python 3.8.13)
Server URL: ws://0.0.0.0:9000
Task Suites: libero_spatial, libero_object, libero_goal, libero_10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Evaluating task suite: libero_spatial
Task 1/10: pick_up_the_black_bowl_on_the_plate_drying_rack_and_place_it_on_the_plate
...
```

### Alternative: Manual Execution

If you prefer to run commands directly:

**Terminal 1 (Server):**
```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/Evo_1/scripts
conda activate Evo1
python Evo1_server.py
```

**Terminal 2 (Client):**
```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/LIBERO_evaluation
conda activate libero
python libero_client_4tasks.py
```

## Expected Results

According to the Evo-1 paper:
- **LIBERO-90 Success Rate: 94.8%**

Your results may vary slightly due to:
- Random seed variations
- MPS vs CUDA performance differences (we're using MPS on Apple Silicon)
- Lack of Flash Attention (requires CUDA)

## Task Suites Evaluated

1. **LIBERO-Spatial** (10 tasks)
   - Focus on spatial reasoning and object manipulation
   
2. **LIBERO-Object** (10 tasks)
   - Object-centric manipulation tasks
   
3. **LIBERO-Goal** (10 tasks)
   - Goal-conditioned manipulation
   
4. **LIBERO-10** (10 tasks)
   - Diverse long-horizon tasks

**Total: 40 tasks across 4 suites**

## Troubleshooting

### Server Won't Start
- Check if port 9000 is already in use: `lsof -i :9000`
- Verify Evo1 environment: `conda activate Evo1 && python -c "import torch; print(torch.__version__)"`

### Client Connection Issues
- Ensure server is running and showing "Server started on port 9000"
- Check WebSocket connection: The client should show "Connected to server" message
- Verify libero environment: `conda activate libero && python -c "import libero; import robosuite"`

### Performance Issues
- MPS (Apple Silicon) may be slower than CUDA GPUs
- Evaluation may take several hours depending on hardware
- Consider reducing the number of episodes per task for testing

### Robosuite Warnings
The warning about missing private macro file is non-critical:
```
[robosuite WARNING] No private macro file found!
[robosuite WARNING] To setup, run: python .../setup_macros.py
```
You can safely ignore this or run the setup script if preferred.

## Results Location

Results will be saved to:
```
/Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/results/evo1_libero/
```

Look for files like:
- `eval_results_[timestamp].json` - Detailed results per task
- `summary_[timestamp].txt` - Overall success rates

## Next Steps

After completing Evo-1 evaluation:
1. Document results in `results/evo1_results.md`
2. Set up next VLA model (Pi0.5, ECoT-Lite, SmolVLA, or OCTO)
3. Compare results across models
4. Generate visualization/plots of success rates
