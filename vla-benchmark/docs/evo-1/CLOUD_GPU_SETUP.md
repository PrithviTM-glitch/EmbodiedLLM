# Running Evo-1 on Cloud GPU with Local Client

This guide explains how to run the Evo-1 server on a cloud GPU (Google Colab, Lambda Labs, etc.) while keeping the LIBERO client on your local Mac.

## Why Use Cloud GPU?

### ❌ Mac GPU (MPS) Issues
- **No bfloat16 support**: Forced to use float32 (2× memory, slower)
- **No Flash Attention**: 3-4× slower attention computation
- **No CUDA optimizations**: Missing critical accelerations
- **Expected slowdown**: 10-20× slower than CUDA GPU
- **Evaluation time**: Days instead of hours

### ✅ Cloud GPU Benefits
- **Full CUDA support**: bfloat16, Flash Attention, optimized kernels
- **Faster inference**: 10-20× speedup over Mac
- **Reasonable cost**: $0.50-$1.50/hour
- **Evaluation time**: 2-3 hours for LIBERO-90

## Option 1: Google Colab (Recommended)

### Setup (5 minutes)

1. **Open the Colab notebook**:
   - Navigate to `docs/evo-1/colab_setup.ipynb`
   - Upload to Google Colab or open in Colab directly
   - Or use this link: [Open in Colab](https://colab.research.google.com/)

2. **Enable GPU**:
   - Go to `Runtime > Change runtime type`
   - Select `GPU` (T4 is fine, A100 is faster)
   - Click `Save`

3. **Get ngrok token** (free):
   - Visit https://dashboard.ngrok.com/get-started/your-authtoken
   - Sign up for free account
   - Copy your auth token

4. **Run all cells**:
   - Click `Runtime > Run all`
   - Enter your ngrok token when prompted
   - Wait 3-5 minutes for setup

5. **Get your server URL**:
   - After cell 8 completes, copy the ngrok URL
   - Example: `tcp://2.tcp.ngrok.io:12345`

### Connect Local Client

1. **Update client configuration**:
   ```bash
   cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/LIBERO_evaluation
   ```

2. **Edit `libero_client_4tasks.py`**:
   ```python
   # Change this line (around line 15):
   # OLD:
   SERVER_URL = "ws://0.0.0.0:9000"
   
   # NEW (use your ngrok URL):
   SERVER_URL = "ws://2.tcp.ngrok.io:12345"  # Replace with your URL
   ```

3. **Run the client**:
   ```bash
   cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark
   ./scripts/run_libero_client.sh
   ```

### Colab Pricing
- **Free tier**: 
  - T4 GPU (limited hours)
  - Disconnects after 12 hours
  - May need to reconnect
  
- **Colab Pro** ($10/month):
  - Longer runtimes
  - Better GPUs (V100/A100)
  - Worth it for this project!

## Option 2: Lambda Labs

### Setup

1. **Create account**: https://lambdalabs.com/

2. **Launch instance**:
   ```bash
   # Choose GPU (RTX A6000 recommended)
   # Cost: ~$0.50-0.80/hour
   ```

3. **SSH into instance**:
   ```bash
   ssh ubuntu@<instance-ip>
   ```

4. **Clone and setup**:
   ```bash
   # Install dependencies
   sudo apt update
   sudo apt install -y python3-pip git
   
   # Clone Evo-1
   git clone https://github.com/MINT-SJTU/Evo-1.git
   cd Evo-1
   
   # Install requirements
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install transformers==4.45.2 timm==1.0.9 accelerate==1.2.1 diffusers==0.31.0
   pip3 install websockets opencv-python pillow numpy fvcore
   pip3 install flash-attn --no-build-isolation
   
   # Download checkpoint
   pip3 install huggingface-hub
   python3 -c "
   from huggingface_hub import snapshot_download
   snapshot_download('MINT-SJTU/Evo1_LIBERO', local_dir='./checkpoints/libero')
   "
   ```

5. **Apply fixes from our setup**:
   - Copy the fixed `Evo1_server.py` from your Mac
   - Copy the fixed `internvl3_embedder.py`
   - Or manually apply the device selection fixes

6. **Start server**:
   ```bash
   cd Evo_1/scripts
   python3 Evo1_server.py
   ```

### SSH Tunnel Setup

Instead of ngrok, use SSH tunneling:

```bash
# On your Mac, create tunnel:
ssh -L 9000:localhost:9000 ubuntu@<lambda-instance-ip>

# Keep this terminal open
# Now your local client can connect to ws://localhost:9000
```

**Advantages**:
- More secure than ngrok
- No third-party service
- Better performance

**Keep in mind**:
- Must keep SSH connection alive
- Need to maintain terminal window

## Option 3: RunPod

Similar to Lambda Labs, but with more GPU options:

1. **Visit**: https://www.runpod.io/
2. **Deploy**: RTX 4090 (~$0.50/hour)
3. **Setup**: Same as Lambda Labs above
4. **Connect**: Use SSH tunnel or expose port

## Updating Client for Remote Connection

### For ngrok (Colab):

Edit `libero_client_4tasks.py`:
```python
# Line ~15
SERVER_URL = "ws://2.tcp.ngrok.io:12345"  # Your ngrok URL
```

### For SSH tunnel (Lambda/RunPod):

Edit `libero_client_4tasks.py`:
```python
# Line ~15
SERVER_URL = "ws://localhost:9000"  # Tunneled through SSH
```

## Running the Evaluation

1. **Start server** (on cloud GPU):
   - Colab: Run notebook cells
   - Lambda/RunPod: `python3 Evo1_server.py`

2. **Verify connection**:
   ```bash
   # On your Mac
   nc -zv localhost 9000  # For SSH tunnel
   # OR
   nc -zv 2.tcp.ngrok.io 12345  # For ngrok
   ```

3. **Run client** (on your Mac):
   ```bash
   cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark
   ./scripts/run_libero_client.sh
   ```

4. **Monitor progress**:
   - Watch client terminal for task progress
   - Monitor GPU usage on server
   - Expected: 2-3 hours for full evaluation

## Expected Results

With proper GPU (CUDA):
- **Success rate**: ~94.8% (from paper)
- **Evaluation time**: 2-3 hours
- **GPU memory**: 6-8GB VRAM
- **Tasks**: 40 total (4 suites × 10 tasks)

## Cost Comparison

| Platform | GPU | Cost/Hour | Full Eval Cost | Pros | Cons |
|----------|-----|-----------|----------------|------|------|
| Colab Free | T4 | $0 | $0 | Free! | Limited hours, may disconnect |
| Colab Pro | A100 | $0 | $10/month | Fast, reliable | Monthly subscription |
| Lambda Labs | A6000 | $0.80 | ~$2.40 | Fast, flexible | Pay per use |
| RunPod | RTX 4090 | $0.50 | ~$1.50 | Cheapest | Setup complexity |

## Troubleshooting

### Connection Refused
```bash
# Check server is running
# On cloud: ps aux | grep Evo1_server
# Check port is open
# On cloud: lsof -i :9000
```

### Ngrok Connection Issues
```bash
# Verify ngrok is running
# On Colab: check cell 8 output
# Test connection locally first
```

### SSH Tunnel Drops
```bash
# Use autossh for persistent connection
brew install autossh
autossh -M 0 -L 9000:localhost:9000 ubuntu@<instance-ip>
```

### Server Crashes
```bash
# Check GPU memory
nvidia-smi

# Check logs
tail -f /tmp/evo1_server.log

# Restart with explicit device
export CUDA_VISIBLE_DEVICES=0
python3 Evo1_server.py
```

## Documenting Results

After evaluation completes:

1. **Save results**:
   ```bash
   # Results should be in:
   # /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/results/evo1_libero/
   ```

2. **Create summary**:
   ```bash
   cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/docs/evo-1
   # Document success rates per task suite
   # Compare with paper results (94.8%)
   # Note any issues or observations
   ```

3. **Commit results**:
   ```bash
   git add results/ docs/
   git commit -m "Add Evo-1 LIBERO evaluation results"
   git push
   ```

## Best Practices

### 1. Don't Waste Time on Mac GPU
- Mac GPU (MPS) is 10-20× slower
- Missing critical features (Flash Attention, bfloat16)
- Save yourself hours of pain - use cloud GPU from the start

### 2. Start with Colab Free
- Test your setup before paying
- Verify connection works
- Run a few tasks to confirm

### 3. Use SSH Tunneling for Production
- More secure than ngrok
- Better performance
- No third-party dependencies

### 4. Monitor GPU Usage
```bash
# On cloud server
watch -n 1 nvidia-smi
```

### 5. Save Checkpoints
- Colab may disconnect
- Save intermediate results
- Resume if needed

## Summary

**Recommended workflow:**
1. ✅ Use Google Colab Pro ($10/month) for best experience
2. ✅ Use provided notebook for easy setup
3. ✅ Connect via ngrok (simplest) or SSH tunnel (more secure)
4. ✅ Run LIBERO client on your Mac
5. ✅ Complete evaluation in 2-3 hours
6. ✅ Document results and commit

**Avoid:**
- ❌ Running on Mac GPU (painfully slow)
- ❌ Trying to optimize MPS (not worth the effort)
- ❌ Using free Colab for full evaluation (may disconnect)

Good luck with your evaluation! 🚀
