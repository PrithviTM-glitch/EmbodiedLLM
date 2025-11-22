# Quick Start: Running Evo-1 Evaluation

## ✅ Setup Complete

Both environments are configured and ready:
- **libero** environment (Python 3.8.13) - for LIBERO benchmark
- **Evo1** environment (Python 3.10) - for Evo-1 model

## 🚀 Run Evaluation in 2 Steps

### Terminal 1: Start Server
```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark
./scripts/run_evo1_server.sh
```
Wait for "Server started on port 9000"

### Terminal 2: Start Client
```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark
./scripts/run_libero_client.sh
```

## 📊 Expected Results
- **Target: 94.8% success rate** (from paper)
- 40 tasks across 4 LIBERO suites
- Evaluation may take several hours

## 📖 Full Documentation
See `docs/evo-1/RUNNING.md` for detailed instructions and troubleshooting.
