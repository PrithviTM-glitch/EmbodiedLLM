#!/bin/bash
# Run LIBERO Client to evaluate Evo-1 on LIBERO tasks
# This script starts the benchmark client that connects to the Evo-1 server

set -e  # Exit on error

echo "🔬 Starting LIBERO Benchmark Client..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment: libero (Python 3.8.13)"
echo "Server URL: ws://0.0.0.0:9000"
echo "Task Suites: libero_spatial, libero_object, libero_goal, libero_10"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  Make sure the Evo-1 server is running before starting the client!"
echo "    (Run ./run_evo1_server.sh in another terminal)"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Activate libero environment and run client
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/LIBERO_evaluation

conda run -n libero python libero_client_4tasks.py

echo ""
echo "✅ Evaluation complete"
