#!/bin/bash
# Run Evo-1 Server for LIBERO evaluation
# This script starts the WebSocket server that loads the Evo-1 model and serves predictions

set -e  # Exit on error

echo "🚀 Starting Evo-1 Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment: Evo1 (Python 3.10)"
echo "Server Port: 9000"
echo "Checkpoint: checkpoints/libero/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Activate Evo1 environment and run server
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/Evo_1/scripts

conda run -n Evo1 python Evo1_server.py

echo ""
echo "✅ Server stopped"
