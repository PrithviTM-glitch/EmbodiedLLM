#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --nodelist=TC1N05
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=2
#SBATCH --time=60
#SBATCH --job-name=EmbodiedLLM_test
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/logs/error_%x_%j.err

set -euo pipefail

module load anaconda
module load cuda/12.1
source activate TestEnv

cd /tc1home/FYP/prithvi004/EmbodiedLLM/test
python test.py