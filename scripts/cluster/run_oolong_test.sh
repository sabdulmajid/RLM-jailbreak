#!/bin/bash
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/slurm_nfs/a6abdulm/projects/RLM_jailbreak
#SBATCH --output=/mnt/slurm_nfs/a6abdulm/projects/RLM_jailbreak/oolong_test_%j.out
#SBATCH --error=/mnt/slurm_nfs/a6abdulm/projects/RLM_jailbreak/oolong_test_%j.err

echo "================================================================"
echo "OOLONG-RLM Benchmark Test"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Working Directory: $(pwd)"
echo "================================================================"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set HuggingFace cache to NFS (CRITICAL: Must be before any Python import)
export HF_HOME=/mnt/slurm_nfs/a6abdulm/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/slurm_nfs/a6abdulm/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/slurm_nfs/a6abdulm/.cache/huggingface/transformers

echo "Python version:"
python --version
echo ""

echo "HuggingFace cache location:"
echo "  HF_HOME=$HF_HOME"
echo ""

# Install required packages if needed
echo "Ensuring required packages are installed..."
pip install -q datasets ollama huggingface_hub 2>&1 | grep -v "Requirement already satisfied" || true
echo ""

echo "================================================================"
echo "Starting OOLONG-RLM Test"
echo "================================================================"
echo ""

# Run with first example and limited chunks for initial test
# To test full capability, remove --max-chunks flag
python scripts/data/test_oolong_loader.py --example 0 --max-chunks 10

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "================================================================"

exit $EXIT_CODE
