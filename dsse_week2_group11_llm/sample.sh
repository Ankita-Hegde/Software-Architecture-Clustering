#!/bin/bash

# ==============================================================================
# SLURM RESOURCE REQUESTS
# These lines (prefixed with #SBATCH) tell the cluster what hardware you need.
# ==============================================================================

# A descriptive name for your job in the queue (visible via 'squeue -u [YOUR_USERNAME]')
#SBATCH --job-name="[GROUP11-DSSE-WEEK2]"

# Maximum wall-clock time (Hours:Minutes:Seconds). 
# Your job will be terminated if it exceeds this.
#SBATCH --time=02:00:00

# This requests 2x NVIDIA A100 GPUs.
#SBATCH --gres=gpu:a100:2 

# Specifies the queue/partition. 'gpu' is standard for AI tasks.
#SBATCH --partition=gpu

# CPU RAM request to load the model weights into memory before they are transferred to the GPUs.
#SBATCH --mem=240G

# Where the console output (also errors) will be saved. 
# %j is a placeholder that SLURM replaces with the unique Job ID.
#SBATCH --output=[group11_dsse_week2]_%j.log


# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================

# Clears any modules loaded by default on the login node to prevent conflicts.
module purge 

# Loads the specific Python version. GCCcore indicates the compiler used to build it.
# search available Python versions in the HPC doc.
module load lang/Python/3.10.4-GCCcore-11.3.0

# Loads the CUDA toolkit (Compiler and Libraries). 
# This is essential for PyTorch to communicate with the A100 GPUs.
# search the available CUDA versions in the HPC doc.
module load system/CUDA/12.4.0

# Activates your pre-built Virtual Environment.
# Remember: Always build your venv on the login node BEFORE submitting this script.
source /pc2/users/s/sambhav/dsse_week2_group11_llm/venv/bin/activate


# ==============================================================================
# 2. OPTIMIZATIONS & SECRETS
# ==============================================================================

# Tells Hugging Face where to store/find model weights. 
export HF_HOME=/pc2/groups/hpc-prf-dssecs/huggingface_cache
# export HF_HOME=/pc2/users/s/sambhav/dsse_week2_group11_llm/huggingface_cache

# Memory optimization for PyTorch. 
# 'expandable_segments' helps prevent "Out of Memory" (OOM) errors by managing 
# how the GPU memory is allocated more efficiently.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Your Hugging Face Access Token. 
# Required for downloading models.
export HF_TOKEN="<HF_TOKEN_HERE>"


# ==============================================================================
# 3. EXECUTION
# ==============================================================================

echo "Starting..."
# Executes your Python script. 
# Because we used #SBATCH --output, all print statements in sample.py 
# will appear in your .log file in real-time.
python sample.py

# --- IMPORTANT ---
# Submitting: Run 'sbatch sample.sh' to put this in the queue.
# Checking Job Details: Run 'scontrol show job [JOBID]'
# Checking Job Start Time: Run 'scontrol show job [JOBID] | grep StartTime'
# Checking Progress: Run 'tail -f [UR_LOG_FILE_NAME]_[JOBID].log' to watch the log file in real-time.
# Cancel a Job: If you see an error in the log, run 'scancel [JOBID]' immediately to free up the GPUs.
