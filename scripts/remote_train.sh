#!/bin/bash
#
# Remote Training Pipeline for AI2MCP
# Automates: sync -> ssh -> train -> pull results
#
# Setup:
#   1. Run: ssh-copy-id xi58pizy@cip7g1.cip.cs.fau.de
#   2. chmod +x scripts/remote_train.sh
#
# Usage:
#   ./scripts/remote_train.sh sync      # Sync code to remote
#   ./scripts/remote_train.sh setup     # First-time setup on remote
#   ./scripts/remote_train.sh train     # Start training in tmux
#   ./scripts/remote_train.sh status    # Check training status
#   ./scripts/remote_train.sh pull      # Pull results back
#   ./scripts/remote_train.sh all       # Full pipeline

set -e

# Configuration
REMOTE_USER="xi58pizy"
REMOTE_HOST="cip7g1.cip.cs.fau.de"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
SSH_TERM="xterm-256color"  # Fix for ghostty/non-standard terminals
REMOTE_DIR="/proj/ciptmp/xi58pizy/AI2MCP"
LOCAL_DIR="$(dirname "$0")/.."
CONDA_ENV="ai2mcp"
TMUX_SESSION="train"
CONDA_PREFIX="/proj/ciptmp/xi58pizy/.conda"

# Training config
EPOCHS="${EPOCHS:-100}"
MAX_FRAMES="${MAX_FRAMES:-55000}"  # Full ALOHA dataset
OUTPUT_DIR="${OUTPUT_DIR:-experiments/remote_training}"
OUTPUT_DIR_A="${OUTPUT_DIR}/relational_gnn"
OUTPUT_DIR_C="${OUTPUT_DIR}/multimodal_gnn_${MAX_FRAMES}"  # Includes frame count for clarity
COMPARISON_DIR="${OUTPUT_DIR}/comparison_real"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Sync code to remote
sync_code() {
    log_info "Syncing code to ${REMOTE}:${REMOTE_DIR}..."
    rsync -avz --progress \
        --exclude='.venv' \
        --exclude='*.pt' \
        --exclude='experiments/' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
    log_info "Sync complete!"
}

# First-time setup on remote
setup_remote() {
    log_info "Setting up remote environment..."
    ssh "${REMOTE}" bash -l << 'REMOTE_SCRIPT'
set -e
WORK_DIR="/proj/ciptmp/xi58pizy"
CONDA_ENVS="${WORK_DIR}/.conda/envs"
ENV_PATH="${CONDA_ENVS}/ai2mcp"

echo "Loading modules..."
module load cuda/12.0
module load python3/anaconda-2024.07
module load tmux

# Source conda for activation
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Creating conda environment in ${CONDA_ENVS}..."
mkdir -p "${CONDA_ENVS}"
if [ ! -d "${ENV_PATH}" ]; then
    conda create --prefix "${ENV_PATH}" python=3.10 -y
else
    echo "Environment already exists"
fi

echo "Activating environment..."
conda activate "${ENV_PATH}"

echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing additional dependencies..."
pip install torch-geometric transformers timm tqdm structlog httpx

echo "Installing project..."
cd "${WORK_DIR}/AI2MCP"
pip install -e . || pip install -r requirements.txt 2>/dev/null || echo "Installing core deps only"

echo ""
echo "Verifying GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.version.cuda}')"

echo ""
echo "✅ Setup complete!"
echo "Conda env: ${ENV_PATH}"
REMOTE_SCRIPT
}

# Start training in tmux (MultiModalGNN on full dataset)
start_training() {
    log_info "Starting MultiModalGNN training on remote..."
    log_info "  - Dataset: lerobot/aloha_static_coffee"
    log_info "  - Frames: ${MAX_FRAMES} (use MAX_FRAMES=N to override)"
    log_info "  - Epochs: ${EPOCHS}"
    log_info "  - Output: ${OUTPUT_DIR_C}"
    
    ssh "${REMOTE}" bash -l << REMOTE_SCRIPT
WORK_DIR="/proj/ciptmp/xi58pizy"
CONDA_ENVS="\${WORK_DIR}/.conda/envs"

# Load modules
module load cuda/12.0 python3/anaconda-2024.07 tmux

# Kill existing session if any
tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true

# Start new tmux session with training
tmux new-session -d -s ${TMUX_SESSION} bash -c '
    WORK_DIR="/proj/ciptmp/xi58pizy"
    CONDA_ENVS="\${WORK_DIR}/.conda/envs"
    
    module load cuda/12.0 python3/anaconda-2024.07
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate "\${CONDA_ENVS}/ai2mcp"
    cd "\${WORK_DIR}/AI2MCP"
    
    mkdir -p ${OUTPUT_DIR_C}
    
    echo "============================================================"
    echo "MultiModalGNN Training"
    echo "============================================================"
    echo "Time: \$(date)"
    echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "Dataset: lerobot/aloha_static_coffee"
    echo "Frames: ${MAX_FRAMES}"
    echo "Epochs: ${EPOCHS}"
    echo "Output: ${OUTPUT_DIR_C}"
    echo "============================================================"
    echo ""
    
    python scripts/train_multimodal_gnn.py \
        --repo lerobot/aloha_static_coffee \
        --max-frames ${MAX_FRAMES} \
        --epochs ${EPOCHS} \
        --output ${OUTPUT_DIR_C} \
        2>&1 | tee ${OUTPUT_DIR_C}/training.log
    
    echo ""
    echo "============================================================"
    echo "Training Complete!"
    echo "Time: \$(date)"
    echo "============================================================"
    echo "Press any key to exit..."
    read
'

echo "Training started in tmux session: ${TMUX_SESSION}"
echo "To attach: ssh ${REMOTE} -t tmux attach -t ${TMUX_SESSION}"
REMOTE_SCRIPT
    
    log_info "Training started! Use './scripts/remote_train.sh status' to check progress"
}

# Full pipeline: Train both models + benchmark with real vision
start_full_pipeline() {
    log_info "Starting FULL training pipeline on remote..."
    log_info "  - RelationalGNN on full ALOHA (${MAX_FRAMES} frames)"
    log_info "  - MultiModalGNN on full ALOHA (${MAX_FRAMES} frames)"
    log_info "  - Comparison with REAL vision (GroundingDINO + ZoeDepth)"
    log_info "  - Epochs: ${EPOCHS}"
    log_info "Estimated time: 6-8 hours"
    
    ssh "${REMOTE}" bash -l << REMOTE_SCRIPT
WORK_DIR="/proj/ciptmp/xi58pizy"
CONDA_ENVS="\${WORK_DIR}/.conda/envs"

# Load modules
module load cuda/12.0 python3/anaconda-2024.07 tmux

# Kill existing session if any
tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true

# Start new tmux session with full pipeline
tmux new-session -d -s ${TMUX_SESSION} bash -c '
    set -e  # Exit on error
    WORK_DIR="/proj/ciptmp/xi58pizy"
    CONDA_ENVS="\${WORK_DIR}/.conda/envs"
    
    module load cuda/12.0 python3/anaconda-2024.07
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate "\${CONDA_ENVS}/ai2mcp"
    cd "\${WORK_DIR}/AI2MCP"
    
    mkdir -p ${OUTPUT_DIR_A} ${OUTPUT_DIR_C} ${COMPARISON_DIR}
    
    echo "================================================================"
    echo "=== FULL TRAINING PIPELINE ==="
    echo "================================================================"
    echo "Start: \$(date)"
    echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo ""
    
    # ===== PHASE 1: Train RelationalGNN =====
    echo ""
    echo "================================================================"
    echo "=== PHASE 1/3: Training RelationalGNN (Option A) ==="
    echo "================================================================"
    echo "Dataset: Full ALOHA (${MAX_FRAMES} frames)"
    echo "Epochs: ${EPOCHS}"
    echo ""
    
    python scripts/train_relational_gnn.py \
        --repo lerobot/aloha_static_coffee \
        --max-frames ${MAX_FRAMES} \
        --epochs ${EPOCHS} \
        --output ${OUTPUT_DIR_A} \
        2>&1 | tee ${OUTPUT_DIR_A}/training.log
    
    echo ""
    echo "Phase 1 complete: \$(date)"
    
    # ===== PHASE 2: Train MultiModalGNN =====
    echo ""
    echo "================================================================"
    echo "=== PHASE 2/3: Training MultiModalGNN (Option C) ==="
    echo "================================================================"
    echo "Dataset: Full ALOHA (${MAX_FRAMES} frames)"
    echo "Epochs: ${EPOCHS}"
    echo ""
    
    python scripts/train_multimodal_gnn.py \
        --repo lerobot/aloha_static_coffee \
        --max-frames ${MAX_FRAMES} \
        --epochs ${EPOCHS} \
        --output ${OUTPUT_DIR_C} \
        2>&1 | tee ${OUTPUT_DIR_C}/training.log
    
    echo ""
    echo "Phase 2 complete: \$(date)"
    
    # ===== PHASE 3: Comparison with REAL Vision =====
    echo ""
    echo "================================================================"
    echo "=== PHASE 3/3: Comparison Benchmark (REAL Vision) ==="
    echo "================================================================"
    echo "Using: GroundingDINO + ZoeDepth (honest latency measurement)"
    echo ""
    
    python scripts/compare_models.py \
        --repo lerobot/aloha_static_coffee \
        --frames 500 \
        --model-a ${OUTPUT_DIR_A}/best_model.pt \
        --model-c ${OUTPUT_DIR_C}/best_model.pt \
        --use-real-vision \
        --output ${COMPARISON_DIR} \
        2>&1 | tee ${COMPARISON_DIR}/benchmark.log
    
    echo ""
    echo "================================================================"
    echo "=== ALL PHASES COMPLETE ==="
    echo "================================================================"
    echo "End: \$(date)"
    echo ""
    echo "Results:"
    echo "  - RelationalGNN: ${OUTPUT_DIR_A}/best_model.pt"
    echo "  - MultiModalGNN: ${OUTPUT_DIR_C}/best_model.pt"
    echo "  - Comparison:    ${COMPARISON_DIR}/comparison_results.json"
    echo ""
    echo "Press any key to exit..."
    read
'

echo "Full pipeline started in tmux session: ${TMUX_SESSION}"
echo "To attach: ssh ${REMOTE} -t tmux attach -t ${TMUX_SESSION}"
REMOTE_SCRIPT
    
    log_info "Full pipeline started! Use './scripts/remote_train.sh status' to check progress"
}

# Check training status
check_status() {
    log_info "Checking training status..."
    ssh "${REMOTE}" bash -l << 'REMOTE_SCRIPT'
WORK_DIR="/proj/ciptmp/xi58pizy"
module load tmux 2>/dev/null

if tmux has-session -t train 2>/dev/null; then
    echo "✅ Training session is RUNNING"
    echo ""
    echo "=== Last 20 lines of output ==="
    tmux capture-pane -t train -p | tail -20
    echo ""
    echo "To attach: tmux attach -t train"
else
    echo "❌ No active training session"
    
    # Check if results exist
    RESULT_FILE="${WORK_DIR}/AI2MCP/experiments/remote_training/training_history.json"
    if [ -f "${RESULT_FILE}" ]; then
        echo ""
        echo "📊 Found completed training results:"
        head -20 "${RESULT_FILE}"
    fi
fi
REMOTE_SCRIPT
}

# Attach to training session
attach_session() {
    log_info "Attaching to training session..."
    TERM="${SSH_TERM}" ssh -t "${REMOTE}" "bash -lc 'module load tmux 2>/dev/null; tmux attach -t ${TMUX_SESSION}'"
}

# Pull results back to local
pull_results() {
    log_info "Pulling results from remote..."
    mkdir -p "${LOCAL_DIR}/experiments"
    rsync -avz --progress \
        "${REMOTE}:${REMOTE_DIR}/experiments/" \
        "${LOCAL_DIR}/experiments/"
    log_info "Results pulled to ${LOCAL_DIR}/experiments/"
}

# Full pipeline
full_pipeline() {
    sync_code
    start_training
    log_info "Training started. Run './scripts/remote_train.sh status' to monitor."
    log_info "When done, run './scripts/remote_train.sh pull' to get results."
}

# Interactive shell on remote
shell() {
    log_info "Opening shell on remote..."
    TERM="${SSH_TERM}" ssh -t "${REMOTE}" bash -l << 'EOF'
WORK_DIR="/proj/ciptmp/xi58pizy"
CONDA_ENVS="${WORK_DIR}/.conda/envs"

module load cuda/12.0 python3/anaconda-2024.07 tmux
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${CONDA_ENVS}/ai2mcp" 2>/dev/null || echo "Env not found, run setup first"
cd "${WORK_DIR}/AI2MCP" 2>/dev/null || cd "${WORK_DIR}"
exec bash
EOF
}

# Show help
show_help() {
    echo "Remote Training Pipeline for AI2MCP"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  sync       - Sync code to remote"
    echo "  setup      - First-time setup (create conda env, install deps)"
    echo "  train      - Start MultiModalGNN training in tmux"
    echo "  train-full - FULL PIPELINE: train both models + real vision benchmark"
    echo "  status     - Check training progress"
    echo "  attach     - Attach to training tmux session"
    echo "  pull       - Pull results back to local"
    echo "  shell      - Open interactive shell on remote"
    echo "  all        - Sync + train"
    echo ""
    echo "Environment variables:"
    echo "  EPOCHS=100       - Number of training epochs (default: 100)"
    echo "  MAX_FRAMES=55000 - Number of dataset frames (default: 55000 = full ALOHA)"
    echo "  OUTPUT_DIR=...   - Output directory on remote"
    echo ""
    echo "Examples:"
    echo "  $0 setup                     # First-time setup"
    echo "  $0 sync && $0 train-full     # Sync code and run FULL pipeline (~6-8h)"
    echo "  EPOCHS=50 $0 train           # Quick multimodal training"
    echo "  $0 status                    # Check progress"
    echo "  $0 pull                      # Get results"
    echo ""
    echo "Full Pipeline (train-full) runs:"
    echo "  1. Train RelationalGNN on full ALOHA (55k frames)"
    echo "  2. Train MultiModalGNN on full ALOHA (55k frames)"
    echo "  3. Benchmark with REAL vision (GroundingDINO + ZoeDepth)"
}

# Main
case "${1:-help}" in
    sync)       sync_code ;;
    setup)      setup_remote ;;
    train)      start_training ;;
    train-full) start_full_pipeline ;;
    status)     check_status ;;
    attach)     attach_session ;;
    pull)       pull_results ;;
    shell)      shell ;;
    all)        full_pipeline ;;
    *)          show_help ;;
esac

