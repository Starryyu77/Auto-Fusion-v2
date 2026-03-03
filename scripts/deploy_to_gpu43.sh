#!/bin/bash
# ============================================================
# Deploy AutoFusion 2.0 to NTU GPU43 Cluster
# ============================================================

set -e

# Configuration
SERVER="s125mdg43_10@gpu43.dynip.ntu.edu.sg"
REMOTE_DIR="/usr1/home/s125mdg43_10/AutoFusion_v2"
LOCAL_DIR="."

echo "=========================================="
echo "AutoFusion 2.0 - GPU43 Deployment"
echo "=========================================="
echo "Server: $SERVER"
echo "Remote: $REMOTE_DIR"
echo "=========================================="

# 1. Check SSH connectivity
echo ""
echo ">>> Testing SSH connection..."
ssh -o ConnectTimeout=5 $SERVER "echo 'SSH OK'" || {
    echo "Error: Cannot connect to $SERVER"
    echo "Please ensure VPN is connected"
    exit 1
}

# 2. Create remote directory structure
echo ""
echo ">>> Creating remote directory structure..."
ssh $SERVER "mkdir -p $REMOTE_DIR/{src,configs,scripts,docs,tests,logs,results}"

# 3. Sync code (excluding data, results, cache)
echo ""
echo ">>> Syncing code to GPU43..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='.cache/' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    $LOCAL_DIR/ \
    $SERVER:$REMOTE_DIR/

# 4. Set up environment on remote
echo ""
echo ">>> Setting up environment on GPU43..."
ssh $SERVER << 'ENDSSH'
cd /usr1/home/s125mdg43_10/AutoFusion_v2

# Check if conda env exists
if ! conda env list | grep -q "autofusion2"; then
    echo "Creating conda environment..."
    conda create -n autofusion2 python=3.10 -y
fi

# Activate and install dependencies
source ~/anaconda3/etc/profile.d/conda.sh
conda activate autofusion2

echo "Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers datasets accelerate
pip install -q pyyaml matplotlib numpy pillow
pip install -q openai requests tqdm

echo "Environment ready!"
ENDSSH

# 5. Create data directory symlink
echo ""
echo ">>> Setting up data directories..."
ssh $SERVER << ENDSSH
mkdir -p /usr1/home/s125mdg43_10/data/{mmmu,vqa_rad,robo_sense}
ln -sf /usr1/home/s125mdg43_10/data $REMOTE_DIR/data
ENDSSH

# 6. Verify deployment
echo ""
echo ">>> Verifying deployment..."
ssh $SERVER << ENDSSH
cd /usr1/home/s125mdg43_10/AutoFusion_v2
echo "Files deployed:"
ls -la
echo ""
echo "Python version:"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate autofusion2
python --version
echo ""
echo "CUDA available:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
ENDSSH

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. SSH to GPU43: ssh $SERVER"
echo "  2. Activate env: conda activate autofusion2"
echo "  3. Prepare data: python scripts/prepare_datasets.py"
echo "  4. Run tests: python tests/test_minimal.py"
echo ""
echo "Or run remotely:"
echo "  ssh $SERVER 'cd $REMOTE_DIR && conda activate autofusion2 && python src/main.py --help'"
