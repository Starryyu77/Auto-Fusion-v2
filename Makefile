# AutoFusion 2.0 - GPU43 Makefile

.PHONY: help deploy test run monitor status sync-up sync-down

SERVER=s125mdg43_10@gpu43.dynip.ntu.edu.sg
REMOTE_DIR=/usr1/home/s125mdg43_10/AutoFusion_v2
API_KEY=sk-fa81e2c1077c4bf5a159c2ca5ddcf200

help:
	@echo "AutoFusion 2.0 - GPU43 Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make deploy      - Deploy code to GPU43"
	@echo "  make test        - Run unit tests on GPU43"
	@echo "  make run         - Start full experiment suite"
	@echo "  make monitor     - Monitor running experiments"
	@echo "  make status      - Check GPU43 status"
	@echo "  make sync-up     - Sync local code to GPU43"
	@echo "  make sync-down   - Sync results from GPU43"
	@echo "  make logs        - View recent logs"

deploy:
	@echo "Deploying to GPU43..."
	@bash scripts/deploy_to_gpu43.sh

test:
	@echo "Running tests on GPU43..."
	@ssh $(SERVER) "cd $(REMOTE_DIR) && source ~/anaconda3/etc/profile.d/conda.sh && conda activate autofusion2 && python tests/test_minimal.py"

run:
	@echo "Starting experiment suite..."
	@bash scripts/run_on_gpu43.sh

monitor:
	@echo "Monitoring GPU43..."
	@ssh $(SERVER) "watch -n 5 'nvidia-smi && echo \"=== Processes ===\" && ps aux | grep python | grep -v grep'"

status:
	@echo "Checking GPU43 status..."
	@ssh $(SERVER) "echo '=== GPU Status ===' && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv && echo '' && echo '=== Disk Usage ===' && df -h /usr1 && echo '' && echo '=== Recent Results ===' && ls -ltr $(REMOTE_DIR)/results 2>/dev/null | tail -5"

sync-up:
	@echo "Syncing local code to GPU43..."
	@rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='data/' --exclude='results/' ./ $(SERVER):$(REMOTE_DIR)/

sync-down:
	@echo "Syncing results from GPU43..."
	@mkdir -p results_from_gpu43
	@rsync -avz $(SERVER):$(REMOTE_DIR)/results/ ./results_from_gpu43/
	@echo "Results downloaded to: ./results_from_gpu43/"

logs:
	@echo "Recent logs from GPU43..."
	@ssh $(SERVER) "cd $(REMOTE_DIR) && find logs -name '*.log' -mtime -1 -exec tail -20 {} \; 2>/dev/null || echo 'No recent logs'"

# Quick commands for single experiment
run-mmmu-kimi:
	@ssh $(SERVER) "cd $(REMOTE_DIR) && source ~/anaconda3/etc/profile.d/conda.sh && conda activate autofusion2 && export ALIYUN_API_KEY=$(API_KEY) && export CUDA_VISIBLE_DEVICES=0 && python src/main.py --config configs/scenario_a_mmmu.yaml --llm-model kimi-k2.5 --output-dir results/scenario_a_kimi"

run-mmmu-glm:
	@ssh $(SERVER) "cd $(REMOTE_DIR) && source ~/anaconda3/etc/profile.d/conda.sh && conda activate autofusion2 && export ALIYUN_API_KEY=$(API_KEY) && export CUDA_VISIBLE_DEVICES=0 && python src/main.py --config configs/scenario_a_mmmu.yaml --llm-model glm-5 --output-dir results/scenario_a_glm"

# Data preparation on GPU43
prepare-data:
	@ssh $(SERVER) "cd $(REMOTE_DIR) && source ~/anaconda3/etc/profile.d/conda.sh && conda activate autofusion2 && python scripts/prepare_datasets.py"
