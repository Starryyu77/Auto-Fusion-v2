#!/bin/bash
# ============================================================
# Run AutoFusion 2.0 Experiments on GPU43 (Serial Execution)
# All 4 models: Kimi, GLM, Qwen, DeepSeek
# ============================================================

set -e

SERVER="s125mdg43_10@gpu43.dynip.ntu.edu.sg"
REMOTE_DIR="/usr1/home/s125mdg43_10/AutoFusion_v2"
API_KEY="sk-fa81e2c1077c4bf5a159c2ca5ddcf200"

# Experiment matrix
# Format: "scenario|model|gpu_id|output_dir"
EXPERIMENTS=(
    # Scenario A: MMMU - All 4 models
    "mmmu|kimi-k2.5|0|results/scenario_a_kimi"
    "mmmu|glm-5|0|results/scenario_a_glm"
    "mmmu|qwen-max|0|results/scenario_a_qwen"
    "mmmu|deepseek-v3|0|results/scenario_a_deepseek"

    # Scenario B: VQA-RAD - All 4 models
    "vqa_rad|kimi-k2.5|1|results/scenario_b_kimi"
    "vqa_rad|glm-5|1|results/scenario_b_glm"
    "vqa_rad|qwen-max|1|results/scenario_b_qwen"
    "vqa_rad|deepseek-v3|1|results/scenario_b_deepseek"

    # Scenario C: Edge - All 4 models
    "robo_sense|kimi-k2.5|2|results/scenario_c_kimi"
    "robo_sense|glm-5|2|results/scenario_c_glm"
    "robo_sense|qwen-max|2|results/scenario_c_qwen"
    "robo_sense|deepseek-v3|2|results/scenario_c_deepseek"
)

echo "=========================================="
echo "AutoFusion 2.0 - GPU43 Experiment Runner"
echo "=========================================="
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Models: Kimi-K2.5, GLM-5, Qwen-Max, DeepSeek-V3"
echo "Scenarios: MMMU, VQA-RAD, RoboSense"
echo "Execution: SERIAL (one at a time)"
echo "API Key: ${API_KEY:0:10}..."
echo "=========================================="

# Function to run single experiment
run_experiment() {
    local scenario=$1
    local model=$2
    local gpu=$3
    local output=$4

    echo ""
    echo "=========================================="
    echo "Running: Scenario=$scenario | Model=$model | GPU=$gpu"
    echo "=========================================="

    ssh $SERVER << ENDSSH
        set -e
        cd $REMOTE_DIR
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate autofusion2

        export CUDA_VISIBLE_DEVICES=$gpu
        export ALIYUN_API_KEY="$API_KEY"
        export PYTHONPATH="$REMOTE_DIR:":$PYTHONPATH

        echo "Environment:"
        echo "  Python: \$(python --version)"
        echo "  CUDA: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "  GPU: $gpu"
        echo ""

        # Map scenario names to configs
        case "$scenario" in
            mmmu)
                CONFIG="configs/scenario_a_mmmu.yaml"
                ;;
            vqa_rad)
                CONFIG="configs/scenario_b_medical.yaml"
                ;;
            robo_sense)
                CONFIG="configs/scenario_c_edge.yaml"
                ;;
            *)
                echo "Unknown scenario: $scenario"
                exit 1
                ;;
        esac

        echo "Config: \$CONFIG"
        echo "Output: $output"
        echo ""

        # Run experiment
        python src/main.py \
            --config \$CONFIG \
            --llm-model $model \
            --output-dir $output \
            --max-iterations 200

        echo ""
        echo "Experiment complete: $scenario/$model"
        echo "Results: $output"
ENDSSH

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Success: $scenario/$model"
    else
        echo "✗ Failed: $scenario/$model (exit code: $exit_code)"
        FAILED_EXPERIMENTS+=("$scenario|$model")
    fi

    return $exit_code
}

# Main execution loop
FAILED_EXPERIMENTS=()
COMPLETED=0
TOTAL=${#EXPERIMENTS[@]}

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r scenario model gpu output <<< "$exp"

    ((COMPLETED++))
    echo ""
    echo "[Experiment $COMPLETED/$TOTAL]"

    run_experiment "$scenario" "$model" "$gpu" "$output"

    # Brief pause between experiments to cool down
    if [ $COMPLETED -lt $TOTAL ]; then
        echo ""
        echo "Pausing 30s before next experiment..."
        sleep 30
    fi
done

# Summary
echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo "Completed: $COMPLETED/$TOTAL"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for failed in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  ✗ $failed"
    done
    exit 1
else
    echo "All experiments successful!"
    exit 0
fi
