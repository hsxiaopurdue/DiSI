#!/bin/bash
#
# Run DiSI llm backdoor training experiments
# 
# Combinations:
#   - aggregation_method: {mean, majority_vote, topk}
#   - task: {jailbreak, refusal}
#   - attack_method: {badnet, sleeper, vpi, mtba, ctba}
#
# Total experiments: 3 x 2 x 5 = 30
#
# Usage:
#   ./run_DiSI_llm_backdoor.sh           # Run all experiments
#   ./run_DiSI_llm_backdoor.sh --dry-run # Show commands without running
#

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parameters
AGGREGATION_METHODS=("mean" "majority_vote" "topk")
TASKS=("jailbreak" "refusal")
ATTACK_METHODS=("badnet" "sleeper" "vpi" "mtba" "ctba")

# Logging
LOG_DIR="./logs/DiSI_llm_backdoor"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check for dry-run mode
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - Commands will be printed but not executed ==="
fi

# Count total experiments
TOTAL_EXPERIMENTS=$((${#AGGREGATION_METHODS[@]} * ${#TASKS[@]} * ${#ATTACK_METHODS[@]}))
CURRENT_EXP=0

echo "============================================================"
echo "  Federated Backdoor Training - All Experiments"
echo "============================================================"
echo "  Aggregation Methods: ${AGGREGATION_METHODS[*]}"
echo "  Tasks: ${TASKS[*]}"
echo "  Attack Methods: ${ATTACK_METHODS[*]}"
echo "  Total Experiments: $TOTAL_EXPERIMENTS"
echo "  Log Directory: $LOG_DIR"
echo "============================================================"
echo ""

# Function to run a single experiment
run_experiment() {
    local agg_method=$1
    local task=$2
    local attack=$3
    
    local config_file="configs/${task}/llama2_7b_chat/llama2_7b_${task}_${attack}_lora_federated.yaml"
    local log_file="${LOG_DIR}/${task}_${attack}_${agg_method}_${TIMESTAMP}.log"
    
    CURRENT_EXP=$((CURRENT_EXP + 1))
    
    echo ""
    echo "============================================================"
    echo "  Experiment $CURRENT_EXP / $TOTAL_EXPERIMENTS"
    echo "============================================================"
    echo "  Task:        $task"
    echo "  Attack:      $attack"
    echo "  Aggregation: $agg_method"
    echo "  Config:      $config_file"
    echo "  Log:         $log_file"
    echo "============================================================"
    
    # Check if config file exists
    if [[ ! -f "$config_file" ]]; then
        echo "  ERROR: Config file not found: $config_file"
        echo "  Skipping this experiment..."
        return 0
    fi
    
    # Build the command with auto output suffix to organize by aggregation method
    local cmd="python train_DiSI_llm_backdoor.py $config_file --aggregation $agg_method --auto_output_suffix"
    
    echo "  Command: $cmd"
    echo ""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would execute: $cmd"
    else
        echo "  Starting experiment at $(date)"
        echo "  Logging to: $log_file"
        
        # Run the experiment and log output
        if $cmd 2>&1 | tee "$log_file"; then
            echo ""
            echo "  SUCCESS: Experiment completed at $(date)"
        else
            echo ""
            echo "  FAILED: Experiment failed at $(date)"
            echo "  Check log file: $log_file"
        fi
    fi
}

# Main execution
echo "Starting experiments at $(date)"
echo ""
echo "Running experiments sequentially..."

for agg_method in "${AGGREGATION_METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        for attack in "${ATTACK_METHODS[@]}"; do
            run_experiment "$agg_method" "$task" "$attack"
        done
    done
done

echo ""
echo "============================================================"
echo "  All experiments completed at $(date)"
echo "============================================================"
echo ""
echo "Summary:"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Log directory: $LOG_DIR"
echo ""
echo "To analyze results, check the backdoor_weight directory:"
echo "  ls -la backdoor_weight/"
echo ""
