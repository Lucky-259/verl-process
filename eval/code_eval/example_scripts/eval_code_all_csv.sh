#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1

PROJECT_NAME=${1}
EXPERIMENT_NAME=${2}
# CHECKPOINT_ROOT="/mnt/luoyingfeng/changkaiyan/verl-process/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
# OUTPUT_DIR=eval/eval_results
CHECKPOINT_ROOT="/mnt/hdfs/if_au/saves/cky/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
OUTPUT_DIR=/mnt/hdfs/if_au/saves/cky/eval_results
CSV_FILE="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/all_checkpoints_summary_code.csv"
GLOBAL_CSV="$OUTPUT_DIR/all_jobs_summary_code.csv"

mkdir -p "$(dirname "$CSV_FILE")"
mkdir -p "$(dirname "$GLOBAL_CSV")"

# 写入 CSV 表头（如果文件不存在）
if [ ! -f "$CSV_FILE" ]; then
    echo "Step,Dataset,Path,Pass@1,Pass@K,AvgLen" > "$CSV_FILE"
fi
if [ ! -f "$GLOBAL_CSV" ]; then
    echo "Project,Experiment,Step,Dataset,Path,Pass@1,Pass@K,Avg@K,AvgLen" > "$GLOBAL_CSV"
fi

# ================= 查找 Checkpoints =================
echo "Looking for checkpoints in $CHECKPOINT_ROOT..."
if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "Warning: Directory not found: $CHECKPOINT_ROOT"
    echo "Skipping evaluation for this experiment."
    exit 0
fi

CHECKPOINT_DIRS=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -V)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "No global_step directories found in $CHECKPOINT_ROOT"
    exit 1
fi

for STEP_DIR in $CHECKPOINT_DIRS; do

    STEP_NAME=$(basename "$STEP_DIR")

    # 定义 FSDP 源目录和目标 HF 目录
    FSDP_DIR="$STEP_DIR/actor"
    MODEL="$STEP_DIR/actor/huggingface"

    echo "========================================================"
    echo "Checkpoint: $STEP_NAME"
    echo "FSDP Source: $FSDP_DIR"
    echo "Target Model Path: $MODEL"
    echo "========================================================"

    # ------------------------------------------------------------------
    # 检查并合并 safetensors
    # ------------------------------------------------------------------
    if [ ! -d "$FSDP_DIR" ]; then
        echo "Error: FSDP source directory not found: $FSDP_DIR. Skipping..."
        continue
    fi

    if [ -d "$MODEL" ]; then
        SAFE_COUNT=$(find "$MODEL" -maxdepth 1 -name "*.safetensors" | wc -l)
    else
        SAFE_COUNT=0
    fi

    if [ "$SAFE_COUNT" -gt 0 ]; then
        echo "[Check] Safetensors found in $MODEL. Skipping merge."
    else
        echo "[Merge] No safetensors found. Starting merge process..."
        python scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir "$FSDP_DIR" \
            --target_dir "$MODEL"

        if [ $? -ne 0 ]; then
            echo "Error: Model merge failed for $STEP_NAME. Skipping evaluation."
            continue
        fi

        SAFE_COUNT_AFTER=$(find "$MODEL" -maxdepth 1 -name "*.safetensors" | wc -l)
        if [ "$SAFE_COUNT_AFTER" -eq 0 ]; then
            echo "Error: Merge script ran but no .safetensors found in $MODEL. Skipping."
            continue
        fi

        echo "[Merge] Successfully merged to $MODEL"
    fi

    # ================= HumanEval =================
    HUMAN_SAVE_DIR="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/$STEP_NAME/human_eval"
    HUMAN_MARKER_JSON="$HUMAN_SAVE_DIR/samples_results.json"

    echo "---- Dataset: human_eval ----"
    if [ -f "$HUMAN_MARKER_JSON" ]; then
        echo "[Skip human_eval] Found existing result file: $HUMAN_MARKER_JSON"
    else
        echo "running human_eval evaluation"
        python eval/code_eval/MBPP_Humaneval/eval/Coding/human_eval/evaluate_human_eval.py \
          --model "$MODEL" \
          --save_dir "$HUMAN_SAVE_DIR" \
          --num-samples-per-task 1 \
          --temperature 0.6
    fi

    HUMAN_PASS_JSON="$HUMAN_SAVE_DIR/samples_results.json"
    HUMAN_LEN_JSON="$HUMAN_SAVE_DIR/output_token_lengths.json"

    if [ -f "$HUMAN_PASS_JSON" ] && [ -f "$HUMAN_LEN_JSON" ]; then
        PASS1=$(python3 -c "import json; d=json.load(open('$HUMAN_PASS_JSON')); print(d['pass@pass@1'])")
        AVGLEN=$(python3 -c "import json; d=json.load(open('$HUMAN_LEN_JSON')); print(d['average_length'])")
        echo "$STEP_NAME,humaneval,$MODEL,$PASS1,$PASS1,$AVGLEN" >> "$CSV_FILE"
        echo "$PROJECT_NAME,$EXPERIMENT_NAME,$STEP_NAME,humaneval,$MODEL,$PASS1,$PASS1,$PASS1,$AVGLEN" >> "$GLOBAL_CSV"
    else
        echo "Warning: Missing HumanEval JSON for $STEP_NAME"
    fi

    # ================= MBPP =================
    MBPP_SAVE_DIR="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/$STEP_NAME/mbpp"
    MBPP_MARKER_JSON="$MBPP_SAVE_DIR/mbpp_completion.json"

    echo "---- Dataset: mbpp ----"
    if [ -f "$MBPP_MARKER_JSON" ]; then
        echo "[Skip mbpp] Found existing result file: $MBPP_MARKER_JSON"
    else
        echo "running mbpp evaluation"
        python eval/code_eval/MBPP_Humaneval/eval/Coding/mbpp/evaluate_mbpp.py \
          --model "$MODEL" \
          --input_data eval/code_eval/MBPP_Humaneval/eval/Coding/mbpp/new_mbpp.json \
          --save_dir "$MBPP_SAVE_DIR"
    fi

    MBPP_TXT="$MBPP_SAVE_DIR/result.txt"
    if [ -f "$MBPP_TXT" ]; then
        ACC=$(python3 -c "import ast; d=ast.literal_eval(open('$MBPP_TXT').readline()); print(d['accuracy'])")
        AVG=$(python3 -c "import ast; d=ast.literal_eval(open('$MBPP_TXT').readline()); print(d['average_tokens'])")
        echo "$STEP_NAME,mbpp,$MODEL,$ACC,$ACC,$AVG" >> "$CSV_FILE"
        echo "$PROJECT_NAME,$EXPERIMENT_NAME,$STEP_NAME,mbpp,$MODEL,$ACC,$ACC,$ACC,$AVG" >> "$GLOBAL_CSV"
    else
        echo "Warning: Missing MBPP results txt for $STEP_NAME"
    fi

done

echo "CSV summary generated at $CSV_FILE"
echo "Global CSV summary updated at $GLOBAL_CSV"