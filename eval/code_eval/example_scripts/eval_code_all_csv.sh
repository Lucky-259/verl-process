#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

PROJECT_NAME=${1}
EXPERIMENT_NAME=${2}
CHECKPOINT_ROOT="/mnt/luoyingfeng/changkaiyan/verl-process/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
OUTPUT_DIR=eval/eval_results
CSV_FILE="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/all_checkpoints_summary.csv"
GLOBAL_CSV="$OUTPUT_DIR/all_jobs_summary_8k_avgk.csv"

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
    MODEL="$STEP_DIR/actor/huggingface"
    echo "model name: $MODEL"
    temperature=0.6

    # ================= HumanEval =================
    echo "running human_eval evaluation"
    python eval/code_eval/MBPP_Humaneval/eval/Coding/human_eval/evaluate_human_eval.py \
      --model $MODEL \
      --save_dir "$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/$STEP_NAME/human_eval" \
      --num-samples-per-task 1 \
      --temperature $temperature

    # HumanEval JSON 文件路径
    HUMAN_PASS_JSON="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/$STEP_NAME/human_eval/samples_results.json"
    HUMAN_LEN_JSON="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/$STEP_NAME/human_eval/output_token_lengths.json"

    if [ -f "$HUMAN_PASS_JSON" ] && [ -f "$HUMAN_LEN_JSON" ]; then
        PASS1=$(python3 -c "import json; d=json.load(open('$HUMAN_PASS_JSON')); print(d['pass@pass@1'])")
        AVGLEN=$(python3 -c "import json; d=json.load(open('$HUMAN_LEN_JSON')); print(d['average_length'])")
        echo "$STEP_NAME,humaneval,$MODEL,$PASS1,$PASS1,$AVGLEN" >> "$CSV_FILE"

        # 追加到全局 CSV
        echo "$PROJECT_NAME,$EXPERIMENT_NAME,$STEP_NAME,humaneval,$MODEL,$PASS1,$PASS1,$PASS1,$AVGLEN" >> "$GLOBAL_CSV"
    else
        echo "Warning: Missing HumanEval JSON for $STEP_NAME"
    fi

    # ================= MBPP =================
    echo "running mbpp evaluation"
    MBPP_SAVE_DIR="$OUTPUT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}/$STEP_NAME/mbpp"
    python eval/code_eval/MBPP_Humaneval/eval/Coding/mbpp/evaluate_mbpp.py \
      --model $MODEL \
      --input_data eval/code_eval/MBPP_Humaneval/eval/Coding/mbpp/new_mbpp.json \
      --save_dir "$MBPP_SAVE_DIR"

    MBPP_TXT="$MBPP_SAVE_DIR/result.txt"
    if [ -f "$MBPP_TXT" ]; then
        ACC=$(python3 -c "import ast; d=ast.literal_eval(open('$MBPP_TXT').readline()); print(d['accuracy'])")
        AVG=$(python3 -c "import ast; d=ast.literal_eval(open('$MBPP_TXT').readline()); print(d['average_tokens'])")
        echo "$STEP_NAME,mbpp,$MODEL,$ACC,$ACC,$AVG" >> "$CSV_FILE"

        # 追加到全局 CSV
        echo "$PROJECT_NAME,$EXPERIMENT_NAME,$STEP_NAME,mbpp,$MODEL,$ACC,$ACC,$ACC,$AVG" >> "$GLOBAL_CSV"
    else
        echo "Warning: Missing MBPP results txt for $STEP_NAME"
    fi

done

echo "CSV summary generated at $CSV_FILE"
echo "Global CSV summary updated at $GLOBAL_CSV"