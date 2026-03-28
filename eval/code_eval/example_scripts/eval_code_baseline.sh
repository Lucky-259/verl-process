#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1

BASELINE_NAME=${1}
# MODEL_ROOT="/mnt/luoyingfeng/model_card"
MODEL_ROOT="/mnt/hdfs/if_au/models"
MODEL_PATH="$MODEL_ROOT/${BASELINE_NAME}"

# OUTPUT_DIR="eval/baseline_res"
OUTPUT_DIR="/mnt/hdfs/if_au/saves/cky/eval_results"

CSV_FILE="$OUTPUT_DIR/${BASELINE_NAME}/all_models_baseline_code.csv"
GLOBAL_CSV="$OUTPUT_DIR/all_jobs_baseline_code.csv"

EXP_TAG="baseline"
TEMPERATURE=0.6

if [ -z "$BASELINE_NAME" ]; then
    echo "Usage: bash eval_baseline_code.sh <BASELINE_NAME>"
    exit 1
fi

echo "Model path: $MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/${BASELINE_NAME}"
mkdir -p "$(dirname "$GLOBAL_CSV")"

# 写入 CSV 表头（如果文件不存在）
if [ ! -f "$CSV_FILE" ]; then
    echo "ModelName,Dataset,Path,Pass@1,Pass@K,AvgLen" > "$CSV_FILE"
fi
if [ ! -f "$GLOBAL_CSV" ]; then
    echo "Experiment,ModelName,Dataset,Path,Pass@1,Pass@K,Avg@K,AvgLen" > "$GLOBAL_CSV"
fi

echo "========================================================"
echo "Baseline Model: $BASELINE_NAME"
echo "Model Path:     $MODEL_PATH"
echo "Output Dir:     $OUTPUT_DIR/${BASELINE_NAME}"
echo "========================================================"

# ================= HumanEval =================
HUMAN_SAVE_DIR="$OUTPUT_DIR/${BASELINE_NAME}/human_eval"
HUMAN_MARKER_JSON="$HUMAN_SAVE_DIR/samples_results.json"
HUMAN_LEN_JSON="$HUMAN_SAVE_DIR/output_token_lengths.json"

mkdir -p "$HUMAN_SAVE_DIR"

echo "---- Dataset: human_eval ----"
if [ -f "$HUMAN_MARKER_JSON" ]; then
    echo "[Skip human_eval] Found existing result file: $HUMAN_MARKER_JSON"
else
    echo "running human_eval evaluation"
    python eval/code_eval/MBPP_Humaneval/eval/Coding/human_eval/evaluate_human_eval.py \
      --model "$MODEL_PATH" \
      --save_dir "$HUMAN_SAVE_DIR" \
      --num-samples-per-task 1 \
      --temperature "$TEMPERATURE"
fi

if [ -f "$HUMAN_MARKER_JSON" ] && [ -f "$HUMAN_LEN_JSON" ]; then
    PASS1=$(python3 -c "import json; d=json.load(open('$HUMAN_MARKER_JSON')); print(d['pass@pass@1'])")
    AVGLEN=$(python3 -c "import json; d=json.load(open('$HUMAN_LEN_JSON')); print(d['average_length'])")

    if ! grep -Fq "$BASELINE_NAME,humaneval,$MODEL_PATH," "$CSV_FILE"; then
        echo "$BASELINE_NAME,humaneval,$MODEL_PATH,$PASS1,$PASS1,$AVGLEN" >> "$CSV_FILE"
    else
        echo "[Skip LOCAL CSV] Already recorded: $BASELINE_NAME - humaneval"
    fi

    if ! grep -Fq "$EXP_TAG,$BASELINE_NAME,humaneval,$MODEL_PATH," "$GLOBAL_CSV"; then
        echo "$EXP_TAG,$BASELINE_NAME,humaneval,$MODEL_PATH,$PASS1,$PASS1,$PASS1,$AVGLEN" >> "$GLOBAL_CSV"
    else
        echo "[Skip GLOBAL CSV] Already recorded: $BASELINE_NAME - humaneval"
    fi
else
    echo "Warning: Missing HumanEval JSON for $BASELINE_NAME"
fi

# ================= MBPP =================
MBPP_SAVE_DIR="$OUTPUT_DIR/${BASELINE_NAME}/mbpp"
MBPP_MARKER_JSON="$MBPP_SAVE_DIR/mbpp_completion.json"
MBPP_TXT="$MBPP_SAVE_DIR/result.txt"

mkdir -p "$MBPP_SAVE_DIR"

echo "---- Dataset: mbpp ----"
if [ -f "$MBPP_MARKER_JSON" ]; then
    echo "[Skip mbpp] Found existing result file: $MBPP_MARKER_JSON"
else
    echo "running mbpp evaluation"
    python eval/code_eval/MBPP_Humaneval/eval/Coding/mbpp/evaluate_mbpp.py \
      --model "$MODEL_PATH" \
      --input_data eval/code_eval/MBPP_Humaneval/eval/Coding/mbpp/new_mbpp.json \
      --save_dir "$MBPP_SAVE_DIR"
fi

if [ -f "$MBPP_TXT" ]; then
    ACC=$(python3 -c "import ast; d=ast.literal_eval(open('$MBPP_TXT').readline()); print(d['accuracy'])")
    AVG=$(python3 -c "import ast; d=ast.literal_eval(open('$MBPP_TXT').readline()); print(d['average_tokens'])")

    if ! grep -Fq "$BASELINE_NAME,mbpp,$MODEL_PATH," "$CSV_FILE"; then
        echo "$BASELINE_NAME,mbpp,$MODEL_PATH,$ACC,$ACC,$AVG" >> "$CSV_FILE"
    else
        echo "[Skip LOCAL CSV] Already recorded: $BASELINE_NAME - mbpp"
    fi

    if ! grep -Fq "$EXP_TAG,$BASELINE_NAME,mbpp,$MODEL_PATH," "$GLOBAL_CSV"; then
        echo "$EXP_TAG,$BASELINE_NAME,mbpp,$MODEL_PATH,$ACC,$ACC,$ACC,$AVG" >> "$GLOBAL_CSV"
    else
        echo "[Skip GLOBAL CSV] Already recorded: $BASELINE_NAME - mbpp"
    fi
else
    echo "Warning: Missing MBPP result.txt for $BASELINE_NAME"
fi

echo "========================================================"
echo "Done."
echo "Local CSV:  $CSV_FILE"
echo "Global CSV: $GLOBAL_CSV"
echo "========================================================"