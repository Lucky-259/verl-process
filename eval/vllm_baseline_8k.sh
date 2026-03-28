#!/bin/bash
# set -e

export VLLM_USE_V1=0

# ================= 配置区域 =================
# 参数1: 模型文件夹名称 (例如: DeepSeek-R1-Distill-Qwen-1.5B)
BASELINE_MODEL=${1}
REPEAT=${2:-1}
CONCURRENCY=${3:-150}

# 1. 检查是否传入了模型名称
if [ -z "$BASELINE_MODEL" ]; then
    echo "Error: Please provide the model name as the first argument."
    echo "Usage: bash script.sh <Model_Folder_Name> [Repeat] [Concurrency]"
    exit 1
fi

# 模型存放的根目录
# MODEL_ROOT="/mnt/luoyingfeng/model_card"
MODEL_ROOT="/mnt/hdfs/if_au/models"

# 2. 构造具体的模型路径并检查是否存在
TARGET_MODEL_PATH="$MODEL_ROOT/$BASELINE_MODEL"

if [ ! -d "$TARGET_MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $TARGET_MODEL_PATH"
    exit 1
fi

# 3. 配置输出根目录
# OUTPUT_ROOT="/mnt/luoyingfeng/changkaiyan/verl-process/eval/baseline_res"
OUTPUT_ROOT="/mnt/hdfs/if_au/saves/cky/eval_results"

DATASETS=("aime" "aime25" "amc" "math" "minerva" "olympiad_bench" "bbh" "mathqa" "mmlu")

ALL_PORTS=(8010 8011 8012 8013 8014 8015 8016 8017)
DEVICES=(0 1 2 3 4 5 6 7)

NUM_DEVICES=${#DEVICES[@]}
PORTS=("${ALL_PORTS[@]:0:$NUM_DEVICES}")
PORTS_STR=$(IFS=,; echo "${PORTS[*]}")

echo "Using $NUM_DEVICES GPUs with ports: ${PORTS[*]}"
echo "Target Model: $BASELINE_MODEL"
echo "Model Path:   $TARGET_MODEL_PATH"
echo "Output Root:  $OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"

# ============ 全局 CSV ============
# 这个文件依然放在 baseline_res 根目录下，用于汇总所有任务
# GLOBAL_OUTPUT_ROOT="/mnt/luoyingfeng/changkaiyan/verl-process/eval/baseline_res"
# GLOBAL_SUMMARY_FILE="$GLOBAL_OUTPUT_ROOT/all_jobs_summary_8k.csv"
GLOBAL_OUTPUT_ROOT="/mnt/hdfs/if_au/saves/cky/eval_results"
GLOBAL_SUMMARY_FILE="$GLOBAL_OUTPUT_ROOT/all_baselines_summary_8k.csv"

mkdir -p "$GLOBAL_OUTPUT_ROOT"
if [ ! -f "$GLOBAL_SUMMARY_FILE" ]; then
  echo "Experiment,ModelName,Dataset,Path,Pass@1,Pass@K,AvgLen" > "$GLOBAL_SUMMARY_FILE"
fi

# ============ 工具函数 ============
append_csv_if_summary_exists () {
    local MODEL_NAME="$1"
    local DATA_NAME="$2"
    local MODEL_PATH="$3"
    local OUT_DIR="$4"
    local SUMMARY_JSON="$OUT_DIR/results_summary.json"

    if [ ! -f "$SUMMARY_JSON" ]; then
        echo "[No summary] $MODEL_NAME - $DATA_NAME"
        return 0
    fi

    # 读取指标
    read P1 PK AVG_LEN <<< $(python -c "import json; d=json.load(open('$SUMMARY_JSON')); print(f\"{d.get('pass@1','N/A')} {d.get('pass@'+str($REPEAT),'N/A')} {d.get('average_token_len','N/A')}\")" 2>/dev/null || echo "N/A N/A N/A")

    # ============ 【修改点2】动态定义模型专属 CSV ============
    # 路径变为: baseline_res/{ModelName}/all_models_summary.csv
    local MODEL_SPECIFIC_CSV="$OUTPUT_ROOT/$MODEL_NAME/all_models_summary.csv"
    
    # 如果该模型的 CSV 不存在，先创建并写表头
    if [ ! -f "$MODEL_SPECIFIC_CSV" ]; then
        echo "Creating model summary CSV at: $MODEL_SPECIFIC_CSV"
        echo "ModelName,Dataset,Path,Pass@1,Pass@K,AvgLen" > "$MODEL_SPECIFIC_CSV"
    fi

    # ---- 模型专属 CSV 写入 ----
    if grep -Fq "$MODEL_NAME,$DATA_NAME,$MODEL_PATH," "$MODEL_SPECIFIC_CSV"; then
        echo "[Skip LOCAL CSV] Already recorded: $MODEL_NAME - $DATA_NAME"
    else
        echo "$MODEL_NAME,$DATA_NAME,$MODEL_PATH,$P1,$PK,$AVG_LEN" >> "$MODEL_SPECIFIC_CSV"
        echo "[LOCAL CSV] Recorded: $MODEL_NAME - $DATA_NAME | P1=$P1 PK=$PK Len=$AVG_LEN"
    fi

    # ---- 全局 CSV 写入 (保持不变) ----
    local EXP_TAG="baseline" 
    if grep -Fq "$EXP_TAG,$MODEL_NAME,$DATA_NAME,$MODEL_PATH," "$GLOBAL_SUMMARY_FILE"; then
        echo "[Skip GLOBAL CSV] Already recorded: $EXP_TAG $MODEL_NAME - $DATA_NAME"
    else
        echo "$EXP_TAG,$MODEL_NAME,$DATA_NAME,$MODEL_PATH,$P1,$PK,$AVG_LEN" >> "$GLOBAL_SUMMARY_FILE"
        echo "[GLOBAL CSV] Recorded: $EXP_TAG $MODEL_NAME - $DATA_NAME"
    fi
}

# ================= 设定待测列表 =================
MODEL_DIRS=("$TARGET_MODEL_PATH")

# ================= 开始 Model 循环 =================
for DIR in "${MODEL_DIRS[@]}"; do
    MODEL_NAME=$(basename "$DIR")
    MODEL_PATH="$DIR"

    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "Warning: No config.json found in $MODEL_PATH."
    fi

    echo "========================================================"
    echo "Model Name: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "========================================================"

    # 1) 判断是否需要跑
    NEED_RUN_DATASETS=()
    for DATA_NAME in "${DATASETS[@]}"; do
        CURRENT_OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_NAME/$DATA_NAME"
        DETAILS_FILE="$CURRENT_OUTPUT_DIR/results_details.json"
        if [ ! -f "$DETAILS_FILE" ]; then
            NEED_RUN_DATASETS+=("$DATA_NAME")
        fi
    done

    # 2) 启动 vLLM
    VLLM_STARTED=0
    if [ ${#NEED_RUN_DATASETS[@]} -gt 0 ]; then
        echo "Need to run datasets: ${NEED_RUN_DATASETS[*]}"
        echo "Starting vLLM servers..."
        # (建议：这里最好替换为你之前的 PID 追踪版本，防止 killall 误杀)
        for i in "${!DEVICES[@]}"; do
            PORT="${PORTS[$i]}"
            DEVICE="${DEVICES[$i]}"
            LOG_FILE="${OUTPUT_ROOT}/vllm_server_${MODEL_NAME}_gpu${DEVICE}.log"
            CUDA_VISIBLE_DEVICES=$DEVICE vllm serve "$MODEL_PATH" \
            --max_model_len 8192 \
            --port "$PORT" > "$LOG_FILE" 2>&1 &
        done
        VLLM_STARTED=1
        echo "Waiting 60s for vLLM to start..."
        sleep 60
    else
        echo "[Skip vLLM] All datasets finished for $MODEL_NAME."
    fi

    # 3) 执行评测
    for DATA_NAME in "${DATASETS[@]}"; do
        DATA_FILE="deepscaler/data/test/${DATA_NAME}.json"
        CURRENT_OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_NAME/$DATA_NAME"
        mkdir -p "$CURRENT_OUTPUT_DIR"

        DETAILS_FILE="$CURRENT_OUTPUT_DIR/results_details.json"

        echo "---- Dataset: $DATA_NAME ----"
        if [ -f "$DETAILS_FILE" ]; then
            echo "[Skip infer] details exists: $DETAILS_FILE"
        else
            if [ -f "$DATA_FILE" ]; then
                python eval/vllm_reason.py \
                    --model "$MODEL_PATH" \
                    --file "$DATA_FILE" \
                    --ports "$PORTS_STR" \
                    --repeat "$REPEAT" \
                    --concurrency "$CONCURRENCY" \
                    --output_dir "$CURRENT_OUTPUT_DIR"
            else
                echo "Error: Data file $DATA_FILE not found!"
            fi
        fi

        # 调用函数写入CSV，此时函数内部会自动处理路径
        append_csv_if_summary_exists "$MODEL_NAME" "$DATA_NAME" "$MODEL_PATH" "$CURRENT_OUTPUT_DIR"
    done

    # 4) 清理 vLLM
    if [ "$VLLM_STARTED" -eq 1 ]; then
        echo "Cleaning up vLLM..."
        killall vllm
        sleep 5
        killall vllm
        sleep 5
        killall vllm
        sleep 5
    fi

done

echo "========================================================"
echo "Done."
echo "Global CSV: $GLOBAL_SUMMARY_FILE"
# 这里的提示修改了，因为 LOCAL CSV 是动态路径
echo "Model CSVs are located in: $OUTPUT_ROOT/{ModelName}/all_models_summary.csv"
echo "========================================================"