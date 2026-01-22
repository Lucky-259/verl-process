#!/bin/bash
# set -e

export VLLM_USE_V1=0

PROJECT_NAME=${1}
EXPERIMENT_NAME=${2}
REPEAT=${3:-16}
CONCURRENCY=${4:-150}

CHECKPOINT_ROOT="/mnt/hdfs/if_au/saves/cky/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
OUTPUT_ROOT="/mnt/hdfs/if_au/saves/cky/eval_results/${EXPERIMENT_NAME}"

DATASETS=("aime" "aime25" "amc" "math" "minerva" "olympiad_bench")
# CHECKPOINT_ROOT="/mnt/luoyingfeng/changkaiyan/ShorterBetter/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
# OUTPUT_ROOT="/mnt/luoyingfeng/changkaiyan/verl-process/eval/eval_results/${EXPERIMENT_NAME}"
# DATASETS=("aime" "math")

ALL_PORTS=(8010 8011 8012 8013 8014 8015 8016 8017)
DEVICES=(0 1 2 3 4 5 6 7)

NUM_DEVICES=${#DEVICES[@]}
PORTS=("${ALL_PORTS[@]:0:$NUM_DEVICES}")
PORTS_STR=$(IFS=,; echo "${PORTS[*]}")

echo "Using $NUM_DEVICES GPUs with ports: ${PORTS[*]}"

mkdir -p "$OUTPUT_ROOT"

# ============ CSV Headers ============
SUMMARY_FILE="$OUTPUT_ROOT/all_checkpoints_summary.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
  echo "Step,Dataset,Path,Pass@1,Pass@K,Avg@K,AvgLen" > "$SUMMARY_FILE"
fi

GLOBAL_OUTPUT_ROOT="/mnt/hdfs/if_au/saves/cky/eval_results"
# GLOBAL_OUTPUT_ROOT="/mnt/luoyingfeng/changkaiyan/verl-process/eval/eval_results"

# 这里换一个新文件名，避免旧 global csv 少一列、且历史结果无 avg@k 导致列数不一致
GLOBAL_SUMMARY_FILE="$GLOBAL_OUTPUT_ROOT/all_jobs_summary_8k_avgk.csv"

mkdir -p "$GLOBAL_OUTPUT_ROOT"
if [ ! -f "$GLOBAL_SUMMARY_FILE" ]; then
  echo "Project,Experiment,Step,Dataset,Path,Pass@1,Pass@K,Avg@K,AvgLen" > "$GLOBAL_SUMMARY_FILE"
fi

# ============ 工具函数 ============
append_csv_if_summary_exists () {
    local STEP_NAME="$1"
    local DATA_NAME="$2"
    local MODEL_PATH="$3"
    local OUT_DIR="$4"
    local SUMMARY_JSON="$OUT_DIR/results_summary.json"

    if [ ! -f "$SUMMARY_JSON" ]; then
        echo "[No summary] $STEP_NAME - $DATA_NAME"
        return 0
    fi

    # 读取指标
    # avg@k 的 key 是 "avg@${REPEAT}"（k 就是 repeat）
    read P1 PK AVGK AVG_LEN <<< $(python -c "import json; d=json.load(open('$SUMMARY_JSON')); k=str($REPEAT); print(f\"{d.get('pass@1','N/A')} {d.get('pass@'+k,'N/A')} {d.get('avg@'+k,'N/A')} {d.get('average_token_len','N/A')}\")" 2>/dev/null || echo "N/A N/A N/A N/A")

    if grep -Fq "$STEP_NAME,$DATA_NAME,$MODEL_PATH," "$SUMMARY_FILE"; then
        echo "[Skip EXP CSV] Already recorded: $STEP_NAME - $DATA_NAME"
    else
        echo "$STEP_NAME,$DATA_NAME,$MODEL_PATH,$P1,$PK,$AVGK,$AVG_LEN" >> "$SUMMARY_FILE"
        echo "[EXP CSV] Recorded: $STEP_NAME - $DATA_NAME | P1=$P1 PK=$PK AVGK=$AVGK Len=$AVG_LEN"
    fi

    if grep -Fq "$PROJECT_NAME,$EXPERIMENT_NAME,$STEP_NAME,$DATA_NAME,$MODEL_PATH," "$GLOBAL_SUMMARY_FILE"; then
        echo "[Skip GLOBAL CSV] Already recorded: $PROJECT_NAME/$EXPERIMENT_NAME $STEP_NAME - $DATA_NAME"
    else
        echo "$PROJECT_NAME,$EXPERIMENT_NAME,$STEP_NAME,$DATA_NAME,$MODEL_PATH,$P1,$PK,$AVGK,$AVG_LEN" >> "$GLOBAL_SUMMARY_FILE"
        echo "[GLOBAL CSV] Recorded: $PROJECT_NAME/$EXPERIMENT_NAME $STEP_NAME - $DATA_NAME"
    fi
}

# ================= 查找 Checkpoints =================
echo "Looking for checkpoints in $CHECKPOINT_ROOT..."
if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "Warning: Directory not found: $CHECKPOINT_ROOT"
    echo "Skipping evaluation for this experiment."
    exit 0  # 正常退出，不报错，方便批量脚本继续运行
fi

CHECKPOINT_DIRS=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -V)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "No global_step directories found in $CHECKPOINT_ROOT"
    exit 1
fi

# ================= 开始 Checkpoint 循环 =================
for STEP_DIR in $CHECKPOINT_DIRS; do
    STEP_NAME=$(basename "$STEP_DIR")
    
    # 定义 FSDP 源目录 (Actor) 和 目标 HF 目录
    FSDP_DIR="$STEP_DIR/actor"
    MODEL_PATH="$STEP_DIR/actor/huggingface"

    echo "========================================================"
    echo "Checkpoint: $STEP_NAME"
    echo "FSDP Source: $FSDP_DIR"
    echo "Target Model Path: $MODEL_PATH"
    echo "========================================================"

    # ------------------------------------------------------------------
    # 【新增逻辑】检查并合并 safetensors
    # ------------------------------------------------------------------
    
    # 1. 检查 FSDP 源目录是否存在
    if [ ! -d "$FSDP_DIR" ]; then
        echo "Error: FSDP source directory not found: $FSDP_DIR. Skipping..."
        continue
    fi

    # 2. 检查目标目录是否已有 safetensors 文件
    # find 命令查找 .safetensors 文件，如果数量为0则说明没合并过或者合并失败
    if [ -d "$MODEL_PATH" ]; then
        SAFE_COUNT=$(find "$MODEL_PATH" -maxdepth 1 -name "*.safetensors" | wc -l)
    else
        SAFE_COUNT=0
    fi

    if [ "$SAFE_COUNT" -gt 0 ]; then
        echo "[Check] Safetensors found in $MODEL_PATH. Skipping merge."
    else
        echo "[Merge] No safetensors found. Starting merge process..."
        echo "Running scripts/legacy_model_merger.py..."
        
        # 运行合并脚本
        python scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir "$FSDP_DIR" \
            --target_dir "$MODEL_PATH"

        # 检查合并脚本的退出状态
        if [ $? -ne 0 ]; then
            echo "Error: Model merge failed for $STEP_NAME. Skipping evaluation."
            continue
        fi
        
        # 二次检查是否真的生成了
        SAFE_COUNT_AFTER=$(find "$MODEL_PATH" -maxdepth 1 -name "*.safetensors" | wc -l)
        if [ "$SAFE_COUNT_AFTER" -eq 0 ]; then
             echo "Error: Merge script ran but no .safetensors found in $MODEL_PATH. Skipping."
             continue
        fi
        echo "[Merge] Successfully merged to $MODEL_PATH"
    fi
    # ------------------------------------------------------------------


    # 1) 先判断该 checkpoint 是否有缺失的 details（决定要不要启动 vLLM）
    NEED_RUN_DATASETS=()
    for DATA_NAME in "${DATASETS[@]}"; do
        CURRENT_OUTPUT_DIR="$OUTPUT_ROOT/$STEP_NAME/$DATA_NAME"
        DETAILS_FILE="$CURRENT_OUTPUT_DIR/results_details.json"
        if [ ! -f "$DETAILS_FILE" ]; then
            NEED_RUN_DATASETS+=("$DATA_NAME")
        fi
    done

    # 2) 如果需要跑，才启动 vLLM
    VLLM_STARTED=0
    if [ ${#NEED_RUN_DATASETS[@]} -gt 0 ]; then
        echo "Need to run datasets (missing details): ${NEED_RUN_DATASETS[*]}"
        echo "Starting vLLM servers..."

        for i in "${!DEVICES[@]}"; do
            PORT="${PORTS[$i]}"
            DEVICE="${DEVICES[$i]}"
            LOG_FILE="${OUTPUT_ROOT}/vllm_server_${STEP_NAME}_gpu${DEVICE}.log"

            CUDA_VISIBLE_DEVICES=$DEVICE vllm serve "$MODEL_PATH" \
            --max_model_len 8192 \
            --port "$PORT" > "$LOG_FILE" 2>&1 &
        done

        VLLM_STARTED=1
        echo "Waiting 60s for vLLM to start..."
        sleep 60
    else
        echo "[Skip vLLM] All datasets already have details for $STEP_NAME."
    fi

    for DATA_NAME in "${DATASETS[@]}"; do
        DATA_FILE="deepscaler/data/test/${DATA_NAME}.json"
        CURRENT_OUTPUT_DIR="$OUTPUT_ROOT/$STEP_NAME/$DATA_NAME"
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

        append_csv_if_summary_exists "$STEP_NAME" "$DATA_NAME" "$MODEL_PATH" "$CURRENT_OUTPUT_DIR"
    done

    if [ "$VLLM_STARTED" -eq 1 ]; then
        echo "Cleaning up vLLM..."
        killall vllm || true
        sleep 5
        killall vllm || true
        sleep 5
        killall vllm || true
        sleep 5
    fi

done

echo "========================================================"
echo "Done."
echo "Experiment CSV: $SUMMARY_FILE"
echo "Global CSV:     $GLOBAL_SUMMARY_FILE"
echo "========================================================"