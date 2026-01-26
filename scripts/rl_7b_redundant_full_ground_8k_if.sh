#!/usr/bin/env bash
set -x

cd /opt/tiger/hqz_debug/cky/verl-process

pip install -e .
pip install math_verify
# pip install tensorboard

export NCCL_DEBUG=WARN
export PROJECT_HOME="verl-process"
export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND="XFORMERS"
export DATASET_DIR="deepscaler/data"
ROOT_DIR=/mnt/hdfs/if_au/saves/cky
MODEL_PATH="/mnt/hdfs/if_au/models/DeepSeek-R1-Distill-Qwen-7B"

PROJECT_NAME='Redundancy'
EXPERIMENT_NAME='DS7B_8k_redundancy_ground_full_1_2e-4'
RUN_DIR="$ROOT_DIR/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LOG_FILE="/opt/tiger/hqz_debug/cky/verl-process/${PROJECT_NAME}_${EXPERIMENT_NAME}.log"
mkdir -p "$RUN_DIR"
# export TENSORBOARD_DIR=$RUN_DIR

# ===== Periodically upload local log to HDFS =====
HDFS_LOG_DIR="$ROOT_DIR/checkpoints/${PROJECT_NAME}"
SYNC_INTERVAL=3600   # 每 60 秒传一次；你可以改成 30/120

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

hdfs_sync_log_once() {
  # 确保 HDFS 目录存在
  hdfs dfs -mkdir -p "$HDFS_LOG_DIR" >/dev/null 2>&1 || true

  # 如果本地 log 还没生成，跳过
  [[ -f "$LOG_FILE" ]] || return 0

  # 覆盖上传（快照式），最简单可靠
  hdfs dfs -put -f "$LOG_FILE" "$HDFS_LOG_DIR/" >/dev/null 2>&1 || true
}

hdfs_sync_log_loop() {
  while true; do
    hdfs_sync_log_once
    sleep "$SYNC_INTERVAL"
  done
}

# 后台启动同步
hdfs_sync_log_loop &
SYNC_PID=$!

# 脚本退出/中断时最后同步一次
cleanup() {
  set +e
  kill "$SYNC_PID" >/dev/null 2>&1 || true
  hdfs_sync_log_once
}
trap cleanup EXIT INT TERM
# ===== Periodically upload local log to HDFS =====

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=redundancy \
    +reward_model.reward_kwargs.alpha=1 \
    +reward_model.reward_kwargs.beta=2e-4 \
    +reward_model.reward_kwargs.extraction=ground \
    +reward_model.reward_kwargs.way=full \
    data.train_files=deepscaler/data/train_deepscaler_filtered_plus.parquet \
    data.val_files=deepscaler/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=100 \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.total_training_steps=1000 "${@:1}" > "$LOG_FILE" 2>&1