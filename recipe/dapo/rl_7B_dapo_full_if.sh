#!/usr/bin/env bash
set -x

cd /opt/tiger/hqz_debug/cky/verl-process

pip install -e .
pip install math_verify
# pip install tensorboard

export NCCL_DEBUG=WARN
export PROJECT_HOME="verl-process"
# export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATASET_DIR="deepscaler/data"
ROOT_DIR=/mnt/hdfs/if_au/saves/cky
MODEL_PATH="/mnt/hdfs/if_au/models/DeepSeek-R1-Distill-Qwen-7B"
PROJECT_NAME='Redundancy_DAPO'
EXPERIMENT_NAME='DS7B_8k_redundancy_ground_full_1_2e-4_1_DAPO'
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

train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
penalty_factor=1.0

python3 -u -m recipe.dapo.main_dapo \
    reward_model.reward_manager=dapo_redundancy \
    +reward_model.reward_kwargs.alpha=1 \
    +reward_model.reward_kwargs.beta=2e-4 \
    +reward_model.reward_kwargs.extraction=ground \
    +reward_model.reward_kwargs.way=full \
    data.train_files="deepscaler/data/train_deepscaler_filtered_plus.parquet" \
    data.val_files="deepscaler/data/aime.parquet" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=10 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=4096 \
    reward_model.overlong_buffer.penalty_factor=${penalty_factor} \
    trainer.logger='["console"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=50 \
    trainer.save_freq=50 \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.total_training_steps=1000 \
    > "$LOG_FILE" 2>&1
