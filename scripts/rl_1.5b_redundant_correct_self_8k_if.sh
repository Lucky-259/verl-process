#!/usr/bin/env bash
set -x

cd /opt/tiger/hqz_debug/cky/verl-process

pip install -e .
pip install math_verify
pip install tensorboard

export NCCL_DEBUG=WARN
export PROJECT_HOME="verl-process"
export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND="XFORMERS"
export DATASET_DIR="deepscaler/data"
ROOT_DIR=/mnt/hdfs/if_au/saves/cky
MODEL_PATH="/mnt/hdfs/if_au/models/DeepSeek-R1-Distill-Qwen-1.5B"

PROJECT_NAME='Redundancy_new'
EXPERIMENT_NAME='DS1.5B_8k_redundancy_correct_self_2_1e-3'
RUN_DIR="$ROOT_DIR/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
LOG_FILE="/opt/tiger/hqz_debug/cky/verl-process/${PROJECT_NAME}_${EXPERIMENT_NAME}_new.log"
mkdir -p "$RUN_DIR"
export TENSORBOARD_DIR=$RUN_DIR

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=redundancy \
    +reward_model.reward_kwargs.alpha=2 \
    +reward_model.reward_kwargs.beta=1e-3 \
    +reward_model.reward_kwargs.extraction=self \
    +reward_model.reward_kwargs.way=correct \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.9 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=200 \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.total_training_steps=1000 "${@:1}" > "$LOG_FILE" 2>&1