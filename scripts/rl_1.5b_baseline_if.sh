#!/usr/bin/env bash
set -x

cd /opt/tiger/hqz_debug/cky/verl-process

pip install math_verify

export PROJECT_HOME="verl-process"
export LOG_DIR="/path/to/logs"
export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND="XFORMERS"
export DATASET_DIR="deepscaler/data"
ROOT_DIR=/mnt/hdfs/if_au/saves/cky

MODEL_PATH="/mnt/hdfs/if_au/models/DeepSeek-R1-Distill-Qwen-1.5B"

PROJECT_NAME="GRPO_Process"
EXPERIMENT_NAME="DS1.5B_16k_baseline"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=naive \
    data.train_files=deepscaler/data/train_deepscaler.parquet \
    data.val_files=deepscaler/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
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
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_training_steps=2000 "${@:1}" > $ROOT_DIR/checkpoints/${PROJECT_NAME}_${EXPERIMENT_NAME}.log