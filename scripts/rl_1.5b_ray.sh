#!/usr/bin/env bash
set -xeuo pipefail

export RAY_ADDRESS=[fdbd:dccd:cde6:1004:f6b:9698:8974:bbec]:9903

export SEC_TOKEN_STRING=`cat $SEC_TOKEN_PATH`

HYDRA_FULL_ERROR=1

# Ray
RUNTIME_ENV="/opt/tiger/hqz_debug/verl_search/verl/trainer/runtime_env.yaml"

MODEL_PATH=""

export PROJECT_HOME="verl-process"
export LOG_DIR="/path/to/logs"
export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATASET_DIR="deepscaler/data"
PROJECT_NAME='GRPO_Process'
EXPERIMENT_NAME='DS1.5B_16k_DS3.2_0.5'
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_process \
    reward_model.reward_manager=grpo_process \
    custom_reward_function.path=recipe/multi_prm/reward_function.py \
    data.train_files=deepscaler/data/train_deepscaler.parquet \
    data.val_files=deepscaler/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','file'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=2000 "${@:1}"