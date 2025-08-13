#!/bin/bash
ray stop --force
rm -rf./tmp/ray*
pkill -9 raylet
pkill -9 plasma_store
pkill -9 ray::
# rm -rf .wandb/ 
export WANDB_MODE=online
export WANDB_API_KEY=xx
# wandb online
export HYDRA_FULL_ERROR=1
ray start --head
set -x

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="xx" #your model path
fi

adv_estimator=grpo #算法

save_all_rollout_data=False

kl_coef=0.0
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=True
overlong_buffer_len=1024
overlong_penalty_factor=1
use_ppl=true
ppl_delta=0.01
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=5
train_prompt_bsz=512
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=8
train_prompt_mini_bsz=32
# Algorithm
## Train
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 8))
## Validation
val_top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

MODEL_NAME=xx
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/train.parquet \
    data.val_files='["/data/aime8.parquet","/data/math.parquet","/data/amc.parquet","/data/aime25_8.parquet"]' \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.save_all_data=${save_all_rollout_data} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.use_ppl_high=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.save_grad=${save_grad} \
    actor_rollout_ref.actor.save_grad_dir=${save_grad_dir} \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.val_kwargs.top_k="${val_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.use_ppl=${use_ppl} \
    actor_rollout_ref.actor.ppl_delta=${ppl_delta} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$sp_size \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    custom_reward_function.overlong_buffer.enable=${enable_overlong_buffer} \
    custom_reward_function.overlong_buffer.len=${overlong_buffer_len} \
    custom_reward_function.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.reward_manager="naive_log" \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl' \
    trainer.experiment_name=$MODEL_NAME \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.resume_mode=auto \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/capacity/userdata/still4/$MODEL_NAME \
    trainer.total_epochs=30 "${@:1}"

#,'wandb'
