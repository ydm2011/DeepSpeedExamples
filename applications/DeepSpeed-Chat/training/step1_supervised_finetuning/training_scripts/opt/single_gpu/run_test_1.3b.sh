#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" = "" ]; then
    OUTPUT=/data/log
fi
if [ "$ZERO_STAGE" = "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

#deepspeed --num_gpus 1 main.py --model_name_or_path facebook/opt-1.3b \
#   --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage $ZERO_STAGE \
#   --enable_tensorboard \
#   --tensorboard_path $OUTPUT \
#   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log
deepspeed main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --offload  \
   --model_name_or_path facebook/opt-1.3b \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0. \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 16 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT >$OUTPUT/info.log
