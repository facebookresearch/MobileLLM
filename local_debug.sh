# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

torchrun --nnodes=1 --nproc_per_node=8 pretrain.py \
--input_model_filename "./configs/125M/" \
--train_data_local_path "basepath" \
--output_dir "output_path" \
--do_train True \
--do_eval False \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--ddp_find_unused_parameters False \
--logging_dir "logging_path" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--save_steps 1000 \
--eval_steps 1000 \
--logging_steps 10 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--report_to "tensorboard" \
--save_total_limit 1 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-8 \
--lr_scheduler_type "cosine" \
--gradient_checkpointing False \
--save_safetensors False \
--max_steps 10000 \
--warmup_step 1000 \
--share_embedding True
