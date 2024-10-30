# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

torchrun --nnodes=1 --nproc_per_node=1 eval.py \
--input_model_filename "checkpoint_path" \
--model_max_length 2048 \
--share_embedding True \
--layer_sharing False
