# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from logging import Logger
from typing import Optional
import datetime
from tqdm import tqdm
from torch import nn
import torch
import transformers
from datasets import load_dataset
from utils.modeling_llama import LlamaForCausalLM
from utils.process_args import process_args
from torch import distributed as dist
from transformers import AutoConfig


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str]) -> logging.Logger:
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


log: Logger = get_logger("mobileLLM")


def train() -> None:
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
    )
    model_args, data_args, training_args = process_args()

    config = AutoConfig.from_pretrained(model_args.input_model_filename)
    config.share_embedding = model_args.share_embedding
    config.layer_sharing = model_args.layer_sharing
    log.info("Start to load model...")
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        config=config,
    )
    model.cuda()
    log.info(
        "model size is "
        + str(sum(param.numel() for param in model.model.parameters()) / 1024 / 1024)
    )
    log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    log.info("Complete tokenizer loading...")

    # The code for eval wiki2 ppl is from https://github.com/mit-han-lab/llm-awq/tree/main
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = training_args.model_max_length
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())


if __name__ == "__main__":
    train()
