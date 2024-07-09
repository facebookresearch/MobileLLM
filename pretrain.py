# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from logging import Logger
import re
import sys
from typing import Dict, Iterator, List, Optional
import datetime

import torch
import transformers

from utils.modeling_llama import LlamaForCausalLM
from utils.pretrain_trainer import PretrainTrainer
from utils.process_args import process_args
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, default_data_collator

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


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()

def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def get_folder_paths(directory: str) -> List[str]:
    folder_paths = [
        os.path.join(directory, item)
        for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item))
    ]
    return folder_paths

def get_iterable_dataloader(iterator, batch_size):
    def custom_collate_fn(batch):
        return dict(input_ids=torch.stack(batch), labels=torch.stack(batch))
    class IteratorDataset(Dataset):
        def __init__(self, iterator):
            self.iterator = iterator
        def __len__(self):
        # Return an arbitrarily large number
            return sys.maxsize
        def __getitem__(self, index):
            try:
                ids = next(self.iterator)
                return torch.tensor(ids)
            except StopIteration:
                raise IndexError
    # Create dataset
    dataset = IteratorDataset(iterator)
    # Create DataLoader with custom collate function
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    return dataloader

class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ) -> None:
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.f = open(fpath, "r", encoding="utf-8", errors="ignore")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen(infinite))
        self.iter_id = 0

    def __iter__(self) -> "JSONLIterator":
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        while True:
            log.info(f"Starting iteration {self.iter_id} over {self.fpath} ...")
            self.iter_id += 1
            while True:
                try:
                    line, self.line_num = self.f.readline(), self.line_num + 1
                    if not line:
                        break
                    if (self.line_num - 1) % self.world_size == self.world_rank:
                        try:
                            yield json.loads(line)['token_ids']
                        except json.JSONDecodeError as e:
                            print("Failed to parse JSON:", e)
                        except Exception as e:
                            print(f"Unexpected Jsonl error: {e}")
                        continue  # Skip to the next iteration
                except Exception as e:
                    print(f"Unexpected error while reading line: {e}")
                continue
            if not infinite:
                break
            self.f.seek(0)
            self.line_num = 0
        self.f.close()

def train() -> None:
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=datetime.timedelta(hours=8)
    )
    model_args, data_args, training_args = process_args()

    global_rank = get_global_rank()
    local_rank = get_local_rank()

    log.info(f"Global Rank: {global_rank}")
    log.info(f"Local Rank: {local_rank}")
    config = AutoConfig.from_pretrained(model_args.input_model_filename)
    config.share_embedding = model_args.share_embedding
    config.layer_sharing = model_args.layer_sharing
    model = LlamaForCausalLM(
        config=config,
    )
    log.info(
        "model size is "
        + str(sum(param.numel() for param in model.model.parameters()) / 1024 / 1024)
    )
    log.info("Start to load tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model_filename,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    log.info("Complete tokenizer loading...")

    # go to current node's data rank
    local_data_folder = os.path.join(data_args.train_data_local_path, str(global_rank//8+1))

    # Data load locally from shard total data, so world_size is 8 and rank is the current node's local rank
    log.info("world_rank for data loader is " + str(local_rank))
    log.info("world_size for data loader is " + str(8))
    assert os.path.isdir(local_data_folder), local_data_folder
    fname_match_re: str = r"\.jsonl$"

    # get the jsonl file name. Currently only support 1 file/folder per node
    fnames = [x for x in os.listdir(local_data_folder) if re.search(fname_match_re, x)][0]
    data_iter = JSONLIterator(
        fpath=os.path.join(local_data_folder,fnames),
        world_rank=local_rank,
        world_size=8,
        infinite=True,
    )
    trainer = PretrainTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=get_iterable_dataloader(data_iter, training_args.per_device_train_batch_size) if training_args.do_train else None,
        eval_dataset=None,
        data_collator=default_data_collator,
    )
    torch.distributed.barrier(device_ids=[local_rank])

    if training_args.do_train:
        _ = trainer.train()
        trainer.save_state()

    torch.distributed.barrier(device_ids=[local_rank])


if __name__ == "__main__":
    train()
