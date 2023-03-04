"""
@File        :  utils.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2023/2/27
@Version     :  1.0
"""
import json
import logging
import os
import sys
from argparse import Namespace
from logging import Logger
from os.path import join
from typing import List

import torch
import torch.distributed as dist


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() in [0, -1]


#### Logging ####

def show_args(args, logger: Logger):
    logger.info("=" * 45)
    for _arg in vars(args):
        logger.info(f"{_arg:20s}\t{getattr(args, _arg)}")
    logger.info("=" * 45)


def set_logger(exp_dir: str = None):
    if exp_dir is None:
        logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M:%S",
                            format="[%(asctime)s] %(levelname)s - %(message)s")
    else:
        os.makedirs(exp_dir, exist_ok=True)
        logging.basicConfig(filename=join(exp_dir, "log.txt"), level=logging.INFO,
                            datefmt='%m-%d %H:%M:%S', format='[%(asctime)s] %(levelname)s - %(message)s')


def get_logger(logger_name: str, exp_dir: str = None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if is_main_process() else logging.WARN)

    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Repeated calls to `addHandler` result in repeated log output
    if len(logger.handlers) == 0:
        formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if exp_dir is not None:
            os.makedirs(exp_dir, exist_ok=True)
            handler = logging.FileHandler(join(exp_dir, "log.txt"), encoding="utf8")
            handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log(msg: str, logger: Logger = None, level="info"):
    if logger is None:
        print(msg, file=sys.stderr)
    else:
        getattr(logger, level)(msg)


#### Training States ####
class LimitedQueue:
    def __init__(self, max_len: int = -1, items: List[float] = None):
        self.max_len = max_len
        self.items: List[float] = [] if items is None else items

    def append(self, item: float):
        while self.max_len != -1 and len(self) >= self.max_len:
            self.pop(0)
        self.items.append(item)

    def pop(self, index=-1):
        return self.items.pop(index)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        yield from self.items

    def __str__(self):
        return str(self.items)

    def __repr__(self):
        return str(self.items)

    @property
    def is_full(self):
        return len(self) == self.max_len

    @property
    def avg_value(self):
        if len(self) == 0:
            return 0
        return sum(self.items) / len(self)


class EarlyStopping:
    def __init__(self, patience: int = 1, decrease=True):
        self.patience = patience
        self.decrease = decrease
        self.counter = 0
        self.best_metric = None

    @property
    def is_stop(self):
        return self.counter >= self.patience

    def is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.decrease:
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def record(self, metric: float):
        if self.best_metric is None:
            self.counter = 0
            self.best_metric = metric
        else:
            if self.is_better(metric):
                self.counter = 0
                self.best_metric = metric
            else:
                self.counter += 1


class RuntimeMetric:
    def __init__(self, best_metrics=None, window_size=10, patience=1, decrease=True):
        self.avg_loss = LimitedQueue(window_size)
        self.early_stop = EarlyStopping(patience, decrease)
        self.early_stop.best_metric = best_metrics

    def metric_is_better_than_ever(self, metric: float) -> bool:
        """ Compare and log the metric """
        is_better = self.early_stop.is_better(metric)
        self.early_stop.record(metric)
        return is_better

    @property
    def do_early_stop(self) -> bool:
        return self.early_stop.is_stop

    def dump(self, filename: str):
        tmp = {
            "best_metrics": self.early_stop.best_metric,
            "window_size": self.avg_loss.max_len,
            "patience": self.early_stop.patience,
            "decrease": self.early_stop.decrease,
            "avg_loss.items": self.avg_loss.items,
            "early_stop.counter": self.early_stop.counter
        }
        with open(filename, "w", encoding="utf8") as f:
            json.dump(tmp, f, indent=1)

    @classmethod
    def load(cls, filename: str):
        with open(filename, encoding="utf8") as f:
            tmp = json.load(f)
            avg_loss_items = tmp.pop("avg_loss.items")
            early_stop_counter = tmp.pop("early_stop.counter")
        ins = cls(**tmp)
        ins.avg_loss.items = avg_loss_items
        ins.early_stop.counter = early_stop_counter
        return ins


#### DP / DDP ####

def use_DP(args: Namespace):
    return ("cuda" in str(args.device)) and (args.parallel == "DP") and (torch.cuda.device_count() > 1)


def use_DDP(args: Namespace):
    return ("cuda" in str(args.device)) and (args.parallel == "DDP") and (torch.cuda.device_count() > 1)


def get_free_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(('', 0))
    ip, port = sock.getsockname()
    sock.close()

    return port


def setup_dist_environment(rank, world_size, master_port: int, store=None, logger: Logger = None):
    if logger:
        log(f"Setup, Rank: {rank}, world_size: {world_size}, master_port: {master_port}", logger)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)

    # initialize the process group
    if os.name == "nt":
        dist.init_process_group("gloo", rank=rank, world_size=world_size, store=store)
    else:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, store=store)


def cleanup_dist_environment():
    dist.destroy_process_group()


#### Device map ####

def get_smart_device_map(checkpoint_path: str, model_class=None, max_memory: dict = None):
    from transformers import AutoConfig, AutoModel
    from accelerate import init_empty_weights, infer_auto_device_map

    if model_class is None:
        model_class = AutoModel

    with init_empty_weights():
        model_config = AutoConfig.from_pretrained(checkpoint_path)
        empty_model = model_class.from_config(model_config)
        device_map = infer_auto_device_map(empty_model, max_memory=max_memory)

    return device_map
