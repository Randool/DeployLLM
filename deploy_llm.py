"""
@File        :  deploy_llm.py
@Contact     :  randool@sjtu.edu.cn
@Author      :  Randool
@Create Time :  2023/2/27
@Version     :  1.0
"""
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from os.path import expanduser, isdir, join
from typing import List, Union

import psutil
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, \
    GenerationConfig, HfArgumentParser, PreTrainedModel, PreTrainedTokenizer

from utils import get_logger, get_smart_device_map, show_args

logger = get_logger(__name__, exp_dir=".")


@dataclass
class ModelArguments:
    model_name: str = field(default="bigscience/bloomz-3b", metadata={"help": "Model name in HuggingFace model hub"})
    torch_dtype: str = field(default="fp32", metadata={"choices": ["auto", "fp16", "fp32"]})
    device_map: str = field(default="auto", metadata={
        "choices": ["auto", "balanced", "balanced_low_0", "sequential", "smart"],
        "help": "https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map"
    })
    gpu_max_memory: str = field(default="6GiB")
    cpu_max_memory: str = field(default="8GiB")
    offload_folder: str = field(default="/tmp/offload")

    @property
    def checkpoint(self) -> str:
        if isdir(self.model_name):
            return self.model_name
        HOME = expanduser("~")
        return join(HOME, "Dataset/transformers", self.model_name)

    def get_torch_dtype(self):
        if self.torch_dtype == "auto":
            return self.torch_dtype
        elif self.torch_dtype == "fp16":
            return torch.float16
        return torch.float32

    def get_device_map(self, MODEL_CLASS) -> Union[str, dict]:
        """ Smarter device_map """
        if self.device_map != "smart":
            return self.device_map

        max_memory = {0: self.gpu_max_memory, "cpu": self.cpu_max_memory}
        device_map = get_smart_device_map(self.checkpoint, model_class=MODEL_CLASS, max_memory=max_memory)

        return device_map


@dataclass
class GenerationArguments:
    """ Refer to `GenerationConfig` """
    max_new_tokens: int = field(default=128)
    early_stopping: bool = field(default=True)
    do_sample: bool = field(default=True)
    temperature: float = field(default=1.0)


@dataclass
class DeployArguments:
    host: str = field(default="0.0.0.0")
    port: int = field(default=31004)
    batch_size: int = field(default=4)
    timeout: int = field(default=0.200, metadata={"help": "Timeout second"})


def parse_args():
    # Parse Arguments
    parser = HfArgumentParser([ModelArguments, GenerationArguments, DeployArguments])
    model_args, gen_args, deploy_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    gen_args: GenerationArguments
    deploy_args: DeployArguments
    show_args(model_args, logger)
    show_args(gen_args, logger)
    show_args(deploy_args, logger)
    return model_args, gen_args, deploy_args


def load_tokenizer_and_model(model_args: ModelArguments):
    gpu_before = torch.cuda.memory_allocated()
    cpu_before = psutil.virtual_memory().used

    TOKENIZER_CLASS, MODEL_CLASS = AutoTokenizer, AutoModelForCausalLM
    if "gpt" in model_args.model_name.lower():
        TOKENIZER_CLASS, MODEL_CLASS = GPT2Tokenizer, GPT2LMHeadModel

    tokenizer: PreTrainedTokenizer = TOKENIZER_CLASS.from_pretrained(
        model_args.checkpoint, padding_side="left", truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    model: PreTrainedModel = MODEL_CLASS.from_pretrained(
        model_args.checkpoint,
        device_map=model_args.get_device_map(MODEL_CLASS),
        torch_dtype=model_args.get_torch_dtype(),
        offload_folder=model_args.offload_folder,
    ).eval()

    logger.info(f"device_map: {model.hf_device_map}")
    logger.info(f"GPU usage: {(torch.cuda.memory_allocated() - gpu_before) / 1024 ** 3:.2f} GB")
    logger.info(f"CPU usage: {(psutil.virtual_memory().used - cpu_before) / 1024 ** 3:.2f} GB")

    return tokenizer, model


@dataclass
class PromptItem:
    timestamp: float
    prompt: str


@dataclass
class GeneratedItem:
    timestamp: float
    finish_time: float
    prompt: str
    generated_text: str


class ModelInferenceQueue:
    def __init__(self, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                 deploy_args: DeployArguments, gen_args: GenerationArguments):
        self.tokenizer = tokenizer
        self.model = model

        self.deploy_args = deploy_args
        self.gen_args = gen_args
        self.gen_config = GenerationConfig(
            **vars(self.gen_args),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Message Queue & Result Pool
        self._queue = queue.Queue()
        self._result_pool = {}

        self._last_infer_time = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        self.worker_thread = threading.Thread(target=self._run, daemon=True)

    def enqueue(self, item: PromptItem):
        """ When the new request is coming, `notify` will notify the working thread """
        with self._lock:
            self._queue.put(item)
            self._last_infer_time = time.monotonic()
            self._condition.notify()

    def dequeue(self, timestamp: float) -> GeneratedItem:
        # TODO: Block-able Dict
        while timestamp not in self._result_pool:
            time.sleep(0.05)
        return self._result_pool.pop(timestamp)

    def _do_inference(self, batch: List[PromptItem]):
        batch_text = [item.prompt for item in batch]
        input_ids = self.tokenizer(batch_text, return_tensors="pt", padding=True)["input_ids"].to(0)

        try:
            output = self.model.generate(input_ids, generation_config=self.gen_config)
            batch_generated_text = self.tokenizer.batch_decode(output)
        except Exception as e:
            logger.error(f"[Inference Error]: {e}")
            for item in batch:
                gen_item = GeneratedItem(item.timestamp, time.time(), item.prompt, f"[Inference Error]: {e}")
                self._result_pool[item.timestamp] = gen_item
        else:
            for item, gen_text in zip(batch, batch_generated_text):
                gen_item = GeneratedItem(item.timestamp, time.time(), item.prompt, gen_text[len(item.prompt):])
                self._result_pool[item.timestamp] = gen_item

    def start(self):
        self.worker_thread.start()

    def _run(self):
        while True:
            with self._lock:
                # Waiting
                while self._queue.qsize() < self.deploy_args.batch_size and \
                        time.monotonic() - self._last_infer_time < self.deploy_args.timeout:
                    self._condition.wait(timeout=self.deploy_args.timeout / 10)

                # Batch size branch
                if self._queue.qsize() >= self.deploy_args.batch_size:
                    batch_list = []
                    while not self._queue.empty() and len(batch_list) < self.deploy_args.batch_size:
                        batch_list.append(self._queue.get())
                    self._last_infer_time = time.monotonic()
                    self._do_inference(batch_list)

                # Timeout branch
                elif time.monotonic() - self._last_infer_time >= self.deploy_args.timeout:
                    if not self._queue.empty():
                        batch_list = []
                        while not self._queue.empty():
                            batch_list.append(self._queue.get())
                        self._last_infer_time = time.monotonic()
                        self._do_inference(batch_list)


def main():
    model_args, gen_args, deploy_args = parse_args()

    # Prepare Language Model
    tokenizer, model = load_tokenizer_and_model(model_args)
    inference_queue = ModelInferenceQueue(tokenizer, model, deploy_args, gen_args)

    # Set up Flask
    app = Flask(__name__)

    @app.route("/state", methods=["GET"])
    def _state():
        state = {
            "ModelArguments": vars(model_args),
            "GenerationArguments": vars(gen_args),
            "DeployArguments": vars(deploy_args),
        }
        return jsonify(state)

    @app.route("/model", methods=["GET"])
    def _model():
        return str(model)

    @app.route("/tokenizer", methods=["GET"])
    def _tokenizer():
        return str(tokenizer)

    @app.route('/inference', methods=["POST"])
    def _inference():
        try:
            raw_data = request.get_data().decode()
            prompt = json.loads(raw_data)["prompt"]
        except Exception as e:
            return f"[Data Decode Error]: {e}"

        logger.info(f"{request}\t{raw_data}")

        timestamp = time.time()
        inference_queue.enqueue(PromptItem(timestamp, prompt))
        response = vars(inference_queue.dequeue(timestamp))
        logger.info(response)
        return jsonify(response)

    # Start threading
    inference_queue.start()

    try:
        app.run(host=deploy_args.host, port=deploy_args.port, debug=False)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
        exit(0)


if __name__ == "__main__":
    main()
