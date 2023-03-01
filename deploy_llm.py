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
from typing import List

import psutil
import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, HfArgumentParser, PreTrainedModel, \
    PreTrainedTokenizer

from utils import show_args, get_logger


logger = get_logger(__name__, exp_dir=None)


@dataclass
class ModelArguments:
    model_name: str = field(default="bigscience/bloomz-3b", metadata={"help": "Model name in HuggingFace model hub"})
    torch_dtype: str = field(default="auto", metadata={"choices": ["auto", "fp16", "fp32"]})
    device_map: str = field(default="auto", metadata={
        "choices": ["auto", "balanced", "balanced_low_0", "sequential"],
        "help": "https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map"
    })
    offload_folder: str = field(default="/tmp/offload")

    @property
    def checkpoint(self) -> str:
        if isdir(self.model_name):
            return self.model_name
        HOME = expanduser("~")
        return join(HOME, "Dataset/transformers", self.model_name)


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


def load_tokenizer_and_model(model_args: ModelArguments):
    gpu_before = torch.cuda.memory_allocated()
    cpu_before = psutil.virtual_memory().used

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_args.checkpoint)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_args.checkpoint,
        device_map=model_args.device_map,
        torch_dtype=model_args.torch_dtype,
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


def do_generate(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, batch: List[PromptItem],
                generation_config: GenerationConfig) -> List[GeneratedItem]:
    batch_text = [item.prompt for item in batch]
    input_ids = tokenizer(batch_text, return_tensors="pt", padding=True)["input_ids"].to(0)
    output = model.generate(input_ids, generation_config=generation_config)
    batch_generated_text = tokenizer.batch_decode(output)
    results = [
        GeneratedItem(item.timestamp, time.time(), item.prompt, gen_text[len(item.prompt):]) for item, gen_text in
        zip(batch, batch_generated_text)
    ]
    return results


def inference(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, prompt_queue: queue.Queue, result_pool: dict,
              deploy_args: DeployArguments, generation_config: GenerationConfig):
    """
    Realize the following requirements.
    1. executing inference immediately when the inference demand is greater than or equal to the batch size.
    2. execute inference immediately when the request time in the queue exceeds timeout.
    """
    batch = []
    next_batch = []

    while True:
        try:
            item: PromptItem = prompt_queue.get(block=True, timeout=deploy_args.timeout)
        except queue.Empty:
            continue

        batch.append(item)
        time.sleep(deploy_args.timeout)

        while not prompt_queue.empty() and len(batch) < deploy_args.batch_size:
            curr_item = prompt_queue.get()
            if curr_item.timestamp - item.timestamp >= deploy_args.timeout:
                next_batch.append(curr_item)
                break
            else:
                batch.append(curr_item)

        generate_items = do_generate(tokenizer, model, batch, generation_config)
        for item in generate_items:
            result_pool[item.timestamp] = item
        batch.clear()
        batch.extend(next_batch)
        next_batch.clear()


def main():
    # Parse Arguments
    parser = HfArgumentParser([ModelArguments, GenerationArguments, DeployArguments])
    model_args, gen_args, deploy_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    gen_args: GenerationArguments
    deploy_args: DeployArguments
    show_args(model_args, logger)
    show_args(gen_args, logger)
    show_args(deploy_args, logger)

    # Prepare Message Queue & Result Pool
    prompt_queue = queue.Queue()
    result_pool = {}

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

    @app.route('/inference', methods=["POST"])
    def _inference():
        timestamp = time.time()
        data = request.get_data().decode()
        logger.info(f"{request}\t{data}")

        prompt = json.loads(data)["prompt"]
        prompt_queue.put(PromptItem(timestamp, prompt))

        # TODO: Block-able Dict
        while timestamp not in result_pool:
            time.sleep(0.05)

        response = vars(result_pool.pop(timestamp))
        logger.info(response)
        return jsonify(response)


    # Prepare Language Model
    tokenizer, model = load_tokenizer_and_model(model_args)
    generation_config = GenerationConfig(**vars(gen_args))

    # Background threading does inference
    generate_thread = threading.Thread(
        target=inference,
        kwargs={
            "tokenizer": tokenizer,
            "model": model,
            "prompt_queue": prompt_queue,
            "result_pool": result_pool,
            "deploy_args": deploy_args,
            "generation_config": generation_config,
        },
        daemon=True,
    )

    # Start threadings
    generate_thread.start()

    try:
        app.run(host=deploy_args.host, port=deploy_args.port, debug=False)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
        exit(0)

    generate_thread.join()


if __name__ == "__main__":
    main()
