# README

Deploy large language model service with HuggingFace and Flask.

## Deploy Language Model

```bash
usage: deploy_llm.py [-h] [--model_name MODEL_NAME]
                     [--torch_dtype {auto,fp16,fp32}]
                     [--device_map {auto,balanced,balanced_low_0,sequential,smart}]
                     [--gpu_max_memory GPU_MAX_MEMORY]
                     [--cpu_max_memory CPU_MAX_MEMORY]
                     [--offload_folder OFFLOAD_FOLDER]
                     [--max_new_tokens MAX_NEW_TOKENS]
                     [--early_stopping [EARLY_STOPPING]] [--no_early_stopping]
                     [--do_sample [DO_SAMPLE]] [--no_do_sample]
                     [--temperature TEMPERATURE] [--host HOST] [--port PORT]
                     [--batch_size BATCH_SIZE] [--timeout TIMEOUT]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name in HuggingFace model hub (default:
                        bigscience/bloomz-3b)
  --torch_dtype {auto,fp16,fp32}
  --device_map {auto,balanced,balanced_low_0,sequential,smart}
                        https://huggingface.co/docs/accelerate/usage_guides/big_modeling#designing-a-device-map (default: auto)
  --gpu_max_memory GPU_MAX_MEMORY
  --cpu_max_memory CPU_MAX_MEMORY
  --offload_folder OFFLOAD_FOLDER
  --max_new_tokens MAX_NEW_TOKENS
  --early_stopping [EARLY_STOPPING]
  --no_early_stopping
  --do_sample [DO_SAMPLE]
  --no_do_sample
  --temperature TEMPERATURE
  --host HOST
  --port PORT
  --batch_size BATCH_SIZE
  --timeout TIMEOUT     Timeout second (default: 0.2)
```

### REST API

#### Show Config State

```bash
curl -X GET http://<target_host>:<target_port>/state
```


#### Show Model

```bash
curl -X GET http://<target_host>:<target_port>/model
```


#### Show Tokenizer

```bash
curl -X GET http://<target_host>:<target_port>/tokenizer
```


#### Do Model Inference

```bash
curl -X GET -d '{"prompt":"What is your name?"}' http://<target_host>:<target_port>/inference
```

请求格式：
```json
{
  "prompt": "1+1="
}
```

回复格式：
```json
{
  "timestamp": 154545.054356,
  "finish_timestamp": 154548.87439,
  "prompt": "1+1=",
  "generated_text": "2"
}
```
