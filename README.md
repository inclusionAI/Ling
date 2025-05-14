# Ling
<p align="center"><img src="./figures/ant-bailing.png" width="100"/></p>

<p align="center">🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a></p>


## Introduction

Ling is a MoE LLM provided and open-sourced by InclusionAI. We introduce two different sizes, which are Ling-lite and Ling-plus. Ling-lite has 16.8 billion parameters with 2.75 billion activated parameters, while Ling-plus has 290 billion parameters with 28.8 billion activated parameters. Both models demonstrate impressive performance compared to existing models in the industry.

Their structure makes it easy to scale up and down and adapt to different tasks, so users can use these models for a wide range of tasks, from processing natural language to solving complex problems. Furthermore, the open-source nature of Ling promotes collaboration and innovation within the AI community, fostering a diverse range of use cases and enhancements.

As more developers and researchers engage with the platform, we can expect rapid advancements and improvements, leading to even more sophisticated applications. This collaborative approach accelerates development and ensures that the models remain at the forefront of technology, addressing emerging challenges in various fields.

## Update

- [2025-5-10] Ling-lite-1.5 has been released! It achieves significant progress in reasoning ability compared with previous Ling-lite. 
- [2025-4-15] Ling-lite is upgraded to Ling-lite-0415. The new model demonstrates notable improvements over its predecessor, Ling-lite-0220, especially on code and math.

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.


|      **Model**       | **#Total Params** | **#Activated Params** | **Context Length** |                                                                        **Download**                                                                        |
| :------------------: | :---------------: | :-------------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    Ling-lite-base-1.5    |       16.8B       |         2.75B         |        128K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-base-1.5) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-lite-base-1.5)     |
|      Ling-lite-1.5       |       16.8B       |         2.75B         |        128K         |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-lite-1.5) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-lite-1.5)          |
|    Ling-plus-base    |       290B        |         28.8B         |        64K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-plus-base) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-plus-base)     |
|      Ling-plus       |       290B        |         28.8B         |        64K         |          [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-plus) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ling-plus)          |
| Ling-coder-lite-base |       16.8B       |         2.75B         |        16K         | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite-base) <br>[🤖 ModelScope](https://modelscope.cn/models/inclusionAI/Ling-Coder-lite-base) |
|   Ling-coder-lite    |       16.8B       |         2.75B         |        16K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ling-Coder-lite) <br>[🤖 ModelScope](https://modelscope.cn/models/inclusionAI/Ling-Coder-lite)      |


Note: If you are interested in previous version, please visit the past model collections in [Huggingface](https://huggingface.co/inclusionAI) or [ModelScope](https://modelscope.cn/organization/inclusionAI).

## Evaluation

### Ling-lite

#### Standard Benchmarks

| **Benchmark**             | **#shots** | **Ling-lite-1.5** | **Ling-lite** | **Qwen3-4B-Instruct** | **Qwen3-8B-Instruct** | **Moonlight-16B-A3B-Instruct** | **LLaMA3.1-8B** |
| :--------------------------------------------: | :--------: | :---------------: | :-----------: | :-------------------: | :-------------------: | :-----------: | :-------------: |
| MMLU(EM)              | 5      | **74.33**         | 71.27     | 70.09             | 75.97             | 70.74     | 68.67       |
| GPQA(Pass@1)          | 0      | **36.55**         | 29.73     | 40.4              | 47.10             | 19.51     | 27.59       |
| HumanEval(Pass@1)     | 0      | **87.27**         | 84.38     | 81.94             | 85.29             | 72.94     | 67.23       |
| LiveCodeBench 2408-2502 (Pass@1) | 0      | **22.7**          | 18.94     | 21.8              | 26.88             | 14.76     | 18.41       |
| LCBench(pass@1)       | 0      | **60.37**         | 46.57     | 48.61             | 60.03             | 28.39     | 23.13       |
| Math(EM)              | 0      | **82.62**         | 72.80     | 81.46             | 82.70             | 67.1      | 52.42       |
| AIME2024(pass@1)      | 0      | **21.88**         | 10.21     | 20.62             | 26.25             | 6.88      | 7.29        |
| OlympiadBench(pass@1) | 0      | **52.30**         | 36.44     | 54.33             | 56.11             | 32.85     | 17.04       |
| BBH(EM)               | 0      | **75.75**         | 66.38     | 78.21             | 79.33             | 63.45     | 68.05       |
| IFEval(Prompt Strict) | 0      | **77.70**         | 77.99     | 81.06             | 83.55             | 49.01     | 73.01       |
| BFCL_live | 0 | **72.15** | 67.93 | 65.35 | 69.83 | 47.14 | 49.98 |

#### Context Window
![](./figures/needle_testing.png)

Evaluation results on the ``Needle In A Haystack`` (NIAH) tests. Ling-lite-1.5 has improved long text generation capability and performs well across most context window lengths up to **128K**. 

## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ling-lite-1.5"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>.

## Deployment

### vLLM

vLLM supports offline batched inference or launching an OpenAI-Compatible API Service for online inference.

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below:

```bash
git clone -b  v0.7.3 https://github.com/vllm-project/vllm.git
cd vllm
git apply Ling/inference/vllm/bailing_moe.patch
pip install -e .
```

#### Offline Inference:

```bash
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-lite-1.5")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=16384)

llm = LLM(model="inclusionAI/Ling-lite", dtype='bfloat16')
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)

```

#### Online Inference:

```bash
vllm serve inclusionAI/Ling-lite \
              --tensor-parallel-size 2 \
              --pipeline-parallel-size 1 \
              --use-v2-block-manager \
              --gpu-memory-utilization 0.90
```

To handle long context in vLLM using YaRN, we need to follow these two steps:
1. Add a `rope_scaling` field to the model's `config.json` file, for example:
```json
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}
```
2. Use an additional parameter `--max-model-len` to specify the desired maximum context length when starting the vLLM service.

For detailed guidance, please refer to the vLLM [`instructions`](https://docs.vllm.ai/en/latest/).

### MindIE

This topic describes the main steps to run an Ling MoE model based on Huawei NPU cards and the MindIE inference framework

#### Hardware Requirements

- The MoE Plus model requires at least 2 Atlas 800I A2 (8\*64G) servers.
- The MoE Lite model requires at least 1 Atlas 800I A2 (8\*64G) server.

#### Configure preparation

Create a model directory on the host for downloading, the directory example is: /root/models', which is used to mount the docker container later.

Download the mindie-related configuration from github:

```bash
cd /root/models
git clone git@github.com:inclusionAI/Ling.git
```

#### Machine network environment check

```bash
# Check the physical link
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done
# Check the links
for i in {0..7}; do hccn_tool -i $i -link -g ; done
# Check your network health
for i in {0..7}; do hccn_tool -i $i -net_health -g ; done
# Check whether the detected IP address is correctly configured
for i in {0..7}; do hccn_tool -i $i -netdetect -g ; done
# Check whether the gateway is configured correctly
for i in {0..7}; do hccn_tool -i $i -gateway -g ; done
# Check the consistency of the underlying TLS verification behavior of the NPU, recommend that all 0 be
for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch
# The underlying TLS check line of the NPU is set to 0
for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done
```

#### Pull the image

Go to [Ascend Community/Development Resources](https://www.hiascend.com/developer/ascendhub) and pull the mindie image

Image version: 1.0.0-800I-A2-py311-openeuler24.03-lts

The versions of each component are as follows:
| Component | Version     |
| --------- | ----------- |
| MindIE    | 1.0.0       |
| CANN      | 8.0.0       |
| PTA       | 6.0.0.beta1 |
| HDK       | 24.1.0      |

#### Container startup and configuration changes

##### Start the container

Execute the following startup command (reference):

```bash
docker run -itd --privileged --name=container name --net=host \
--shm-size 500g \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device /dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /root/models:/home/HwHiAiUser/Ascend \
mindie: 1.0.0-XXX-800I-A2-arm64-py3.11 (modified according to the name of the loaded image) \
bash
```

##### Download the model

In this case, we use ModelScope to download the model, and install ModelScope first:

```bash
pip install modelscope
```

Download the model:

```bash
# The model takes a long time to download and can be executed in the background
nohup modelscope download --model inclusionAI/Ling-plus --local_dir /home/HwHiAiUser/Ascend/Ling_plus 2>&1 > /tmp/ling_plus.log &

nohup modelscope download --model inclusionAI/Ling-plus-base --local_dir /home/HwHiAiUser/Ascend/Ling_plus_base 2>&1 > /tmp/ling_plus_base.log &

nohup modelscope download --model inclusionAI/Ling-lite --local_dir /home/HwHiAiUser/Ascend/Ling_lite 2>&1 > /tmp/ling_lite.log &

nohup modelscope download --model inclusionAI/Ling-lite-base --local_dir /home/HwHiAiUser/Ascend/Ling_lite_base 2>&1 > /tmp/ling_lite_base.log &
```

After the download is completed, you need to change the file permissions, otherwise an error will be reported when MindIE-Service is started:

```bash
chmod -R 750 *.json *.py
```

##### Model weight format conversion

> This section applies to the Ling Lite model, the Ling Plus model does not need to worry about this chapter

mindie supports safetensors format weights, if the download weights are not in safetensors format, you need to convert the weights, take Ling Lite as an example, the conversion command is as follows:

```bash
# Convert Ling lite
python /home/HwHiAiUser/Ascend/Ling/inference/mindie/convert_bin_to_safetensor.py

cd /home/HwHiAiUser/Ascend/Ling_lite
cp README.md configuration.json config.json special_tokens_map.json modeling_bailing_moe.py tokenizer.json tokenizer_config.json ../Ling_lite_safetensor/

# Convert Ling lite base
python /home/HwHiAiUser/Ascend/Ling/inference/mindie/convert_bin_to_safetensor_base.py

cd /home/HwHiAiUser/Ascend/Ling_lite_base
cp README.md configuration.json config.json special_tokens_map.json modeling_bailing_moe.py tokenizer.json tokenizer_config.json ../Ling_lite_base_safetensor/
```

The path of loading the Ling Lite model is changed to '/home/HwHiAiUser/Ascend/Ling_lite_safetensor', and the path of the Ling Lite Base model is changed to '/home/HwHiAiUser/Ascend/Ling_lite_base_safetensor'

##### Change the model configuration

The default model configuration file (config.json) mindie cannot be loaded directly, and needs to be changed:

```bash
# Adapt to mindie's Ling lite model configuration
cp /home/HwHiAiUser/Ascend/Ling_lite_safetensor/config.json /home/HwHiAiUser/Ascend/Ling_lite_safetensor/config.json.bak
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/lite/model_chat_config.json /home/HwHiAiUser/Ascend/Ling_lite_safetensor/config.json
chmod 750 /home/HwHiAiUser/Ascend/Ling_lite_safetensor/config.json

# Adapt to mindie's Ling lite base model configuration
cp /home/HwHiAiUser/Ascend/Ling_lite_base_safetensor/config.json /home/HwHiAiUser/Ascend/Ling_lite_base_safetensor/config.json.bak
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/lite/model_base_config.json /home/HwHiAiUser/Ascend/Ling_lite_base_safetensor/config.json
chmod 750 /home/HwHiAiUser/Ascend/Ling_lite_base_safetensor/config.json

# Adapt to mindie's Ling plus model configuration
cp /home/HwHiAiUser/Ascend/Ling_plus/config.json /home/HwHiAiUser/Ascend/Ling_plus/config.json.bak
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/plus/model_chat_config.json /home/HwHiAiUser/Ascend/Ling_plus/config.json
chmod 750 /home/HwHiAiUser/Ascend/Ling_plus/config.json

# Adapt to mindie's Ling plus base model configuration
cp /home/HwHiAiUser/Ascend/Ling_plus_base/config.json /home/HwHiAiUser/Ascend/Ling_plus_base/config.json.bak
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/plus/model_base_config.json /home/HwHiAiUser/Ascend/Ling_plus_base/config.json
chmod 750 /home/HwHiAiUser/Ascend/Ling_plus_base/config.json
```

Execute the shell script that adapts the mindie to the Ling model:

```bash
bash /home/HwHiAiUser/Ascend/Ling/inference/mindie/patch_atb_llm.sh
```

#### Stand-alone Servitization Inference (Ling lite)

Set the underlying environment variables:

```bash
source /usr/local/Ascend/atb-models/set_env.sh
```

Set different mindie configurations according to the model type:

```bash
# Ling Lite
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/lite/config.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

# Ling Lite base
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/lite/config.base.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

Start the mindie service:

```bash
chmod 640 /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

cd $MIES_INSTALL_PATH
nohup ./bin/mindieservice_daemon > /tmp/service.log 2>&1 &
```

Check /tmp/service.log to check whether the output is Daemon start success!, if so, it means that MindIE-Service has started successfully.

Test if the request is correct:

```bash
# Chat model
wget -O- --post-data="{\"messages\":[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"Who are you?\"}], \"stream\": false, \"max_tokens\":100, \"model\": \"bailing_moe\", \"temperature\":0}" \
--header='Content-Type:application/json' \
'http://127.0.0.1:1025/v1/chat/completions'

# base model

wget -O- --post-data='{"inputs":"My name is Olivier and I","stream":false,"parameters":{"temperature":1,"max_new_tokens":100,"do_sample":false}}' \
--header='Content-Type:application/json' \
'http://127.0.0.1:1025/infer'
```

#### Multi-machine service-based inference (Ling plus)

All of the following commands need to be executed simultaneously on all machines.

To enable multi-machine service-based inference, you need to configure a multi-machine ranktable file.

- Get the IP address of each card (on the host)

```bash
for i in {0..7}; do hccn_tool -i $i -ip -g; done
```

- Configure 'rank_table.json' in the following format and put it in '/root/models' so that it can be mounted to the container

```json
{
"server_count": "...", # Total number of nodes
# The first server in the server_list is the primary node
"server_list": [
{
"device": [
{
"device_id": "...", # The number of the current card, the value range is [0, the number of cards in the machine)
"device_ip": "...", # The IP address of the current card, which can be obtained by hccn_tool command
"rank_id": "..." # The global number of the current card, the value range is [0, total number of cards)
},
...
],
"server_id": "...", # IP address of the current node
"container_ip": "..." # The IP address of the container (required for service-based deployment) is the same as that of the server_id unless otherwise configured
},
...
],
"status": "completed",
"version": "1.0"
}
```

Enter the container and run the following command:

```bash
# Set the basic environment variables:
source /home/HwHiAiUser/Ascend/Ling/inference/mindie/set_env.sh
# Enable communication environment variables
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200
export WORLD_SIZE=16
export HCCL_EXEC_TIMEOUT=0

# Configure virtual memory environment variables
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #开启
# Fixed the issue of slow weight loading

export OMP_NUM_THREADS=1


export RANKTABLEFILE=/home/HwHiAiUser/Ascend/rank_table.json
chmod 640 /home/HwHiAiUser/Ascend/rank_table.json

# To serve, you need to configure the 'container_ip' field in 'ranktable.json', and the configuration of all machines should be consistent, except for the MIES_CONTAINER_IP of the environment variable is the local IP address.
export MIES_CONTAINER_IP=IP address of the container

```

Set different mindie configurations according to the model type:

```bash
# Ling plus
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/plus/config.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

# Ling plus base
cp /home/HwHiAiUser/Ascend/Ling/inference/mindie/plus/config.base.json /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

Modify the servitization parameters:

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
# The following configurations need to be changed

# "ipAddress" : "Change to primary node IP",
# "managementIpAddress" : "Change to primary node IP",
```

To set the memory usage ratio:

```bash
export NPU_MEMORY_FRACTION=0.95
```

Pull up servitization:

```bash
cd $MIES_INSTALL_PATH
nohup ./bin/mindieservice_daemon > /tmp/service.log 2>&1 &
```

When the command is executed, all the parameters used for this startup are first printed, and then until the following output appears:

`Daemon start success!`

The service is considered to have started successfully.

Test if the request is correct:

```
# Chat model
wget -O- --post-data="{\"messages\":[{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"Who are you?\"}], \"stream\": false, \"max_tokens\":100, \"model\": \"bailing_moe\", \"temperature\":0}" \
--header='Content-Type:application/json' \
'http://<Change to primary node IP>:1025/v1/chat/completions'

# base model

wget -O- --post-data='{"inputs":"My name is Olivier and I","stream":false,"parameters":{"temperature":1,"max_new_tokens":100,"do_sample":false}}' \
--header='Content-Type:application/json' \
'http://<Change to primary node IP>:1025/infer'
```

## Finetuning

We recommend you to use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to finetune Ling with SFT, DPO, etc.

We use [`identity`](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json) to demonstrate how to finetune our Ling models by replacing `name` with `Ling` and `author` with `inclusionAI`.

```json
{
  "instruction": "hi",
  "input": "",
  "output": "Hello! I am Ling, an AI assistant developed by inclusionAI. How can I assist you today?"
}
```

We provide a demo configuration of `Llama-Factory` to SFT Ling models as follows:

```bash
llamafactory-cli train examples/sft/ling_full_sft.yaml
```

## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ling/blob/master/LICENCE).

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{ling,
    title   = {Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs}, 
    author  = {Ling Team},
    journal = {arXiv preprint arXiv:2503.05139},
    year    = {2025}
}
```
