## 1. Introduction

We present Ling-moe, a strong Mixture-of-Experts (MoE) language model xxxxxxxxxxx
<p align="center">
  <img width="80%" src="figures/ling-moe-lite.jpg">
</p>

## 2. Model Summary

---

**Compatibility of computing power**

- To avoid over-reliance on a specific training environ- ment, e.g., NVIDIA GPU series, and ensure compatibility with different training environments, e.g., Huawei GPU series, we propose a cross-platform training framework, namely dlrover, to train our MoE series on various computing devices.
-  We investigate a Multi-Token Prediction (MTP) objective and prove it beneficial to model performance. 
    It can also be used for speculative decoding for inference acceleration. 

---

**High-quality data**

- Throughout the entire training process, we take a series of strategies to clean and choose high-quality data. During pre-training, we train our MoE models on xxT high- quality tokens, involving different abilities, i.e., knowledge, language, reasoning (code, math, and logical reasoning), and xxx.

---

**High-performance inference**

- To speed up the generation of above-mentioned high-quality data, we delelop a high-performance inference framework, i.e., flood, which can increase the generation speed by more than 50% compared to vllm in data generation scenarios. Also, compared to other LLM models, high-performance inference capabilities offer significant advantages in practical application scenarios.

---


## 3. Model Downloads

<div align="center">

| **Model** | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| Ling-Moe-Lite-Base |  |  |    | link|
| Ling-Moe-Lite-Chat  |  |  |     | link|
| Ling-Moe-Plus-Base   |  |  |     |link|
| Ling-Moe-Plus-Chat |  |  |     |link|

</div>

## 4. Evaluation Results
### Base Model
#### Lite Benchmarks

<div align="center">

|  | Benchmark (Metric) | # Shots | DeepSeek-V2 | Qwen2.5 72B | LLaMA3.1 405B | DeepSeek-V3 |
|---|-------------------|----------|--------|-------------|---------------|---------|

</div>


#### Plus Benchmarks

<div align="center">

|  | Benchmark (Metric) | # Shots | DeepSeek-V2 | Qwen2.5 72B | LLaMA3.1 405B | DeepSeek-V3 |
|---|-------------------|----------|--------|-------------|---------------|---------|

</div>

The Bailing base model is pre-trained in a multilingual training set that includes both English and Chinese data. Therefore, we evaluate the performance of our base model using various datasets including both Chinese and English. Specifically, the evaluation benchmarks we utilize are categorized into the following 4 types, where Chinese benchmarks are underlined and English benchmarks are double-underlined.
For more evaluation details, please check our paper. 


### Chat Model
#### Lite Benchmarks
<div align="center">

| | **Benchmark (Metric)** | **DeepSeek V2-0506** | **DeepSeek V2.5-0905** | **Qwen2.5 72B-Inst.** | **Llama3.1 405B-Inst.** | **Claude-3.5-Sonnet-1022** | **GPT-4o 0513** | **DeepSeek V3** |
|---|---------------------|---------------------|----------------------|---------------------|----------------------|---------------------------|----------------|----------------|

Comparison between Bailing-MoE-Lite-Chat model and other representative models.
</div>

#### Plus Benchmarks
<div align="center">

| | **Benchmark (Metric)** | **DeepSeek V2-0506** | **DeepSeek V2.5-0905** | **Qwen2.5 72B-Inst.** | **Llama3.1 405B-Inst.** | **Claude-3.5-Sonnet-1022** | **GPT-4o 0513** | **DeepSeek V3** |
|---|---------------------|---------------------|----------------------|---------------------|----------------------|---------------------------|----------------|----------------|

Comparison between Bailing-MoE-Plus-Chat and other representative models.
</div>


## 5. How to Run Locally

Ling-moe can be deployed locally using the following hardware and open-source community software:

1. **LLama Factory Demo**: We provide a simple demo for training, inference and evaluation of model using LLama Factory.
2. **Huawei Ascend NPU**: Supports running Ling-moe on Huawei Ascend devices.

**NOTE: Huggingface's Transformers has not been directly supported yet.**

### 5.1 Inference with LLama Factory Demo (example only)

#### Model Weights & Demo Code Preparation
**Prepare the Environment**
First, clone our inclusionAI GitHub repository:

```shell
git clone https://github.com/inclusionAI/moe.git
```

Clone LLaMA-Factory GitHub repository and install dependencies:

```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Enter the following command to get guidance on training-related parameters and to test whether the llama factory was installed successfully.
```
llamafactory-cli train -h
```

**Download the model weights**
Download the model weights from HuggingFace, and put them into `/path/model/Ling_moe_lite` folder.
**Prepare a dataset**
Taking the SFT stage as an example, the llama factory supports two data formats, Alpaca and ShareGPT. The Alpaca format is as follows:
```
{
  "instruction": "写一个有效的比较语句",
  "input": "篮球和足球",
  "output": "篮球和足球都是受欢迎的运动。"
}
```
More examples can be found in the files under the ```data``` directory of the llama factory, and the new dataset needs to be registered in ```data/dataset_info.json```.
**Prepare the script**
Taking full SFT as an example, its script is as follows:
```
#ling_full_sft.yaml
### model
model_name_or_path: /path/model/Ling_moe_lite
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: identity
template: bailing
packing: true
cutoff_len: 2048
overwrite_cache: true
preprocessing_num_workers: 2

### output
output_dir: saves/Ling_moe_lite/full/sft
report_to: tensorboard
logging_dir: saves/Ling_moe_lite/full/sft/run
logging_steps: 10
save_strategy: epoch
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 2.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```
The meanings of some key parameters are as shown in the table below.
| Parameter Name               | Parameter Description                                                                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| stage                        | The current training stage. Enum values include "sft", "pt", "rm", "ppo", etc., representing different training phases. |
| do_train                     | Indicates whether the model is in training mode.                                                                                  |
| dataset                      | The list of datasets to be used. All fields must be registered in `data_info.json` as mentioned above. Multiple datasets can be separated by commas (","). |
| finetuning_type              | The type of fine-tuning training. Enum values include "lora", "full", "freeze", etc. Here, "full" is used.                                               |
| output_dir                   | The directory where the training results will be saved.                                                                                 |
| cutoff_len                   | The maximum sequence length for truncating the training dataset.                                                                               |
| per_device_train_batch_size  | The batch size per device. The minimum value is 1. If GPU memory is sufficient, this value can be increased as needed.                                   |
| fp16                         | Indicates whether to use mixed precision training (half-precision).                                                                      |
| max_samples                  | The number of samples to draw from each dataset.                                                                               |
deepspeed                     | DeepSpeed-related configurations, specifically including `ds_z0_config.json`, `ds_z2_config.json`, `ds_z3_config.json`.                                             |

**Start Training**
Use the following commands to run Full fine-tuning:
```
llamafactory-cli train examples/train_sft/ling_full_sft.yaml
```
**Inference**  
You can perform inference and interact with the model using `llamafactory-cli chat inference_config.yaml`. When configuring the file for interaction, you only need to specify the base model `model_name_or_path` and the `template`. The configuration file is as follows:
```
### examples/inference/Ling-moe-lite.yaml
model_name_or_path: inclusionAI/Ling-moe-lite
template: bailing
infer_backend: huggingface #choices： [huggingface, vllm]
```

### 5.2 Recommended Inference Functionality with Huawei Ascend NPUs
The [MindIE](https://www.hiascend.com/en/software/mindie) framework from the Huawei Ascend community has successfully adapted the BF16 version of Ling-moe. xxxxxxx.


## 6. License
This code repository is licensed under [the MIT License](LICENSE-CODE). xxxx

## 7. Citation
```

```

## 8. Contact
If you have any questions, please raise an issue or contact us at xxx