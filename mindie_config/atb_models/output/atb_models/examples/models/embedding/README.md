# README

# 特性矩阵
- 此矩阵罗列了各向量化模型支持的特性

| 模型及参数量             | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|--------------------|----------------------------|-----------------------------|------|------|-----------------|-----------------|--------|---------|------------|------|-------|----------------|-----|-----|
| bge-large-zh-v1.5  | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |
| bge-reranker-large | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |
| bge-m3             | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |
| Conan-embedding-v1 | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |
| puff-large-v1      | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |

# 向量化模型-推理指导

<!-- TOC -->
* [README](#readme)
* [特性矩阵](#特性矩阵)
* [向量化模型-推理指导](#向量化模型-推理指导)
  * [概述](#概述)
    * [模型介绍](#模型介绍)
    * [开源权重](#开源权重)
    * [路径变量](#路径变量)
  * [推理环境准备](#推理环境准备)
  * [快速上手](#快速上手)
    * [获取本项目源码](#获取本项目源码)
    * [安装依赖](#安装依赖)
    * [获取开源模型权重](#获取开源模型权重)
    * [获取测试数据集](#获取测试数据集)
  * [模型推理](#模型推理)
  * [模型推理性能&精度](#模型推理性能精度)
<!-- TOC -->

## 概述

### 模型介绍

向量化模型是可将任意文本映射为低维稠密向量的语言模型，以用于检索、分类、聚类或语义匹配等任务，并可支持为大模型调用外部知识  
本项目支持 `BERT` 及 `XLMRoBERTa` 两种结构、 `embedding` 及 `rerank` 两种向量化类型的模型

> 💡 **如何确认模型的结构和向量化类型？**  
> 模型权重目录中的 `config.json` 文件配置了模型的结构和向量化类型，`"model_type"` 的值表示了模型结构，`"architectures"` 的值表示了模型的向量化类型（`*Model` 表示是 `embedding`，`*ForSequenceClassification` 表示是 `rerank`）


### 开源权重

[bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5/tree/main)  
[bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large/tree/main)  
[bge-m3](https://huggingface.co/BAAI/bge-m3/tree/main)  
[Conan-embedding-v1](https://huggingface.co/TencentBAC/Conan-embedding-v1/tree/main)  
[puff-large-v1](https://huggingface.co/infgrad/puff-large-v1/tree/main)

### 路径变量

**路径变量解释**

| 变量名            | 含义                                                                                                                            |
|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| `working_dir`  | 加速库及模型库下载后放置的目录                                                                                                               |
| `llm_path`     | 模型仓所在路径<br/>若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`<br/>若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| `script_path`  | 脚本所在路径<br/>向量化模型的脚本所在路径为 `${llm_path}/examples/models/embedding`                                                              |
| `weight_path`  | 模型权重所在路径                                                                                                                      |
| `dataset_path` | 数据集所在路径                                                                                                                       |


## 推理环境准备

- 参考[atb_models的README文件](../../../README.md)配置好推理环境
- 设置环境变量
    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

## 快速上手

### 获取本项目源码
    
```shell
cd ${working_dir}
git clone https://gitee.com/ascend/MindIE-LLM.git
cd MindIE-LLM
git checkout master
```

### 安装依赖

1. 安装 `atb_llm` 和 `atb_speed`

    > 需要设置好环境变量

    ```shell
    cd ${llm_path}/examples/atb_models
    pip install .
    ```
   
    ```shell
    cd ${llm_path}/examples/atb_models/examples/models/atb_speed_sdk
    pip install .
    ```

2. 安装其他python依赖

    ```shell
    cd ${script_path}
    pip install -r requirements.txt
    ```

### 获取开源模型权重

1. 点击[开源权重](#开源权重)中的链接，下载🤗HuggingFace模型官方页面中的所有文件至 `${weight_path}` 目录

2. 参考模型权重配置文件 `${weight_path}/config.json` 中 `"model_type"` 这项配置的值，用 `${script_path}` 下对应模型结构的 `modeling_${model_type}.py` 文件替换下载模型权重的 `${weight_path}/modeling_${model_type}.py`

3. 修改模型权重配置文件 `${weight_path}/config.json`，修改或添加 `"_name_or_path"` 和 `"auto_map"` 两项配置的值，使其映射至权重的实际路径
    - `"_name_or_path"`：修改为存放模型权重的实际路径 `${weight_path}`
    - `"auto_map"`：一般情况下，该配置需要手动添加，格式为一个“键为 `transformers` 库中对应模型结构的 `AutoModel` 类、值为 `"${weight_path}--modeling_${model_type}.'${architectures}'的值"` ”的字典（`${architectures}` 为配置文件 `${weight_path}/config.json` 中 `"architectures"` 这项配置的值）

例如 `XLMRoBERTa` 结构的 `rerank` 模型，配置文件中 `"model_type"` 的值为 `"xlm-roberta"`，因此需要替换模型权重路径中的 `modeling_xlm_roberta.py`，并对配置文件作如下修改：

```json
{
  "_name_or_path": "${weight_path}",
  "auto_map": {
    "AutoModelForSequenceClassification": "${weight_path}--modeling_xlm_roberta.XLMRobertaForSequenceClassification"
  }
}
```

4. 如果模型分词器配置文件 `${weight_path}/tokenizer_config.json` 中的 `model_max_length` 的值为类似 `1e30` 等的超大数值，需要修改其为 `${weight_path}/config.json` 中的 `max_position_embeddings` 的值

### 获取测试数据集

- `embedding` 模型使用 `T2Retrieval` 数据集，该数据集包含两部分，需要分别下载  
    [T2Retrieval](https://huggingface.co/datasets/C-MTEB/T2Retrieval/tree/main)  
    [T2Retrieval-qrels](https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels/tree/main)
- `rerank` 模型使用 `T2Reranking`数据集  
    [T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking/tree/main)

数据集下载后放在 `${dataset_path}` 目录中，并确保数据集拥有独立的子目录

## 模型推理

```shell
cd ${script_path}
python run.py \
  ${request} \
  --model_name_or_path=${weight_path} \
  --trust_remote_code \
  --device_type=${device_type} \
  --device_id=${device_id} \
  --text=${text}
```

- 参数说明
- `request`：执行的推理任务
  - `embedding` 模型输入 `embed`
  - `rerank` 模型输入 `rerank`
- `weight_path`：模型类型或模型文件路径
- `trust_remote_code` 为可选参数，代表是否信任本地可执行文件，默认不执行。传入此参数，则信任本地可执行文件
  - 使用ATB加速时，需要传入此参数用于读取本地的 `modeling_${model_type}.py` 文件
- `device_type`：加载模型的芯片类型
- `device_id`：加载模型的芯片id
- `text`：输入模型推理计算向量的文本
  - `embedding` 模型可输入多条文本，用空格分隔，如 `什么是大熊猫 属于食肉 
目熊科的一种哺乳动物 是一种小型犬品种`
  - `rerank` 模型可输入多条文本对，文本对需要用引号 `"` 包裹，文本对之间用空格分隔，文本对中的文本用 `,` 分隔，如 `"什么是大熊猫,属于食肉目熊科的一种哺乳动物" "什么是大熊猫,是一种小型犬品种"`

## 模型推理性能&精度

1. 性能测试

    吞吐率计算公式：$1000 \times \frac{batch\_size}{compute\_time}$

    ```shell
    python test.py \
      performance \
      --model_name_or_path=${weight_path} \
      --trust_remote_code \
      --device_type=${device_type} \
      --device_id=${device_id} \
      --batch_size=${batch_size} \
      --max_seq_len=${seq_len} \
      --loop=${loop} \
      --outputs=${outputs}
    ```
    
    - 参数说明
      - `weight_path`：模型类型或模型文件路径
      - `trust_remote_code` 为可选参数，代表是否信任本地可执行文件，默认不执行。传入此参数，则信任本地可执行文件
        - 使用ATB加速时，需要传入此参数用于读取本地的 `modeling_${model_type}.py` 文件
      - `device_type`：加载模型的芯片类型
      - `device_id`：加载模型的芯片id
      - `batch_size`：每轮推理的文本批次
      - `seq_len`：每轮推理的文本长度
      - `loop`：测试的循环次数，需要是正整数
      - `outputs`：测试结果的保存路径

2. 精度测试

    ```shell
    python test.py \
      ${task} \
      --model_name_or_path=${weight_path} \
      --trust_remote_code \
      --device_type=${device_type} \
      --device_id=${device_id} \
      --dataset_path=${dataset_path} \
      --batch_size=${batch_size} \
      --outputs=${outputs}
    ```
    
    - 参数说明
      - `task`：精度测试任务，`embedding` 模型输入 `retrieval`，`rerank` 模型输入 `reranking`
      - `weight_path`：模型类型或模型文件路径
      - `trust_remote_code` 为可选参数，代表是否信任本地可执行文件，默认不执行。传入此参数，则信任本地可执行文件
        - 使用ATB加速时，需要传入此参数用于读取本地的 `modeling_${model_type}.py` 文件
      - `device_type`：加载模型的芯片类型
      - `device_id`：加载模型的芯片id
      - `dataset_path`：数据集地址，`embedding` 模型的精度测试只需要输入 `T2Retrieval` 数据集的路径
      - `batch_size`：每轮推理的文本批次
      - `outputs`：测试结果的保存路径
