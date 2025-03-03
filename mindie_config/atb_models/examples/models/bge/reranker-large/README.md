# README

> ⚠️ 本模型仓库已迁移至 `atb_models/examples/models/embedding`，请参考新仓库的[README](../../embedding/README.md)

# 特性矩阵
- 此矩阵罗列了各bge-reranker-large模型支持的特性

| 模型及参数量             | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|--------------------|----------------------------|-----------------------------|------|------|-----------------|-----------------|--------|---------|------------|------|-------|----------------|-----|-----|
| bge-reranker-large | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |

# bge-reranker-large模型-推理指导

本项目同时支持OM离线版本模型和ATB加速版本模型推理，可根据使用版本选择必要的步骤

<!-- TOC -->
* [README](#readme)
* [特性矩阵](#特性矩阵)
* [bge-reranker-large模型-推理指导](#bge-reranker-large模型-推理指导)
  * [概述](#概述)
    * [模型介绍](#模型介绍)
    * [开源模型地址](#开源模型地址)
    * [路径变量](#路径变量)
    * [输入输出数据](#输入输出数据)
  * [推理环境准备](#推理环境准备)
  * [快速上手](#快速上手)
    * [获取本项目源码](#获取本项目源码)
    * [安装依赖](#安装依赖)
    * [获取开源模型权重](#获取开源模型权重)
    * [获取测试数据集](#获取测试数据集)
    * [【加速版本】编译模型仓](#加速版本编译模型仓)
    * [【离线版本】模型转换](#离线版本模型转换)
  * [模型推理](#模型推理)
  * [模型推理性能&精度](#模型推理性能精度)
    * [测试方法](#测试方法)
    * [模型推理性能](#模型推理性能)
    * [模型推理精度](#模型推理精度)
<!-- TOC -->

## 概述

### 模型介绍

`bge-reranker-large` 是由智源研究院研发的交叉编码器重排模型，可对查询和答案实时计算相关性分数，这比向量模型（即双编码器）更准确，但比向量模型更耗时

### 开源模型地址

```text
url=https://huggingface.co/BAAI/bge-reranker-large
commit_id=bc0c7056d15eaea221616887bf15da63743d19e1
model_name=bge-reranker-large
```

### 路径变量

**路径变量解释**

| 变量名            | 含义                                                                                                                            |
|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| `working_dir`  | 加速库及模型库下载后放置的目录                                                                                                               |
| `llm_path`     | 模型仓所在路径<br/>若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`<br/>若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| `script_path`  | 脚本所在路径<br/>`bge-reranker` 系列模型的脚本所在路径为 `${llm_path}/examples/models/bge/reranker-large`                                       |
| `weight_path`  | 模型权重所在路径                                                                                                                      |
| `dataset_path` | 数据集所在路径<br/>数据集下载路径为 `${llm_path}/tests/modeltest/dataset/full`                                                               |

**推荐目录结构**

```text
{llm_path}
├─ examples
│  └─ models
│     └─ bge
│        └─ reranker-large: {script_path}
│           └─ {weight_path}
└─ tests
   └─ modeltest
      └─ dataset
         └─ full: {dataset_path}
```

### 输入输出数据

**输入数据**

| 输入数据           | 数据类型  | 大小                   | 数据排布格式 |
|----------------|-------|----------------------|--------|
| input_ids      | INT64 | batch_size * seq_len | ND     |
| attention_mask | INT64 | batch_size * seq_len | ND     |

**输出数据**

| 输出数据   | 数据类型    | 大小                 | 数据排布格式 |
|--------|---------|--------------------|--------|
| output | FLOAT32 | batch_size * class | ND     |

## 推理环境准备

**该模型需要以下插件与驱动**

| 配套                 | 版本                         | 环境准备指导                                                                                                        |
|--------------------|----------------------------|---------------------------------------------------------------------------------------------------------------|
| Ascend HDK         | 24.1.RC2                   | [Pytorch框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies/pies_00001.html) |
| CANN               | 8.0.RC2                    | -                                                                                                             |
| FrameworkPTAdapter | FrameworkPTAdapter 6.0.RC2 |                                                                                                               |
| Python             | 3.10                       | -                                                                                                             |
| Pytorch            | 2.1.0                      | -                                                                                                             |

## 快速上手

### 获取本项目源码
    
```shell
cd ${working_dir}
git clone https://gitee.com/ascend/MindIE-LLM.git
cd MindIE-LLM
git checkout master
```

### 安装依赖

1. 【共通】安装 `atb_llm` 和 `atb_speed`

    说明：需要source `ascend-toolkit` 及 `nnal` 环境变量

    如果尝试导入 `atb_llm` 时出现 `ImportError`，则需要参考[【加速版本】编译模型仓](#加速版本编译模型仓)编译模型仓

    ```shell
    cd ${llm_path}/examples/atb_models
    pip install .
    ```
   
    ```shell
    cd ${llm_path}/examples/atb_models/examples/models/atb_speed_sdk
    pip install .
    ```

2. 【共通】安装其他python依赖

    ```shell
    cd ${script_path}
    pip install -r requirements.txt
    ```

3. 【离线版本】下载安装 `ais_bench` 推理工具

    [ais_bench推理工具使用指南](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README.md)

    ```shell
    pip install ./aclruntime-{version}-{python_version}-linux_{arch}.whl
    pip install ./ais_bench-{version}-py3-none-any.whl
    # {version}表示软件版本号，{python_version}表示Python版本号，{arch}表示CPU架构
    ```

### 获取开源模型权重

可在以下方法中任选一种获取开源模型权重

- 使用命令行获取
  - ```shell
    mkdir ${weight_path}
    cd ${script_path}
    # Make sure you have git-lfs installed (https://git-lfs.com)
    git lfs install
    git clone https://huggingface.co/BAAI/bge-reranker-large
    mv bge-reranker-large ${weight_path}
    ```
- 访问[模型页面](https://huggingface.co/BAAI/bge-reranker-large/tree/main)获取
  - 下载 `HuggingFace` 原始模型所有文件至 `${weight_path}` 目录

修改 `${script_path}/config_bge_reranker.json` 配置文件中模型权重路径参数为权重的实际路径

修改 `${weight_path}/config.json` 配置文件，添加 `"_name_or_path"` 和 `"auto_map"` 两项配置并映射至权重的实际路径

```json
"_name_or_path": "${weight_path}",
"auto_map": {
  "AutoModel": "${weight_path}--modeling_xlm_roberta.XLMRobertaModel",
  "AutoModelForSequenceClassification": "${weight_path}--modeling_xlm_roberta.XLMRobertaModelForSequenceClassification"
}
```

### 获取测试数据集

可在以下方法中任选一种下载 [C-MTEB/T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking) 数据集
 
- 使用命令行获取
  - ```shell
    cd ${dataset_path}
    git clone https://huggingface.co/datasets/C-MTEB/T2Reranking
    mv T2Reranking/data/dev-00000-of-00001-65d96bde8023d9b9.parquet T2Reranking/
    ```
- 访问数据集页面获取
  - ```shell
    mkdir T2Reranking
    ``` 
  - 下载数据集文件 [data/dev-00000-of-00001-65d96bde8023d9b9.parquet](https://huggingface.co/datasets/C-MTEB/T2Reranking/resolve/main/data/dev-00000-of-00001-65d96bde8023d9b9.parquet)至 `${dataset_path}/T2Reranking` 目录中

修改 `${script_path}/eval_performance.py` 和 `${script_path}/eval_t2reranking.py` 脚本文件中的数据集路径为实际路径

### 【加速版本】编译模型仓

可参考[模型仓的README文件](../../../../README.md)

```shell
cd ${llm_path}
bash scripts/build.sh
source output/atb_models/set_env.sh
```

### 【离线版本】模型转换

运行脚本将 `ONNX` 格式模型转换为 `OM` 格式模型

```shell
bash ${script_path}/convert.sh ${weight_path} ${om_path} ${precision_mode}
```

- 参数说明，参考 [ATC工具参数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devaids/auxiliarydevtool/atlasatc_16_0039.html)
  - `om_path`：转换后的om模型文件的存放目录
  - `precision_mode`：模型精度模式，精度高低排序 `origin>mixed_float16>fp16`，性能优劣排序 `fp16>=mixed_float16>origin`，推荐使用 `mixed_float16` 以在保证精度的前提下获得最大性能，默认为 `mixed_float16`

## 模型推理

```shell
cd ${script_path}
python run.py \
  --model_type_or_path=${model_type} or ${model_path}
  --device=${device}
```

- 参数说明
- `model_type_or_path`：模型类型或模型文件路径
- `device`：加载模型的芯片id

## 模型推理性能&精度

NPU环境使用 `OM` 模型，GPU环境使用 `ONNX` 模型

### 测试方法

1. 性能测试

    ```shell
    python eval_performance.py \
      --model_type_or_path=${model_type} or ${model_path} \
      --input_shape=${batch_size},${seq_len} \
      --loop=${loop} \
      --device=${device}
    ```
    
    - 参数说明
      - `model_type_or_path`：模型类型或模型文件路径
      - `batch_size`：每轮推理的文本数量
      - `seq_len`：每轮推理的文本长度
      - `loop`：验证循环次数，需要是正整数，默认为50
      - `device`：加载模型的芯片id

2. 精度测试

    ```shell
    python eval_t2reranking.py \
      --model_type_or_path=${model_type} or ${model_path} \
      --batch_size=${batch_size} \
      --device=${device}
    ```
    
    - 参数说明
      - `model_type_or_path`：模型类型或模型文件路径
      - `batch_size`：单次推理时读取的数据集文本数量，需要是正整数，默认为20
      - `device`：加载模型的芯片id

### 模型推理性能

吞吐率计算公式：$1000 \times \frac{batch\_size}{compute\_time}$

**离线版本**

NPU环境使用 `OM` 格式模型，GPU环境使用 `ONNX` 格式模型

> 说明：Atlas 300I Duo (300I DUO) 推理卡为单卡双芯，比较吞吐率时需要 $×2$，下表已按照该方法处理

| 环境  | 芯片型号        | dtype         | batch_size | seq_len | 吞吐率（fps） |
|-----|-------------|---------------|------------|---------|----------|
| NPU | 300I DUO    | mixed_float16 | 20         | 512     | 43.84    |
| NPU | 300I DUO    | mixed_float16 | 50         | 512     | 44.23    |
| NPU | 800I A2     | mixed_float16 | 20         | 512     | 145.91   |
| NPU | 800I A2     | mixed_float16 | 50         | 512     | 145.49   |
| GPU | NVIDIA A10  | float32       | 20         | 512     | 46.43    |
| GPU | NVIDIA A10  | float32       | 50         | 512     | 49.16    |
| GPU | NVIDIA L40S | float32       | 20         | 512     | 108.86   |
| GPU | NVIDIA L40S | float32       | 50         | 512     | 103.11   |

**加速版本**

| 环境  | 芯片型号        | dtype | batch_size | seq_len | 吞吐率（fps） |
|-----|-------------|-------|------------|---------|----------|
| NPU | 800I A2     | fp16  | 20         | 512     | 232.76   |
| NPU | 800I A2     | fp16  | 50         | 512     | 228.24   |
| GPU | NVIDIA L40S | fp16  | 20         | 512     | 177.27   |
| GPU | NVIDIA L40S | fp16  | 50         | 512     | 203.35   |

### 模型推理精度

> 说明：精度测试采用 `batch_size = 50` 方法测试

| 环境  | 芯片型号        | dtype | MAP（%） | MRR@10（%） |
|-----|-------------|-------|--------|-----------|
| NPU | 800I A2     | fp16  | 67.60  | 77.67     |
| GPU | Nvidia L40S | fp16  | 67.60  | 77.66     |

说明：

- MAP：平均精度均值（Mean Average Precision）$MAP = \frac{1}{|U|} \sum_{i=1}^{|U|} hit(i) \times \frac{1}{P_i}$
- MRR：平均倒数排名（Mean Reciprocal Rank）$MRR = \frac{1}{N} \sum_{i=1}^N \frac{1}{p_i}$
