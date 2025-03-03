# README

> ⚠️ 本模型仓库已迁移至 `atb_models/examples/models/embedding`，请参考新仓库的[README](../../embedding/README.md)

# 特性矩阵
- 此矩阵罗列了各bge-large-zh模型支持的特性

| 模型及参数量       | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|--------------|----------------------------|-----------------------------|------|------|-----------------|-----------------|--------|---------|------------|------|-------|----------------|-----|-----|
| bge-large-zh | 支持world size 1             | 支持world size 1              | √    | ×    | √               | ×               | ×      | ×       | ×          | ×    | ×     | ×              | ×   | ×   |

# bge-large-zh模型推理指导

本项目同时支持OM离线版本模型和ATB加速版本模型推理，可根据使用版本选择必要的步骤

<!-- TOC -->
* [README](#readme)
* [特性矩阵](#特性矩阵)
* [bge-large-zh模型推理指导](#bge-large-zh模型推理指导)
  * [模型介绍](#模型介绍)
    * [开源模型地址](#开源模型地址)
    * [路径变量](#路径变量)
  * [推理环境准备](#推理环境准备)
  * [快速上手](#快速上手)
    * [获取本项目源码](#获取本项目源码)
    * [安装依赖](#安装依赖)
    * [获取开源模型权重](#获取开源模型权重)
    * [获取测试数据集](#获取测试数据集)
    * [【加速版本】编译模型仓](#加速版本编译模型仓)
    * [【离线版本】模型格式转换](#离线版本模型格式转换)
      * [开源模型转换为 `ONNX` 格式](#开源模型转换为-onnx-格式)
      * [`ONNX` 格式转换为 `OM` 格式](#onnx-格式转换为-om-格式)
  * [模型推理](#模型推理)
  * [模型推理性能&精度](#模型推理性能精度)
    * [测试方法](#测试方法)
    * [模型推理性能](#模型推理性能)
    * [模型推理精度](#模型推理精度)
    * [300I DUO性能说明](#ascend310p3性能说明)
<!-- TOC -->

## 模型介绍

bge-large-zh是由智源研究院研发的中文版文本表示模型，可将任意文本映射为低维稠密向量，以用于检索、分类、聚类或语义匹配等任务，并可支持为大模型调用外部知识。其中**1.5版本**的相似度分布更加合理

### 开源模型地址

```text
url=https://huggingface.co/BAAI/bge-large-zh-v1.5
commit_id=79e7739b6ab944e86d6171e44d24c997fc1e0116
model_name=bge-large-zh-v1.5
```

### 路径变量

**路径变量解释**

| 变量名            | 含义                                                                                                                            |
|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| `working_dir`  | 加速库及模型库下载后放置的目录                                                                                                               |
| `llm_path`     | 模型仓所在路径<br/>若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`<br/>若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| `script_path`  | 脚本所在路径<br/>`bge-reranker` 系列模型的脚本所在路径为 `${llm_path}/examples/models/bge/large-zh-v1.5`                                        |
| `weight_path`  | 模型权重所在路径                                                                                                                      |
| `dataset_path` | 数据集所在路径<br/>数据集下载路径为 `${llm_path}/tests/modeltest/dataset/full`                                                               |

**推荐目录结构**

```text
{llm_path}
├─ examples
│  └─ models
│     └─ bge
│        └─ large-zh-v1.5: {script_path}
└─ tests
   └─ modeltest
      └─ dataset
         └─ full: {dataset_path}
```

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

    说明：需要先source `ascend-toolkit` 及 `nnal` 环境变量

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
    git clone https://huggingface.co/BAAI/bge-large-zh-v1.5
    mv bge-large-zh-v1.5 ${weight_path}
    ```
- 访问[模型页面](https://huggingface.co/BAAI/bge-large-zh-v1.5/tree/main)获取
  - 下载 `HuggingFace` 原始模型所有文件至 `${weight_path}` 目录

修改 `${script_path}/config_bge.json` 配置文件中模型权重路径参数为权重的实际路径

修改 `${weight_path}/config.json` 配置文件，添加 `"_name_or_path"` 和 `"auto_map"` 两项配置并映射至权重的实际路径

```json
"_name_or_path": "${weight_path}",
"auto_map": {
  "AutoModel": "${weight_path}--modeling_bert.BertModel"
}
```

### 获取测试数据集

```shell
cd ${dataset_path}
mkdir T2Retrieval
cd T2Retrieval
```

下载数据集文件 [corpus、queries](https://huggingface.co/datasets/C-MTEB/T2Retrieval/tree/main/data) 及 [dev](https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels/tree/main/data) 至 `${dataset_path}/T2Retrieval` 目录中

修改 `${script_path}/eval_performance.py` 和 `${script_path}/eval_t2retrieval.py` 脚本文件中的数据集路径为实际路径

### 【加速版本】编译模型仓

可参考[模型仓的README文件](../../../../README.md)

```shell
cd ${llm_path}
bash scripts/build.sh
source output/atb_models/set_env.sh
```

### 【离线版本】模型格式转换

#### 开源模型转换为 `ONNX` 格式

```shell
cd ${script_path}
python bin2onnx.py --model_path ${weight_path}
```

#### `ONNX` 格式转换为 `OM` 格式

在环境上使用[昇腾ATC](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devaids/auxiliarydevtool/atlasatc_16_0001.html)将 `ONNX` 格式转换为 `OM` 格式的离线模型

source相应的CANN环境变量，在 `${script_path}` 目录下运行脚本

```shell
atc --model="${weight_path}/model.onnx" \
    --framework=5 \
    --output="${weight_path}/bge-large-zh" \
    --soc_version=${soc_version} \
    --input_shape="input_ids:-1,-1;attention_mask:-1,-1;token_type_ids:-1,-1" \
    --optypelist_for_implmode="Gelu" \
    --op_select_implmode=high_performance \
    --input_format=ND \
    --precision_mode_v2=${precision_mode} \
    --modify_mixlist="${weight_path}/ops_info.json"
```

- 参数说明
  - bert模型的三个输入依次为`input_ids`、 `attention_mask`、 `token_type_ids`， 按顺序指定模型输入数据的shape。
  - 参照ATC说明文档，设置shape范围时，若设置为 -1，表示此维度可以使用 >=0 的任意取值，该场景下取值上限为 int64 数据类型表达范围，但受限于host和device侧物理内存的大小，用户可以通过增大内存来支持。
  - Gelu算子在不影响精度的情况下开启高性能模式，提升模型性能
  - 所配置的精度模式不同，网络模型精度以及性能有所不同，具体为： \
  精度高低排序：`origin>mixed_float16>fp16` \
  性能优劣排序：`fp16>=mixed_float16>origin` \
  推荐配置: **mixed_float16**
  - `modify_mixlist` 参数为配置混合精度下的黑白灰名单，目的是控制在 `fp16` 精度溢出的算子保持原精度格式，避免其溢出，这里定义了一个将算子写入黑名单的json文件

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
    python eval_t2retrieval.py \
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

> 说明：Atlas 300I Duo (300I DUO) 推理卡为单卡双芯，比较吞吐率时需要 $×2$，下表已按照该方法处理

**离线版本**

NPU环境使用 `OM` 格式模型，GPU环境使用 `ONNX` 格式模型

| 环境  | 芯片型号       | dtype         | batch_size | seq_len | 吞吐率（fps） |
|-----|------------|---------------|------------|---------|----------|
| NPU | 300I DUO   | mixed_float16 | 8          | 100     | 449.22   |
| NPU | 300I DUO   | mixed_float16 | 20         | 512     | 39.40    |
| NPU | 300I DUO   | mixed_float16 | 128        | 512     | 39.63    |
| NPU | 800I A2    | mixed_float16 | 8          | 100     | 451.18   |
| NPU | 800I A2    | mixed_float16 | 20         | 512     | 103.24   |
| NPU | 800I A2    | mixed_float16 | 128        | 512     | 96.01    |
| GPU | NVIDIA A10 | fp32          | 8          | 100     | 149.93   |
| GPU | NVIDIA A10 | fp32          | 20         | 512     | 48.21    |
| GPU | NVIDIA A10 | fp32          | 128        | 512     | 49.38    |
| GPU | NVIDIA L20 | fp32          | 8          | 100     | 384.60   |
| GPU | NVIDIA L20 | fp32          | 20         | 512     | 112.80   |
| GPU | NVIDIA L20 | fp32          | 128        | 512     | 104.37   |

**加速版本**

| 环境  | 芯片型号        | dtype | batch_size | seq_len | 吞吐率（fps） |
|-----|-------------|-------|------------|---------|----------|
| NPU | 800I A2     | fp16  | 8          | 100     | 488.92   |
| NPU | 800I A2     | fp16  | 20         | 512     | 228.49   |
| NPU | 800I A2     | fp16  | 128        | 512     | 233.37   |
| GPU | NVIDIA L40S | fp16  | 8          | 100     | 170.10   |
| GPU | NVIDIA L40S | fp16  | 20         | 512     | 144.97   |
| GPU | NVIDIA L40S | fp16  | 128        | 512     | 166.21   |

### 模型推理精度

> 说明：精度测试采用 `batch_size = 50` 方法测试

| 环境  | 芯片型号        | dtype | ndcg@10（%） | ndcg@1（%） |
|-----|-------------|-------|------------|-----------|
| NPU | 800I A2     | fp16  | 83.68      | 88.81     |
| GPU | Nvidia L40S | fp16  | 83.68      | 88.82     |

### 300I DUO性能说明

在300I DUO上，需要进行一项操作来发挥出算子更好的性能

1. SoftmaxV2使能VectorCore：需要在以下路径的json文件中找到 `SoftmaxV2`

    ```text
    /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend310p/aic-ascend310p-ops-info.json
    ```
    
    加入使能VectorCore
    
    ```json
    "enableVectorCore": {
      "flag": "true"
    }
    ```

2. 并且在以下路径中把已经存在的 `softmax_v2` 改为其它名称，否则使能不生效

    ```shell
    ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/ascend310p
    ```

3. 重新进行ATC转换再进行性能测试即可
