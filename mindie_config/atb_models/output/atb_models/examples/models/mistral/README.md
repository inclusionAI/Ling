# README

- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 为 Mistral 7B v0.2 Base Model 的指令调优版本。该模型在2023年9月首次发布，在多个基准测试中表现优异，被评价为同尺寸级别中最优秀的模型之一。

- 此代码仓中实现了一套基于NPU硬件的Mistral推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了Mistral-7B-Instruct-v0.2模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE | MindIE Service | TGI | 长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|-----|-----|
| Mistral-7B-Instruct-v0.2 | 支持world size 1,2,4,8     | 支持world size 1,2,4     | 是   | 否                  | 否         | 是              | 否   | 否            | 否           | 否                       | 否  | 否 | 否   | 否 |

- 此模型仓已适配的模型版本
  - Mistral-7B-Instruct-v0.2 (transformers==4.36.0)

# 使用说明
- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。


## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；Mistral的工作脚本所在路径为`${llm_path}/examples/models/mistral`               |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**

- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main)

**权重转换**
- 参考[此README文件](../../README.md)

**量化权重生成**

- 基于原始的BF16的权重，生成量化权重

- W8A8 Antioutlier量化权重请使用以下指令生成

  - 执行量化脚本
  
    ```
    bash generate_quant_weight.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -trust_remote_code
    ```
  
    - 注意：`src`和`dst`请勿使用同一个文件夹，避免浮点权重和量化权重混淆
  
  - 修改量化权重的 config.json 文件
  
    ```
    torch_dtype:float16
    ```

**基础环境变量**

- 参考[此README文件](../../../README.md)

## 推理

### 对话测试
**运行Paged Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} -trust_remote_code
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_pa.py`；该文件的参数说明见[此README文件](../../README.md)