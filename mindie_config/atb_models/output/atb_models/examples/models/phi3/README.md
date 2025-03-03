# Phi-3 模型推理指导

# 概述

- [Phi-3](https://github.com/microsoft/Phi-3CookBook) 是 Microsoft 开发的一系列开放式 AI 模型。Phi-3 模型是一个功能强大、成本效益高的小语言模型 (SLM)，在各种语言、推理、编码和数学基准测试中，在同级别参数模型中性能表现优秀。为开发者构建生成式人工智能应用程序时提供了更多实用的选择。
- 此代码仓中实现了一套基于NPU硬件的 Phi-3 推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了 Phi-3 模型支持的特性

| 模型及参数量    | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI  | 长序列 |
| --------------- | -------------------------- | --------------------------- | ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | ------ | ---- | ------ |
| Phi-3-mini-128k | 支持world size 1,2,4,8           | 支持world size 1,2,4,8            | 是   | 否   | 否              | 是              | 否       | 否             | 否           | 否       | 否      | 否     | 否   | 否     |


# 使用说明

- 执行推理前需要将权重目录下的config.json中的`torch_dtype`改为`"float16"`
- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | ATB_Models模型仓所在路径；若使用编译好的包，则路径为`${working_dir}/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models/`    |
| script_path | 脚本所在路径; Phi-3系列模型的工作脚本所在路径为`${llm_path}/examples/models/phi3`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**
- [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/tree/bb5bf1e4001277a606e11debca0ef80323e5f824) 模型仓近期更新，需要下载 commit id 为 bb5bf1e4001277a606e11debca0ef80323e5f824 的权重（建议直接在 huggingface 先切换 commit id，再下载）。

## 推理

### 对话测试
**运行Page Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} -trust_remote_code
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0`
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
    export HCCL_BUFFSIZE=120
    export HCCL_WHITELIST_DISABLE=1
    export ATB_CONTEXT_WORKSPACE_RING=1
    export ATB_CONTEXT_WORKSPACE_SIZE=2629145600
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
    export ATB_LAUNCH_KERNEL_WITH_TILING=0
    export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1
    export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=0

    ```

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
