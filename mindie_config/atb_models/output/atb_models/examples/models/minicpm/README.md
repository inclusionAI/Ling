# README

- [MiniCpm 2B]模型是一个语言模型。主体语言模型 MiniCPM-2B 仅有 24亿（2.4B）的非词嵌入参数量,基于MiniCPM-2B的多模态模型MiniCPM-V，能力超越基于Phi-2的同参数级别多模态模型
- [MiniCpm 1B]是面壁与清华大学自然语言处理实验室共同开源的系列端侧语言大模型，主体语言模型 仅有 12亿（1.2B）的非词嵌入参数量
- 此代码仓中实现了一套基于 NPU 硬件的 MiniCpm 模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

# 特性矩阵
- 矩此阵罗列了各MiniCpm模型支持的特性

| 模型及参数量              | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | W4A16量化 | KV cache量化 | 稀疏量化 | MindIE | TGI | 长序列   |
|---------------------|----------------------------|----------------------------|------|----------------------|-----------------|-----------------|---------|---------|----------|---------------|--------------------------|--------|-----|-------|
| MiniCPM-1B-sft-bf16 | 支持world size 1             | √                | √   | ×                   | √              | √              | ×   | ×    | ×        | ×           | ×                       | ×  | ×     | ×  | ×    |
| MiniCPM-2B-sft-bf16 | 支持world size 1             | ×               | √   | ×                   | √              | √              | ×  | ×    | ×        | ×           | ×                       | ×     | ×  | ×    |

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                    |
| ----------- |-----------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                       |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用 gitee 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；MiniCpm 的工作脚本所在路径为`${llm_path}/examples/models/minicpm`                                                        |
| weight_path | 模型权重路径                                                                                                                |

## 权重

**权重下载**

- [MiniCPM-1B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-1B-sft-bf16)
- [MiniCPM-2B-sft-bf16](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)


**基础环境变量**

- 参考[此 README 文件](../../../../../../MindIE-LLM%20-%20副本/examples/atb_models/README.md)

## 推理

### 对话测试

**运行 Paged Attention FP16**

- 运行启动脚本（MiniCpm_1B transformers 版本需求：4.36.0.dev0 | MiniCpm_8x22B transformers 版本需求：4.39.0）
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path}
    ```
- 启动脚本中可设置自定义问题，具体在 input_text 后面修改即可 (默认问题为"Who is the CEO of Google?")
- 启动脚本中可设置自定义输出长度，具体在 max_output_length 后面修改即可（默认长度为 10）
- 若当前所用权重版本为"chat"版本，请将"--is_chat_model"赋值给 extra_param；若当前所用权重版本为"base"版本，可以将空字符串赋值给 extra_param（默认为 chat_model）
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../../../../../MindIE-LLM%20-%20副本/examples/atb_models/examples/README.md)的【启动脚本相关环境变量】章节
    - 对于 300I DUO 卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用 20030 端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export LCCL_ENABLE_FALLBACK=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    ```

## 精度测试

- 参考[此 README 文件](../../../../../../MindIE-LLM%20-%20副本/examples/atb_models/tests/modeltest/README.md)
  - 示例 
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0
    export MAX_MEMORY_GB=29
    bash run.sh pa_fp16 full_BoolQ 1 minicpm ${minicpm权重路径} 1
    bash run.sh pa_fp16 full_CEval 5 1 minicpm ${minicpm权重路径} 1
    bash run.sh pa_fp16 full_GSM8K 1 minicpm ${minicpm权重路径} 1
    ```

## 性能测试

- 参考[此 README 文件](../../../../../../MindIE-LLM%20-%20副本/examples/atb_models/tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0
    export MAX_MEMORY_GB=29
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 minicpm ${minicpm权重路径} 8
    ```

## FAQ

- 更多环境变量见[此 README 文件](../../../../../../MindIE-LLM%20-%20副本/examples/atb_models/examples/README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；这个文件的参数说明见[此 README 文件](../../../../../../MindIE-LLM%20-%20副本/examples/atb_models/examples/README.md)
- 运行时，需要通过指令 pip list ｜ grep protobuf 确认 protobuf 版本，如果版本高于 3.20.x，请运行指令 pip install protobuf==3.20.0 进行更新
