# README

- [Wizardcoder系列模型](https://huggingface.co/WizardLMTeam/WizardCoder-Python-34B-V1.0/tree/main) WizardCoder是由WizardLM团队推出了一个新的指令微调代码大模型，打破了闭源模型的垄断地位，超越了闭源大模型Anthropic Claude和谷歌的Bard。WizardCoder大幅度地提升了开源莫i选哪个的SOTA水平，创造了惊人的进步，提高了22.3%的性能，成为了开源领域的新时代引领者。

- 此代码仓中实现了一套基于NPU硬件的WizardCoder推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各CodeLlama模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|---------|--------------|----------|--------|--------|-----|-----|
| WizardCoder-34B   | 支持world size 4,8   | 支持world size 4,8     | 是   | 是   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  | 否 |

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | ATB_Models模型仓所在路径；若使用编译好的包，则路径为`${working_dir}/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models/`    |
| script_path | 脚本所在路径；WizardCoder的工作脚本所在路径为`${llm_path}/examples/models/wizard`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**
- [WizardCoder-34B](https://huggingface.co/WizardLMTeam/WizardCoder-Python-34B-V1.0/tree/main)

**权重转换**
> 若权重中不包含safetensors格式，则执行权重转换步骤，否则跳过
- 参考[此README文件](../../README.md)
- 将bin权重转为safetensors格式


**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试

**运行Paged Attention BF16**
- 运行启动脚本
  - 将`${llm_path}`加入`PYTHONPATH`搜索目录
    ```shell
    export PYTHONPATH=${llm_path}:${PYTHONPATH}
    ```
  - 在${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path}
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
- 300I DUO卡不支持BF16数据类型

**运行Paged Attention FP16**
- 运行启动脚本
  - 与“运行Paged Attention BF16”的启动方式相同
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 相比于BF16，运行FP16时需修改${weight_path}/config.json中的`torch_dtype`字段，将此字段对应的值修改为`float16`


## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 运行Paged Attention BF16
    bash run.sh pa_bf16 full_HumanEval 1 wizard ${weight_path} 8
    # 运行Paged Attention FP16
    bash run.sh pa_fp16 full_HumanEval 1 wizard ${weight_path} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 运行Paged Attention BF16
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 wizard ${weight_path} 8
    # 运行Paged Attention FP16
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 wizard ${weight_path} 8
    ```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)