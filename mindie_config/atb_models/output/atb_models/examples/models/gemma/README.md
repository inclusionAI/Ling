# README

- [Gemma](https://github.com/google/gemma_pytorch)，是由 Google 推出的一系列轻量级最先进的开放模型，采用与Gemini模型相同的研究和技术构建。Gemma模型非常适合各种文本生成任务，包括问答、摘要和推理。

- 此代码仓中实现了一套基于NPU硬件的Gemma推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了Gemma模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI |  长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|-----|-----|
| Gemma-2B    | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | 是   | 是                   | 否              | 是              | 否       | 否        | 否           | 否                       | 否  | 否     | 否  |  否  |
| Gemma-7B   | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | 是   | 是                  | 否              | 是              | 是       | 否        | 否           | 否                       | 否  | 否     | 否  |  否  |

- 此模型仓已适配的模型版本
  - [Gemma系列](https://github.com/google/gemma_pytorch)
  

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；gemma的工作脚本所在路径为`${llm_path}/examples/models/gemma`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**
- [Gemma-2B](https://huggingface.co/google/gemma-2b/tree/main)
- [Gemma-7B](https://huggingface.co/google/gemma-7b/tree/main)


**权重转换**
- 参考[此README文件](../../README.md)

**量化权重生成**
- 基于原始的FP16的权重，生成量化权重
- W8A8 Antioutlier量化权重请使用以下指令生成
  - 当前Gemma-7B支持W8A8 Antioulier量化
  - 设置环境变量
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
  在\${llm_path}目录下执行以下指令
  - 执行量化脚本  （也可以指定自己的数据集,修改shell脚本里的路径）
  ```shell
  bash examples/models/gemma/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type gemma_7b_w8a8
  ```


**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试

**运行Paged Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
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

**运行Paged Attention BF16**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行BF16时需修改${weight_path}/config.json中的`torch_dtype`字段，将此字段对应的值修改为`bfloat16`
- 300I DUO卡暂不支持BF16特性

**运行Paged Attention W8A8**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为W8A8量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改W8A8量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8`
  - 若config.json中无此字段，则新增



## 精度测试

- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_fp16 full_BoolQ 1 gemma ${gemma-2b权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 gemma ${gemma-7b权重路径} 8
    ```
- 运行量化权重和BF16时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)

## 性能测试

- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 gemma ${gemma-2b权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 gemma ${gemma-7b权重路径} 8
    
    ```
- 运行量化权重和BF16时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)


## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)
- gemma-7b的开源权重有个bug，在运行时需要改一下源码：..../transformers/models/gemma/modeling_gemma.py 280行
```
    #attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = attn_output.reshape(bsz, q_len, 4096)
```
- gemma-2b/gemma-7b 推荐使用`transformers >= 4.38.0`, block_size 需要设置为64