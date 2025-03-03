# ChatGLM2-6B 模型推理指导 <!-- omit in toc -->

# 概述

- [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B/) 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，ChatGLM2-6B有更强大的性能、更长的上下文、更高效的推理和更开放的协议。
- 此代码仓中实现了一套基于NPU硬件的ChatGLM2推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了ChatGLM2-6B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化（仅300I DUO支持） | MOE | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|-----|-----|
| ChatGLM2-6B    | 支持world size 1,2,4,8  | 支持world size 1,2,4    | 是   | 否   | 否              | 是              | 是       | 否         | 否           | 是       | 否     | 是     | 是  | 否 |

- 此模型仓已适配的模型版本
  - [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b/tree/main)

# 使用说明
- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；路径为${llm_path}/examples/models/chatglm/v2_6b                            |
| weight_path | 模型权重路径                            |

## 版本依赖
ChatGLM2-6B模型需使用[默认依赖](../../../../requirements/requirements.txt)版本

## 权重转换
- 参考[此README文件](../../../README.md)

## 量化权重导出
量化权重可通过msmodelslim（昇腾压缩加速工具）实现。

### 环境准备
环境配置可参考msmodelslim官网：https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0002.html

### 导出w8a8量化权重
通过`${llm_path}/examples/models/chatglm/v2_6b/quant_chatglm_w8a8.sh`文件导出模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：
```shell
# 一定要设置该线程数
export OMP_NUM_THREADS=48
bash quant_chatglm_w8a8.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -trust_remote_code
```

导出量化权重后应生成`quant_model_weight_w8a8.safetensors`和`quant_model_description_w8a8.json`两个文件。

注：

1.quant_chatglm_w8a8.sh文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行脚本生成量化权重时，会在生成的权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，当前仅支持`w8a8`。

3.执行完以上步骤后，执行量化模型只需要替换权重路径。

4.如果生成权重时遇到`OpenBLAS Warning: Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP = 1 option`，可通过设置`export OMP_NUM_THREADS=1`来关闭多线程规避。

### 导出稀疏量化权重
> 运行前需要确保压缩工具编译过
>
> `cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/msmodelslim/pytorch/weight_compression/compress_graph`
>
> `bash build.sh /usr/local/Ascend/ascend-toolkit/latest`

执行generate_sparse.sh导出稀疏量化权重（注意量化权重不要和浮点权重放在同一个目录下）：
```shell
# 设置CANN包的环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 导出稀疏量化权重
bash generate_sparse.sh ${浮点权重路径} ${稀疏量化权重保存路径} ${llm_path}/examples/models/chatglm/v2_6b/calib_data.jsonl ${Tensor并行数} -trust_remote_code
```

执行后`${稀疏量化权重保存路径}`下会生成compress目录，使用`${稀疏量化权重保存路径}/compress`目录作为权重目录进行推理。

注：

1.generate_sparse.sh文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行完以上步骤后，执行量化模型只需要替换权重路径为`${稀疏量化权重保存路径}/compress`。

3.当在npu上生成稀疏量化权重（即`--device_type`为`npu`时）时，注意需要将`${浮点权重路径}/modeling_chatglm.py`文件168行的`@torch.jit.script`注释。

## 300I DUO 运行操作说明
- 可开启CPU Performance模式以提高模型推理性能

  ```
  cpupower frequency-set -g performance
  ```

### 对话测试
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_300i_duo_pa.sh ${weight_path} -trust_remote_code
    ```
- 环境变量说明
  - `export BIND_CPU=1`
    - 绑定CPU核心开关
    - 默认进行绑核
    - 若当前机器未设置NUMA或绑核失败，可将 BIND_CPU 设为 0
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=2`
    - 指定模型运行时的TP数，即world size
    - 默认为单卡双芯
    - 各模型支持的TP数参考“特性矩阵”
    - “单卡双芯”运行请指定`TP_WORLD_SIZE`为`2`
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径
  - - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    # 内存
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export HCCL_BUFFSIZE=110
    ```


## 800I A2 运行操作说明
- 可开启CPU Performance模式以提高模型推理性能

  ```
  cpupower frequency-set -g performance
  ```
### 对话测试

**运行Paged Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_800i_a2_pa.sh ${weight_path} -trust_remote_code
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../../README.md)的【启动脚本相关环境变量】章节
  - `export TP_WORLD_SIZE=1`
    - 指定模型运行时的TP数，即world size
    - 默认为单卡
    - 各模型支持的TP数参考“特性矩阵”
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径
  - `export IS_BF16=false`
    - 是否使用BF16精度进行推理
    - 默认使用FP16
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ```shell
    # 内存
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    # 性能
    export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    ```


**运行Paged Attention BF16**
- 暂不支持

**运行Paged Attention W8A8量化**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为W8A8量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改W8A8量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8`
  - 若config.json中无此字段，则新增

**运行KV cache量化**
- 暂不支持

**运行Paged Attention 稀疏量化**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为稀疏量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改稀疏量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8sc`
  - 若config.json中无此字段，则新增
- 注意：压缩算法与硬件强相关，当前仅300I DUO卡支持稀疏量化


## 精度测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## FAQ
- `import torch_npu`遇到`xxx/libgomp.so.1: cannot allocate memory in static TLS block`报错，可通过配置`LD_PRELOAD`解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`