# README

- [Code Llama](https://github.com/Meta-Llama/codellama) 是Meta发布的代码生成类大语言模型，在编程任务上具备填充、0-shot指令跟随能力，并支持长序列文本输入，在开源模型中拥有先进的性能。Code Llama 是 Llama 2 的代码专用版本，它是通过在代码数据集上对 Llama 2 进行进一步训练，并在同一数据集上长时间采样更多数据而创建的。从本质上讲，Code Llama 具有更强的编码能力。它可以根据代码和自然语言提示（例如，"给我写一个输出斐波那契数列的函数"）生成代码和有关代码的自然语言。它还可用于代码补全和调试。它支持许多当今最流行的编程语言，包括 Python、C++、Java、PHP、Typescript (Javascript)、C#、Bash 等。

- 此代码仓中实现了一套基于NPU硬件的Code Llama推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各CodeLlama模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI |  长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|-----|-----|
| CodeLlama-7B  | 支持world size 1,2,4,8   | 否      | 是   | 否   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  | 否  |
| CodeLlama-13B  | 支持world size 1,2,4,8   | 否      | 是   | 是   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 是     | 否  | 否  |
| CodeLlama-34B  | 支持world size 4,8   | 支持world size 4,8      | 是   | 是   | 否              | 是              | 是       | 否       | 否           | 是       | 否     | 是     | 否  | 否  |
| CodeLlama-70B  | 支持world size 8   | 否      | 是   | 是   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  | 否  |

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | ATB_Models模型仓所在路径；若使用编译好的包，则路径为`${working_dir}/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models/`    |
| script_path | 脚本所在路径；CodeLlama的工作脚本所在路径为`${llm_path}/examples/models/codellama`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**
- [CodeLlama-7B](https://huggingface.co/codellama/CodeLlama-7b-hf/tree/main)
- [CodeLlama-13B](https://huggingface.co/codellama/CodeLlama-13b-hf/tree/main)
- [CodeLlama-34B](https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main)
- [CodeLlama-70B](https://huggingface.co/codellama/CodeLlama-70b-hf/tree/main)

**权重转换**
> 若权重中不包含safetensors格式，则执行权重转换步骤，否则跳过
- 参考[此README文件](../../README.md)

**量化权重生成**
> 基于原始的浮点权重，生成量化权重

- 设置环境变量
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  # 推荐使用transformers 4.33.0版本进行量化权重转换，执行模型推理时transformers的版本大于等于4.33.0
  pip uninstall transformers -y
  pip install transformers=={指定版本}
  # NPU多卡量化时关闭虚拟内存
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  # 指定当前机器上可用的逻辑NPU核心
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  # 将`${llm_path}`加入`PYTHONPATH`搜索目录
  export PYTHONPATH=${llm_path}:${PYTHONPATH}
  ```
- W8A8量化权重请使用以下指令生成
  - Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
  - Step 2 W8A8量化权重生成
    ```shell
    cd ${llm_path}/examples/models/codellama
    bash generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type codellama_34b_w8a8
    ```
  > NPU多卡量化注意事项和环境要求见[此README中的【NPU多卡量化】章节](../../README.md)

- 稀疏量化权重请使用以下指令生成
  > 执行权重生成步骤前，建议预留90G以上的磁盘空间和350G以上的系统内存
  >
  > 稀疏量化方式生成的权重只支持在300I DUO硬件上推理
  >
  > 虚拟机场景时，建议配置32个以上的cpu核心（若出现命令执行超时，则继续增加cpu核数）
  - Step 1
    - 修改模型权重config.json中`torch_dtype`字段为`float16`
    - 下载msmodelslim量化工具
    - 下载地址为https://gitee.com/ascend/msit/tree/master/msmodelslim
    - 根据msmodelslim量化工具readme进行相关操作
  - Step 2 稀疏量化权重生成
    ```shell
    cd ${llm_path}/examples/models/codellama
    bash generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8S量化权重路径} -type codellama_34b_w8a8s
    ```
  - Step 3：量化权重切分及压缩
    ```shell
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
    ```
    > TP数为tensor parallel并行个数
    > 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行

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

**运行Paged Attention W8A8**
- W8A8量化权重生成
- 运行启动脚本
  - 与“运行Paged Attention BF16”的启动方式相同
  - `${weight_path}`为W8A8量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 相比于BF16，运行量化时需修改W8A8量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8`
  - 若config.json中无此字段，则新增

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 运行Paged Attention BF16
    bash run.sh pa_bf16 full_HumanEval 1 codellama ${weight_path} 8
    # 运行Paged Attention FP16
    bash run.sh pa_fp16 full_HumanEval 1 codellama ${weight_path} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 运行Paged Attention BF16
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 codellama ${weight_path} 8
    # 运行Paged Attention FP16
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 codellama ${weight_path} 8
    ```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)