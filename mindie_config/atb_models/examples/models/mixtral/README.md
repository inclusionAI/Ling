# README

- [Mixtral 8x7B]是一个稀疏混合专家(SMoE)语言模型。对于每个 token 输入，在每一层，路由网络(router network)在 8 个专家中选择 2 个专家进行计算。所以每次运行时，模型只会激活 47B 参数中的 13B。在不损失精度的前提下，提高了模型的性能。[Mixtral 8x22B]的总参数量为 141B，激活参数量为 39B。（当前脚本支持 8x7B-Base，8x7B-Chat，8x22B-Base，和 8x22B-Chat）
- 此代码仓中实现了一套基于 NPU 硬件的 Mixtral-MoE 模型。配合加速库使用，旨在 NPU 上获得极致的推理性能。

# 特性矩阵
- 矩此阵罗列了各Mixtral模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8动态量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MindIE Service | TGI | 长序列   |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|---------|----------|---------------|--------------------------|--------|-----|
| Mixtral-8x7B-Instruct   | 支持world size 8     | 支持world size 4,8  | √   | ×                   | √              | √              | √   | ×          | ×           | ×                       | ×  | ×     | ×  | ×    |
| Mixtral-8x22B-Instruct   | 支持world size 8     | ×                | √   | ×                   | √              | √              | ×  | ×         | ×           | ×                       | ×     | ×  | ×    |

- 注意：模型目前仅支持W8A8动态量化，不支持其它W8A8量化方法。

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                           |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用 gitee 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；Mixtral-MoE 的工作脚本所在路径为`${llm_path}/examples/models/mixtral`                                                                      |
| weight_path | 模型权重路径                                                                                                                                             |

## 权重

**权重下载**

- [Mixtral-8x7B-Base](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main)
- [Mixtral-8x7B-Chat/Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main)
- [Mixtral-8x22B-Base](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main)
- [Mixtral-8x22B-Chat/Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/tree/main)

**量化权重生成**

- W8A8 Dynamic量化权重请使用以下指令生成

  - 执行量化脚本
    ```
    # 指定当前机器上可用的逻辑NPU核心
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    cd ${llm_path}
    python examples/models/mixtral/convert_quant_weights.py \
    --model_path {浮点权重路径} \
    --save_directory {W8A8量化权重路径}
    (--trust_remote_code{是否外部下载})
    ```
    - 注意：`model_path`和`save_directory`请勿使用同一个文件夹，避免浮点权重和量化权重混淆

- 从浮点权重路径下复制一下文件到量化权重路径
    - config.json
    - gitattributes
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer.model
    - tokenizer_config.json


- 修改量化权重的 config.json 文件 加入quantize
    ```
    "quantize": "w8a8_dynamic"
    ```

**基础环境变量**

- 参考[此 README 文件](../../../README.md)

## 推理

### 对话测试

**运行 Paged Attention FP16**

- 运行启动脚本（Mixtral_8x7B transformers 版本需求：4.36.0.dev0 | Mixtral_8x22B transformers 版本需求：4.39.0）
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh ${weight_path} -trust_remote_code
    ```
  - trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。
- 启动脚本中可设置自定义问题，具体在 input_text 后面修改即可 (默认问题为"Who is the CEO of Google?")
- 启动脚本中可设置自定义输出长度，具体在 max_output_length 后面修改即可（默认长度为 10）
- 若当前所用权重版本为"chat"版本，请将"--is_chat_model"赋值给 extra_param；若当前所用权重版本为"base"版本，可以将空字符串赋值给 extra_param（默认为 chat_model）
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑 NPU 核心，多个核心间使用逗号相连
    - 核心 ID 查阅方式见[此 README 文件](../../README.md)的【启动脚本相关环境变量】章节
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
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=1
    export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
    ```

## 精度测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例 
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_fp16 full_BoolQ 1 mixtral ${mixtral-8x7b-base权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 mixtral ${mixtral-8x7b-instruct权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 mixtral ${mixtral-8x22b-base权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 1 mixtral ${mixtral-8x22b-instruct权重路径} 8
    ```

## 性能测试

- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 mixtral ${mixtral-8x7b-base权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 mixtral ${mixtral-8x7b-instruct权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 mixtral ${mixtral-8x22b-base权重路径} 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 mixtral ${mixtral-8x22b-instruct权重路径} 8
    ```

## FAQ

- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；这个文件的参数说明见[此 README 文件](../../README.md)