# README

- [DeepSeek-LLM](https://github.com/deepseek-ai/deepseek-LLM)从包含2T token的中英文混合数据集中，训练得到7B Base、7B Chat、67B Base与67B Chat四种模型

# 支持特性
| 模型及参数量       | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅300I DUO支持） | MOE | MindIE Service | TGI |长序列|
|------------------|----------------------------|-----------------------------|------|---------------------|-----------------|-----------------|---------|-----------|--------------|------------------------|-----|--------|-----|-----|
| DeepSeek-LLM-7B  | 支持world size 1,2,4,8        | 支持world size 1,2,4,8        | √   | ×                   | ×              | √               | ×       | ×        | ×           | ×                      | ×  | ×     | ×  |×  |
| DeepSeek-LLM-67B | 支持world size 8            | ×                          | √    | ×                   | ×              | √               | ×       | ×        | ×           | ×                      | ×  | ×     | ×  |×  |


# 使用说明

## 路径变量解释

| 变量名         | 含义                             |
| --------------| --------------------------------|
| `working_dir` | 加速库及模型库下载后放置的目录       |
| `llm_path`    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用 gitee 下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| `script_path` | 脚本所在路径；Deepseek-LLM的工作脚本所在路径为`${llm_path}/examples/models/deepseek` |
| `weight_path` | 模型权重路径                      |

## 权重

### 权重下载
- [Deepseek-LLM-7B-Base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base/tree/main)
- [Deepseek-LLM-7B-Chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat/tree/main)
- [Deepseek-LLM-67B-Base](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/tree/main)
- [Deepseek-LLM-67B-Chat](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/tree/main)

### 权重转换
- 当前仅支持加载safetensor格式的权重文件，若权重文件为bin格式，请参考[此README文件](../../README.md)


## 基础环境变量
- 参考[此 README 文件](../../../README.md)

## 推理

### 对话测试

**运行 Paged Attention FP16**
- 运行启动脚本（`transformers` 版本需求：>=4.35.0）
  - 在`${llm_path}`目录下执行以下指令
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
    ```
- 该系列的chat模型暂不支持默认对话模板，其权重目录中的`tokenizer_config.json`文件里的必须包含有效的`chat_template`

## 精度测试
- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_fp16 full_BoolQ 1 deepseek_llm ${deepseek-llm-7b-base权重路径} 2
    bash run.sh pa_fp16 full_BoolQ 1 deepseek_llm ${deepseek-llm-67b-base权重路径} 8
    ```

## 性能测试
- 参考[此 README 文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 deepseek_llm ${deepseek-llm-7b-base权重路径} 2
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 deepseek_llm ${deepseek-llm-67b-base权重路径} 8
    ```

## FAQ
- 更多环境变量见[此 README 文件](../../README.md)
- 对话测试实际执行的 Python 文件为`${llm_path}/examples/run_pa.py`；这个文件的参数说明见[此 README 文件](../../README.md)