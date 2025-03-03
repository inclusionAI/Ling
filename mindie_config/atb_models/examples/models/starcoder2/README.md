# STARCODER2 README

- [StarCoder2](https://github.com/bigcode-project/starcoder2)是一系列代码生成模型（3B、7B 和 15B），在 [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) 的 600+ 种编程语言和一些自然语言文本（如 Wikipedia、Arxiv 和 GitHub 问题）上进行了训练
- 此代码仓目前支持StarCoder2-7B与StarCoder2-15B

# 支持特性
| 模型及参数量    | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅300I DUO支持） | MOE | MindIE Service | TGI | 长序列|
|---------------|----------------------------|-----------------------------|------|---------------------|-----------------|-----------------|---------|-----------|--------------|------------------------|-----|--------|-----|----|
| StarCoder2-7B | 支持world size 1,2,4,8        | ×                        | ×   | √                   | ×             | √              | ×       | ×        | ×           | ×                      | ×  | ×     | ×  |×|
| StarCoder2-15B | 支持world size 4             | ×                        | √   | ×                   | ×             | √              | √       | ×        | ×           | ×                      | ×  | √     | ×  |×|

# 使用说明

## 路径变量解释
| 变量名         | 含义                                            |
|---------------|-------------------------------------------------|
| `working_dir` | 加速库及模型库下载后放置的目录                       |
| `llm_path`    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| `script_path` | 脚本所在路径；StarCoder2的工作脚本所在路径为`${llm_path}/examples/models/starcoder2` |
| `weight_path` | 模型权重路径                                      |


## 权重
### 权重下载
- 下载starcoder2模型权重，放置到`${weight_path}`下
  - [StarCoder2-15B](https://huggingface.co/bigcode/starcoder2-15b/tree/main)
  - [StarCoder2-7B](https://huggingface.co/bigcode/starcoder2-7b/tree/main)

### 权重转换
- 当前仅支持加载safetensor格式的权重文件，若权重文件为bin格式，请参考[此README文件](../../README.md)

### 量化权重生成（W8A8）
- 当前仅StarCoder2-15B支持W8A8量化
- 到`${script_path}`路径下，运行`convert_w8a8_quant_weights.py`（`transformers`版本需求：>=4.39.0）
```shell
cd ${script_path}
python convert_w8a8_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径}
```
- 权重生成后确认模型配置文件，确认`${weight_path}/config.json`文件中的`torch_dtype`和`quantize`
  - `torch_dtype`和`quantize`类型用于标识量化类型和精度
    | 量化类型及精度  | torch_dtype | quantize |
    |----------------|-------------|----------|
    | FP16           | "float16"   | ""       |
    | BF16           | "bfloat16"  | ""       |
    | W8A8           | "float16"   | "w8a8"   |
    | W8A16          | "float16"   | "w8a16"  |
  - 示例
    - starcoder2模型使用FP16精度，W8A8量化
      ```json
      {
        "torch_dtype": "float16",
        "quantize": "w8a8",
      }
      ```
- 若要测试HumanEval量化精度并符合与浮点精度保持1%差距，可配置中`convert_w8a8_quant_weights.py`的回退层`disable_names`
```python
disable_names = [
    "model.layers.0.mlp.c_proj",
    "model.layers.1.mlp.c_proj",
    "model.layers.2.mlp.c_proj",
    "model.layers.3.mlp.c_proj",
    "model.layers.4.mlp.c_proj",
    "model.layers.5.mlp.c_proj",
    "model.layers.6.mlp.c_proj",
    "model.layers.7.mlp.c_proj",
    "model.layers.8.mlp.c_proj",
    "model.layers.9.mlp.c_proj",
    "model.layers.10.mlp.c_proj",
    "model.layers.11.mlp.c_proj",
    "model.layers.12.mlp.c_proj",
    "model.layers.13.mlp.c_proj",
    "model.layers.14.mlp.c_proj",
    "model.layers.15.mlp.c_proj",
    "model.layers.16.mlp.c_proj",
    "model.layers.17.mlp.c_proj",
    "model.layers.18.mlp.c_proj",
    "model.layers.19.mlp.c_proj",
    "model.layers.20.mlp.c_proj",
    "model.layers.21.mlp.c_proj",
    "model.layers.22.mlp.c_proj",
    "model.layers.23.mlp.c_proj",
    "model.layers.24.mlp.c_proj",
    "model.layers.25.mlp.c_proj",
    "model.layers.26.mlp.c_proj",
    "model.layers.27.mlp.c_proj",
    "model.layers.28.mlp.c_proj",
    "model.layers.29.mlp.c_proj",
    "model.layers.30.mlp.c_proj",
    "model.layers.31.mlp.c_proj",
    "model.layers.32.mlp.c_proj",
    "model.layers.33.mlp.c_proj",
    "model.layers.34.mlp.c_proj",
    "model.layers.35.mlp.c_proj",
    "model.layers.36.mlp.c_proj",
    "model.layers.37.mlp.c_proj",
    "model.layers.38.mlp.c_proj",
    "model.layers.39.mlp.c_proj",
]
```

## 800I A2 运行操作说明

### 对话测试
**运行Paged Attention FP16**
- 运行启动脚本
  - 在`${llm_path}`目录下执行以下指令
    ```shell
    bash ${script_path}/run_800i_a2_pa.sh ${weight_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ``` shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export LCCL_ENABLE_FALLBACK=1
    ```

**运行W8A8量化**
- 获取量化权重后操作步骤同上

### 对话测试脚本参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- 所有参数可见run_pa.py文件中

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
- 示例
  ```shell
  cd ${llm_path}/tests/modeltest
  bash run.sh pa_fp16 full_HumanEval 1 starcoder2 ${weight_path} 4
  ```
- 运行量化权重时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
- 示例
  ```shell
  cd ${llm_path}/tests/modeltest
  bash run.sh pa_fp16 performance [[256,256],[512,512],[1024,1024],[2048,2048]] 1 starcoder2 ${weight_path} 4
  ```
- 运行量化权重时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)