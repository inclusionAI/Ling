# STARCODER README

StarCoder模型是在The Stack (v1.2)的80+种编程语言上训练的15.5B参数模型，不包括选择退出请求。该模型使用多查询注意力，一个包含8192个令牌的上下文窗口，并在1万亿个令牌上使用填充中间目标进行训练。

- 参考实现：
```
https://huggingface.co/bigcode/starcoder
```
# 特性矩阵
- 此矩阵罗列了各starcoder模型支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅300I DUO支持） | MOE | MindIE Service | TGI |长序列|
|-------------|----------------------------|-----------------------------|------|------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|---|---|
| starcoder-15.5B   | 支持world size 8     | 支持world size 4            | √    | ×                | ×               | √               | √       | ×        | ×           | ×                       | ×  | √      | √ |×|

# 使用说明

## 权重下载
- 下载starcoder模型权重，放置到自定义路径下
```
https://huggingface.co/bigcode/starcoder/tree/main
```
- 修改`config.json`中的`model_type`为`starcoder`

## 权重转换
- 参考[此README文件](../../README.md)


## 量化权重转换（W8A8）
- 去目标文件目录下执行
```
python convert_w8a8_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径}
```
- 若要测试HumanEval量化精度并符合与浮点精度保持1%差距，可配置`convert_w8a8_quant_weights.py`中回退层`disable_names`（例如，将`disable_names`中的注释取消）


- 配置
  | 量化类型及精度  | torch_dtype | quantize |
  |----------------|-------------|----------|
  | FP16           | "float16"   | ""       |
  | BF16           | "bfloat16"  | ""       |
  | W8A8           | "float16"   | "w8a8"   |
  | W8A16          | "float16"   | "w8a16"  |

- 示例
  - starcoder模型使用FP16精度，W8A8量化
    ```json
    {
      "torch_dtype": "float16",
      "quantize": "w8a8",
    }
    ```

## 路径变量解释
| 变量名  | 含义                                                                                                                  |
|--------|---------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                     |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；starcoder工作脚本所在路径为`${llm_path}/examples/models/starcoder`                                                 |
| weight_path | 模型权重路径   

## 300I DUO 运行操作说明

### 对话测试
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_300i_duo.sh ${weight_path}
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改
    ``` shell
    export ATB_LAUNCH_KERNEL_WITH_TILING=1
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export LCCL_ENABLE_FALLBACK=1
    ```

### 对话测试脚本参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- 所有参数可见run_pa.py文件中

## 800I A2 运行操作说明

### 对话测试
**运行Flash Attention FP16**
- 暂不支持

**运行Flash Attention BF16**
- 暂不支持

**运行Paged Attention FP16**
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
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

### 对话测试脚本参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- 所有参数可见run_pa.py文件中

**运行Paged Attention BF16**    
- 暂不支持

**运行W8A8量化**
- 获取量化权重后操作步骤同上

**运行KV cache量化**
- 暂不支持

**运行稀疏量化**
- 暂不支持


## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
- 示例
  ```shell
  cd ${llm_path}/tests/modeltest
  bash run.sh pa_fp16 full_HumanEval 1 starcoder ${weight_path} 8
  ```
- 运行量化权重时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
- 示例
  ```shell
  cd ${llm_path}/tests/modeltest
  bash run.sh pa_fp16 performance [[256,256],[512,512],[1024,1024],[2048,2048]] 1 starcoder ${weight_path} 8
  ```
- 运行量化权重时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](../../README.md)
