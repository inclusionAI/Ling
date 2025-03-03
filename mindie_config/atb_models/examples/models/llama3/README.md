# README

- [Llama3（Large Language Model Meta AI 3）](https://github.com/meta-llama/llama3)和[Llama3.1](https://github.com/meta-llama/llama-models/tree/main/models/llama3_1)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- Llama3当前包含两个参数版本：Llama3-8B和Llama3-70B。相较于Llama2，Llama3支持8K长文本，改进的tokenizer具有128K token的词汇量，可实现更好的性能；同时，Llama3在代码生成等任务上实现了领先，能够进行复杂的推理，更遵循指令并解决很多微妙的问题。Llama3.1包含3个参数版本：Llama3.1-8B、Llama3.1-70B-Instruct和Llama3.1-405B-Instruct，并进一步支持了RoPE-Scaling计算。

- 此代码仓中实现了一套基于NPU硬件的Llama3与Llama3.1推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了Llama3模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | Attention 量化 | lccl 量化 | MindIE Service | TGI |  长序列 | Multi LoRA|
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|---------|-------|-----------|--------------|--------------------------|-----|-----|--------|-----|-----|-----|
| Llama3-8B    | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | √   | √                   | √              | √              | ×       | ×              | ×                       | ×  | × | × | √     | ×  | ×  | ×  |
| Llama3-70B   | 支持world size 8           | ×                            | √   | √                   | √              | √              | ×       | √                | ×                       | ×  | × | × | √     | ×  |  ×  | ×  |
| Llama3.1-8B    | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | √   | √                   | √              | √              |      √        | ×              | ×                       | ×  | × | √ | √     | ×  | ×  | ×  |
| Llama3.1-70B | 支持world size 8           | 支持world size 2,4              | √   | √                   | √              | √              | √       | ×              | √                       | √  | √  | × | √     | ×  | √  | √   |
| Llama3.1-405B    | 支持world size 16(单卡64G)     | ×           | √   | √                   | √              | √              |      ×        | ×              | ×                       | ×  | × | × | ×     | ×  | ×  | ×   |
| Llama-3.2-1B-Instruct   | 支持world size 1,2,4,8 | 支持world size 1,2,4    |  √ |   √      | √       |    √            |          ×     |      ×          |       ×                  | ×   |   ×    |   × |  ×  | ×    |  ×  |  ×  |
| Llama3.2-3B-Instruct   | 支持world size 1,2,4,8 | 支持world size 1,2,4     |  √ |  √    |     √     |   √             |           ×    |        ×        |        ×                 |   × |   ×    |  ×  |  ×  | ×    |  ×  |  ×  |

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | MindIE加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；Llama3的工作脚本所在路径为`${llm_path}/examples/models/llama3`                            |
| weight_path | 模型权重路径                            |
| max_output_length | 对话测试中最大输出token数 |

## 权重
**权重下载**
- [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main)
- [Llama3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B/tree/main)
- [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/main)
- [Llama3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B/tree/main)
- [Llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/tree/main)
- [Llama3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct/tree/main)
- [Llama3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main)
- [Llama3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main)

## 环境准备
**基础环境配置**
- 参考[此README文件](../../../README.md)
- 检查python依赖库中transformers版本的配置，Llama3.1要求transformers库版本为4.43.1及以上。
  ```shell
  pip show transformers
  # 请将transformers更新至对应版本
  # Llama3.1
  pip install transformers==4.43.1
  # Llama3.2
  pip install transformers==4.45.0
  ```
- 根据需要执行的推理模式来修改模型配置文件`${weight_path}/config.json`中的`torch_dtype`字段。如果需要执行FP16推理，那么将`torch_dtype`字段修改为`float16`；执行BF16推理，则将`torch_dtype`字段修改为`bfloat16`。
- 300I DUO硬件不支持BF16推理，仅支持FP16推理。

**量化权重生成**
- 生成量化权重依赖msModelSlim工具，安装方式见[此README](https://gitee.com/ascend/msit/tree/dev/msmodelslim)。
- 基于原始的FP16或BF16的权重，生成量化权重
- 量化权重统一使用`${llm_path}/examples/convert/model_slim/quantifier.py`脚本生成，以下提供LLaMa3模型量化权重生成快速启动命令，各模型量化方式的具体参数配置见`${llm_path}/examples/models/llama3/generate_quant_weight.sh`
- W8A8量化权重请使用以下指令生成
  - 当前仅Llama3.1-8B,Llama3.1-70B-Instruct支持W8A8量化
  - 执行量化脚本
    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 关闭虚拟内存
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    cd ${llm_path}
    # Llama3.1-8b量化，无回退层，antioutlier使用m1算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算
    bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama3.1_8b_w8a8
    # Llama3.1-70B-Instruct量化 fp16，无回退层，antioutlier使用m3算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在NPU上进行运算
    bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama3.1_70b_instruct_fp16_w8a8
    # Llama3.1-70B-Instruct量化 bf16，有回退层，antioutlier使用m3算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在NPU上进行运算
    bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama3.1_70b_instruct_bf16_w8a8
    ```

- KV cache量化权重请使用以下指令生成
  - 当前仅Llama3.1-70B-Instruct W8A8量化支持搭配KV cache int8量化
  - 相比于W8A8量化，需额外设置`use_kvcache_quant`参数为True
    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    # 执行kvcache int8量化，注意除了`use_kvcache_quant`参数，其余参数设置以具体模型为准
    bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama3.1_70b_instruct_bf16_w8a8 -use_kvcache_quant True
    ```

- Attention量化权重请使用以下指令生成
  - 当前仅支持基于BF16权重生成量化权重
  - 当前仅Llama3.1-70B-Instruct W8A8量化支持搭配Attention量化
  - 需修改`modeling_llama.py`文件和`config.json`文件，配置方法参考[FA量化使用说明](https://gitee.com/ascend/msit/blob/master/msmodelslim/docs/FA%E9%87%8F%E5%8C%96%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)。  
  - 相比于W8A8量化，需额外设置`use_fa_quant`参数为True
    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    # 执行Attention量化，注意除了`use_fa_quant`参数，其余参数设置以具体模型为准
    bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama3.1_70b_instruct_bf16_w8a8 -use_fa_quant True
    ```


- W8A16量化权重请使用以下指令生成
  - 当前仅LLaMa3-70B支持W8A16量化
    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    bash examples/models/llama3/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A16量化权重路径} -type llama3_70b_w8a16
    ```


- lccl all reduce int8量化权重请使用以下指令生成
  - 当前仅Llama3.1-8B支持搭配lccl all reduce int8量化
  - 该量化权重需要使用多卡npu生成
    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 设置卡数
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 
    # 关闭虚拟内存
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
    cd ${llm_path}
    python examples/models/llama3/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {量化权重路径} --w_bit 8 --a_bit 8 --act_method 2 --disable_level L1000 --device_type npu --calib_file ${llm_path}/examples/convert/model_slim/boolq.jsonl --use_reduce_quant True --tp_size 4
    ```
  - 卡数与tp_size统一，模型运行时的卡数应与量化时用的卡数相同
## 推理

### 对话测试
**运行Paged Attention BF16**
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
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
    export ATB_LAUNCH_KERNEL_WITH_TILING=0
    ```
- 将模型配置文件`config.json`中的`torch_dtype`字段修改为`bfloat16`。
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    # 执行单卡推理
    export ASCEND_RT_VISIBLE_DEVICES=0
    bash ${script_path}/run_pa.sh ${weight_path} ${max_output_length}
    # 执行4卡推理
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    bash ${script_path}/run_pa.sh ${weight_path} ${max_output_length}
    # 执行8卡推理
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash ${script_path}/run_pa.sh ${weight_path} ${max_output_length}
    ```
- 注：当机器运行405B模型显存不够时，可以在多机环境运行(如64G双机16卡)，执行方法参考[此链接](https://www.hiascend.com/document/detail/zh/mindie/10RC2/mindiellm/llmdev/mindie_llm0030.html)

**运行Paged Attention FP16**
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 将模型配置文件`config.json`中的`torch_dtype`字段修改为`float16`。
- 运行启动脚本
  - 与“运行Paged Attention BF16”的启动方式相同

**运行Paged Attention W8A8**
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 修改模型配置文件`config.json`中的`torch_dtype`字段，保证与量化权重生成时原始权重配置文件`config.json`中的`torch_dtype`字段一致。
- 运行启动脚本
  - 与“运行Paged Attention BF16”的启动方式相同

**运行Flash Attention FP16**
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 将模型配置文件`config.json`中的`torch_dtype`字段修改为`float16`。
- 运行启动脚本
  - 参数说明
    - `--model_path`
      - 模型权重路径
    - `--input_text`
      - 输入文本
    - `--max_input_length`
      - 最大输入长度
    - `--max_output_length`
      - 最大输出长度
    - `--max_position_embeddings`
      - 对应模型配置文件`config.json`中的`max_position_embeddings`字段，且默认为`config.json`中该字段的值。为避免显存占用过大，建议此参数设置小于32768。
  - 示例
  ```shell
  # 使用8卡运行Flash Attention，设置模型权重路径，设置输出长度为2048个token。
  torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_fa --model_path ${weight_path} --max_output_length 2048 
  ```

**运行Flash Attention BF16**
- 环境变量说明
  - 参见“运行Paged Attention BF16”中的环境变量说明
- 将模型配置文件`config.json`中的`torch_dtype`字段修改为`bfloat16`。
- 运行启动脚本
  - 与“Flash Attention FP16”的启动方式相同

**运行Multi-Lora**
- 下载Lora权重：Lora权重中需包含至少一个safetensors格式的文件，和一个名为`adapter_config.json`的配置文件
- 在基础模型的权重文件夹中，新增`lora_adapter.json`文件，内容为需要预加载的Lora权重，例如：
    ```json
    {"adapter1": "/path/to/lora/llama-3.1-70b/adapter1", "adapter2": "/path/to/lora/llama-3.1-70b/adapter2"}
    ```
- 进行推理时需指定每个请求所使用的adapter权重，默认仅使用基础模型权重
- 运行示例
    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export INF_NAN_MODE_ENABLE=0
    torchrun --nproc_per_node 8 --master_port 20030 -m examples.run_pa --model_path ${基础模型权重} --max_output_length 50 --max_batch_size 2 --input_dict '[{"prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "adapter": "adapter1"}, {"prompt": "What is deep learning?"}]'
    ```
- 约束与限制
    - 仅支持在Atlas 800I A2上运行
    - Lora权重不支持热加载
    - 在显存充足的情况下至多加载10个Lora权重
    - 仅支持浮点模型
    - `adapter_id`唯一且不能与`base`重名

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    # 测试8卡精度。
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 在bf16模式下测试
    bash run.sh pa_bf16 full_BoolQ 1 llama ${llama3系列模型权重路径} 8 
    bash run.sh pa_bf16 full_BoolQ 1 llama ${llama3.1系列模型权重路径} 8
    # 在bf16模式下使用chat版本、128k输入长度测试NeedleBench数据集精度
    bash run.sh pa_bf16 full_NeedleBench 128k 1 llama chat ${llama3.1系列模型权重路径} 8
    # 在fp16模式下测试，并注意参考“环境准备”一栏修改config.json文件
    bash run.sh pa_fp16 full_BoolQ 1 llama ${llama3系列模型权重路径} 8 
    bash run.sh pa_fp16 full_BoolQ 1 llama ${llama3.1系列模型权重路径} 8
    bash run.sh pa_fp16 full_BoolQ 2 llama lora ${包含每个请求对应lora adapter名称的文件路径} ${llama系列模型权重路径} 8
    # 在w8a8模式下测试
    # 先修改模型配置文件config.json中的torch_dtype字段，保证与量化权重生成时原始权重配置文件config.json中的torch_dtype字段一致。
    bash run.sh pa_fp16 full_BoolQ 1 llama ${llama3.1-8B模型权重路径} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    # 测试8卡性能。如果需要测试fp16，请参考精度测试使用pa_fp16来替换pa_bf16并修改config.json。
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama ${llama3系列模型权重路径} 8
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama ${llama3.1系列模型权重路径} 8
    bash run.sh pa_fp16 performance [[1024,1024]] 2 llama lora ${包含每个请求对应lora adapter名称的文件路径} ${llama系列模型权重路径} 2

    # 测试长序列性能（800I A2 32G支持32k、64k序列长度， 800I A2 64G支持32k、64k、128k、192k、256k序列长度）。
    # 在800I A2 64G上测试长序列性能时请设置环境变量`export MATMUL_ND_NZ_ENABLE=1`
    # 如果需要测试fp16，请参考精度测试使用pa_fp16来替换pa_bf16并修改config.json。
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_bf16 performance [[32768,1024],[65536,1024],[131072,1024]] 1 llama ${llama3-70b权重路径} 8
    ```

## FAQ
- 虚拟内存默认为开启，开启时能提升显存利用率。
  ```shell
  # 开启虚拟内存
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  # 关闭虚拟内存
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  ```
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_pa.py`与`${llm_path}/examples/run_fa.py`；文件参数说明请见[此README文件](../../README.md)
