# README

- [LLaMA（Large Language Model Meta AI）](https://github.com/facebookresearch/llama/tree/llama_v1)和 [LLaMA2（Large Language Model Meta AI 2）](https://github.com/facebookresearch/llama)，是由 Meta AI 发布的一个开放且高效的大型基础语言模型，可以通过自然语言交互的方式提供知识、文本生成、语言翻译、语言理解、代码编写和解释等任务。

- 此代码仓中实现了一套基于NPU硬件的LLaMa推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各LLaMa模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16（仅800I A2支持） | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化（仅300I DUO支持） | MOE | MindIE Service | TGI | 长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|----------|----------|--------------|--------------------------|-----|----------------|-----|--------|
| LLaMa-7B    | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | √    | √                    | √               | √               | ×        | ×        | ×            | ×                        | ×   | ×              | ×   |×       |
| LLaMa-13B   | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | √    | √                    | √               | √               | ×        | ×        | ×            | ×                        | ×   | ×              | ×   |×       |
| LLaMa-33B   | 支持world size 4,8         | 支持world size 4,8           | √    | √                    | √               | √               | ×        | ×        | ×            | √                        | ×   | ×              | ×   |×       |
| LLaMa-65B   | 支持world size 8           | ×                            | √    | √                    | √               | √               | ×        | √        | ×            | ×                        | ×   | √              | √   |×       |
| LLaMa2-7B   | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | √    | √                    | √               | √               | √        | ×        | ×            | √                        | ×   | √              | √   |×       |
| LLaMa2-13B  | 支持world size 1,2,4,8     | 支持world size 1,2,4,8           | √    | √                    | √               | √               | √        | ×        | ×            | √                        | ×   | √              | √   |×       |
| LLaMa2-70B  | 支持world size 8           | ×                            | √    | √                    | √               | √               | √        | √        | ×            | ×                        | ×   | √              | √   |×       |

- 此模型仓已适配的模型版本
  - [LLaMa系列](https://github.com/facebookresearch/llama/tree/llama_v1)
  - [LLaMa2系列](https://github.com/facebookresearch/llama/tree/v2)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/llm_model/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| weight_path | 模型权重路径                            |

## 权重
### 权重下载
- [LLaMa-7B](https://huggingface.co/huggyllama/llama-7b/tree/main)
- [LLaMa-13B](https://huggingface.co/huggyllama/llama-13b/tree/main)
- [LLaMa-33B](https://huggingface.co/pinkmanlove/llama-33b-hf/tree/main)
- [LLaMa-65B](https://huggingface.co/huggyllama/llama-65b/tree/main)
- [LLaMa2-7B](https://huggingface.co/NousResearch/Llama-2-7b-hf/tree/main)
- [LLaMa2-13B](https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main)
- [LLaMa2-70B](https://huggingface.co/NousResearch/Llama-2-70b-hf/tree/main)

#### LLaMa 33B权重添加Special token
- LLaMa 33B中tokenizer原始的special token为空，需手动将权重文件中的`special_tokens_map.json`文件替换成以下内容（若不存在此文件则新增）
  ```json
  {
    "add_bos_token": true,
    "add_eos_token": false,
    "bos_token": {
      "content": "<s>",
      "lstrip": false,
      "normalized": true,
      "rstrip": false,
      "single_word": false
    },
    "clean_up_tokenization_spaces": false,
    "eos_token": {
      "content": "</s>",
      "lstrip": false,
      "normalized": true,
      "rstrip": false,
      "single_word": false
    },
    "model_max_length": 2048,
    "pad_token": null,
    "sp_model_kwargs": {},
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": true,
      "rstrip": false,
      "single_word": false
    }
  }
  ```

### 权重转换
- 参考[此README文件的【权重转换】章节](../../README.md)

### 量化权重生成
- 基于原始的浮点权重，使用量化工具，将高位浮点数转为低位的定点数，生成量化权重。
- 量化权重统一使用`${llm_path}/examples/convert/model_slim/quantifier.py`脚本生成，以下提供LLaMa模型量化权重生成快速启动命令，各模型量化方式的具体参数配置见`${llm_path}/examples/models/llama/generate_quant_weight.sh`
  - trust_remote_code为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
  - 脚本依赖`jq`指令，执行`jq --version`查看是否安装`jq`，若返回`jq: command not found`，则执行`apt install jq`进行安装。
- 注意事项：
  - `model_path`和`save_directory`请勿使用同一个文件夹，避免浮点权重和量化权重混淆
  - NPU多卡量化注意事项和环境要求见[此README中的【NPU多卡量化】章节](../../README.md)

#### W8A8
- LLaMa2-7B/13B推荐使用W8A8 + Antioulier（离群值抑制）量化
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd ${llm_path}
  # 生成llama2-7b量化权重，无回退层，antioutlier使用m1算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama2_7b_w8a8 -trust_remote_code
  # 生成llama2-13b量化权重，无回退层，antioutlier使用m2算法配置，使用min-max量化方式，校准数据集使用50条BoolQ数据，在CPU上进行运算
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama2_13b_w8a8 -trust_remote_code
  ```
- 大参数量模型LLaMa2-70B推荐使用NPU多卡W8A8量化
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd ${llm_path}
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8量化权重路径} -type llama2_70b_w8a8 -trust_remote_code
  ```

#### W8A16
- LLaMa-65B、LLaMa2-70B推荐使用以下量化配置
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd ${llm_path}
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A16量化权重路径} -type llama1_65b/llama2_70b_w8a16 -trust_remote_code
  ```

#### W8A8SC 稀疏量化
- 稀疏量化注意事项和环境要求见[此README中的【稀疏量化权重多卡切分及压缩脚本】章节](../../README.md)
- 步骤一：生成稀疏量化权重
  ```shell
  # 设置CANN包的环境变量
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd ${llm_path}
  # LLaMa2 7B/13B推荐使用以下量化配置
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8S量化权重路径} -type llama2_7b_w8a8s -trust_remote_code
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8S量化权重路径} -type llama2_13b_w8a8s -trust_remote_code
  # LLaMa1 33B推荐使用以下量化配置（生成权重后需将浮点权重下的special_tokens_map.json文件复制到W8A8S量化权重路径）
  bash examples/models/llama/generate_quant_weight.sh -src {浮点权重路径} -dst {W8A8S量化权重路径} -type llama1_33b_w8a8s -trust_remote_code
  ```
- 步骤二：切分及压缩量化权重
  ```shell
  # 切分并压缩量化权重
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} --trust_remote_code
  ```

## 推理
> **说明：**
> 运行时请确认权重`${weight_path}/config.json`中的`torch_dtype`、`kv_quant`和`quantize`字段配置正确，参考[此README文件的【权重设置】章节](../../README.md)

### 对话测试
**运行Flash Attention**
- 运行启动脚本
  ```shell
  cd ${llm_path}
  bash examples/models/llama/run_fa.sh ${weight_path} -trust_remote_code
  ```
  - trust_remote_code为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件的【部分环境变量介绍】章节](../../README.md)
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考【特性矩阵】章节
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export INF_NAN_MODE_ENABLE=0`
    - 设置溢出值的处理方式，开启此环境变量后，若出现溢出，溢出至会被置为NaN；若不开启此变量，则会对溢出值进行截断

**运行Paged Attention**
- 运行启动脚本
  ```shell
  cd ${llm_path}
  bash examples/models/llama/run_pa.sh ${weight_path} -trust_remote_code
  ```
  - trust_remote_code为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件的【部分环境变量介绍】章节](../../README.md)
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
    - 各模型支持的核心数参考【特性矩阵】章节
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export INF_NAN_MODE_ENABLE=0`
      - 设置溢出值的处理方式，开启此环境变量后，若出现溢出，溢出至会被置为NaN；若不开启此变量，则会对溢出值进行截断

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 运行BF16时将`pa_fp16`替换为`pa_bf16`。
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 测试8卡精度
    bash run.sh pa_fp16 full_BoolQ 1 llama ${weight_path} 8
    bash run.sh pa_bf16 full_BoolQ 1 llama ${weight_path} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 运行BF16时将`pa_fp16`替换为`pa_bf16`。
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # 测试8卡性能
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama ${weight_path} 8
    bash run.sh pa_bf16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama ${weight_path} 8
    ```
## 虚拟内存使用
- `export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"` 
    - 开启虚拟内存使用，带来更好的内存碎片使用，节约内存，但会使性能会略有劣化
- `export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False"` 
    - 若内存够用，对性能有要求，建议关闭虚拟内存使用

## FAQ
- 更多环境变量见[此README文件的【部分环境变量介绍】章节](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件的【启动脚本】章节](../../README.md)