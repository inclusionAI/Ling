# README

[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) 项目开源了中文LLaMA模型和指令精调的Alpaca大模型，以进一步促进大模型在中文NLP社区的开放研究。这些模型在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练，进一步提升了中文基础语义理解能力。同时，中文Alpaca模型进一步使用了中文指令数据进行精调，显著提升了模型对指令的理解和执行能力。

- 此代码仓中实现了一套基于NPU硬件的Chinese-LLaMA-Alpaca系列模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各Chinese-LLaMA-Alpaca模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE | TGI |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|---------|--------------|----------|--------|--------|-----|
| Chinese-Alpaca-13B    | 支持world size 1,2,4,8   | 支持world size 1,2,4     | 是   | 否   | 否              | 是              | 否       | 否       | 否           | 否       | 否     | 否     | 否  |

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径；若使用编译好的包，则路径为`${working_dir}/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models/`    |
| script_path | 脚本所在路径；Chinese-Alpaca-13B的工作脚本所在路径为`${llm_path}/examples/models/chinese_alpaca`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**

- lora权重: [Chinese-Alpaca-Lora-13B](https://pan.baidu.com/s/1wYoSF58SnU9k0Lndd5VEYg?pwd=mm8i)
- 原模型权重: [LLaMA-13B](https://huggingface.co/huggyllama/llama-13b/tree/main)
> 下载后务必检查压缩包中模型文件的SHA256是否一致，请查看[SHA256.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md)

**lora权重合并**
- 合并lora权重和原模型权重，请参考[合并教程](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2#%E5%A4%9Alora%E6%9D%83%E9%87%8D%E5%90%88%E5%B9%B6%E9%80%82%E7%94%A8%E4%BA%8Echinese-alpaca-plus)

**权重转换**
> 若权重中不包含safetensors格式，则执行权重转换步骤，否则跳过
- 参考[此README文件](../../README.md)

**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试

**运行Paged Attention FP16**
- 运行启动脚本
  - 将`${llm_path}`加入`PYTHONPATH`搜索目录
    ```shell
    export PYTHONPATH=${llm_path}:${PYTHONPATH}
    ```
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

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_fp16 full_CEval 1 llama ${Chinese-Alpaca-13B权重路径} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 llama ${Chinese-Alpaca-13B权重路径} 8
    ```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)