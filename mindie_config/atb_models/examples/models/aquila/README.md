# README

- 悟道·天鹰（Aquila） 语言大模型是首个具备中英双语知识、支持商用许可协议、国内数据合规需求的开源语言大模型。

- 此代码仓中实现了一套基于NPU硬件的Aquila推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各Aquila模型支持的特性

| 模型及参数量           | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI  | 长序列 |
| ---------------------- |-------------------------|-----------------------| ---- |-----| --------------- | --------------- | -------- | --------- | --------- | ------------ | -------------------------- | ---- | ------ | ---- |
| Aquila-7B                | 支持world size 1,2,4,8    | 支持world size 1,2      | √    | ×   | √               | √               | ×        | ×                  | ×            | ×                          | ×    | ×      | ×    | ×    |
| Aquila2-7B               | 支持world size 1,2,4,8    | 支持world size 1,2      | √    | ×   | √               | √               | ×        | ×                  | ×            | ×                          | ×    | ×      | ×    | ×    |
| Aquila2-34B              | 支持world size 4,8        | ×                     | √    | ×   | √               | √               | ×        | ×                  | ×            | ×                          | ×    | ×      | ×    | ×    |

- 此模型仓已适配的模型版本
    - [FalshAI GitHub仓](https://github.com/FlagAI-Open/FlagAI/)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                                                                                                  |
|--------|---------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                     |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；Aquila和Aquila2的工作脚本所在路径为`${llm_path}/examples/models/aquila`                                                 |
| weight_path | 模型权重路径                                                                                                              |

## 权重
**权重下载**
- [Aquila-7B](https://huggingface.co/BAAI/Aquila-7B/tree/main)
- [Aquila2-7B](https://huggingface.co/BAAI/Aquila2-7B/tree/main)
- [Aquila2-34B](https://huggingface.co/BAAI/Aquila2-34B/tree/main)
**权重转换**
- 参考[此README文件](../../README.md)

**量化权重生成**
- 基于原始的FP16的权重，生成量化权重
- W8A8 Antioutlier量化权重请使用以下指令生成
- 暂不支持

- W8A8量化权重请使用以下指令生成
- 暂不支持

- W8A16量化权重请使用以下指令生成
- 暂不支持

- 稀疏量化权重请使用以下指令生成
- 暂不支持

**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试
**运行Flash Attention FP16**
- 其余Aquila模型参考以下运行方式
    - 运行启动脚本
        - 在\${llm_path}目录下执行以下指令
          ```shell
          bash ${script_path}/run_fa.sh ${weight_path}
          ```
    - 环境变量说明
        - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
            - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
            - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
            - 各模型支持的核心数参考“特性矩阵”
        - `export MASTER_PORT=20031`
            - 设置卡间通信端口
            - 默认使用20031端口
            - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
            - 设置时端口建议范围为：20000-20050
        - 以下环境变量与性能和内存优化相关，通常情况下无需修改
          ```shell
          export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
          export INF_NAN_MODE_ENABLE=0
          export ATB_OPERATION_EXECUTE_ASYNC=1
          export TASK_QUEUE_ENABLE=1
          export ATB_CONVERT_NCHW_TO_ND=1
          export HCCL_BUFFSIZE=120
          export HCCL_WHITELIST_DISABLE=1
          export ATB_CONTEXT_WORKSPACE_RING=1
          export ATB_CONTEXT_WORKSPACE_SIZE=2629145600
          export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
          export ATB_LAUNCH_KERNEL_WITH_TILING=0
          export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1
          export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=0
    
          ```

**运行Flash Attention BF16**
- 暂不支持

**运行Flash Attention W8A8**
- 暂不支持

**运行Flash Attention W8A16**
- 暂不支持

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
    - `export MASTER_PORT=20031`
        - 设置卡间通信端口
        - 默认使用20031端口
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
- 暂不支持

**运行Paged Attention W8A8**
- 暂不支持

**运行Paged Attention W8A16**
- 暂不支持

**运行KV cache量化**
- 暂不支持

**运行稀疏量化**
- 暂不支持

**运行MOE量化**
- 暂不支持

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)
    - 示例
      ```shell
      cd ${llm_path}/tests/modeltest
      export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      bash run.sh pa_fp16 full_BoolQ 1 aquila_7b ${aquila-7b权重路径} 8
      bash run.sh pa_fp16 full_BoolQ 1 aquila2_7b ${aquila2-7b权重路径} 8
      bash run.sh pa_fp16 full_BoolQ 1 aquila2_34b ${aquila2-34b权重路径} 8
      ```
    - MMLU测试集精度测试
      - 使用GPU测试Aquila模型测试MMLU数据集，需修改如下配置：
      - 1、修改开源文件config.json中max_position_embeddings大于3072
      - 2、修改开源文件tokenizer_config.json中model_max_length为3072

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
    - 示例
      ```shell
      cd ${llm_path}/tests/modeltest
      export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      export ATB_LLM_BENCHMARK_ENABLE=1
      bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 aquila_7b ${aquila-7b权重路径} 8
      bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 aquila2_7b ${aquila2-7b权重路径} 8
      bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 aquila2_34b ${aquila2-34b权重路径} 8
      ```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)