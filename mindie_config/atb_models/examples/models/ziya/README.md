# README

- [Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0) 是IDEA研究院封神榜团队开源的代码大模型。


# 特性矩阵
- 此矩阵罗列Ziya-Coding模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI |  长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|-----|-----|
| Ziya-Coding-34B-v1.0 | 支持world size 4,8     | 支持world size 4,8           | 是   | 否                  | 否         | 是              | 否       | 否        | 否           | 否                       | 否  | 否  | 否 |  否 |

- 此模型仓已适配的模型版本
  - Ziya-Coding-34B-v1.0 (transformers==4.36.2)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；Ziya-Coding-34B-v1.0的工作脚本所在路径为`${llm_path}/examples/models/ziya`               |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**

- [Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0/tree/main)

**权重转换**
- 参考[此README文件](../../README.md)


**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 对话测试
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
    bash run.sh pa_fp16 full_BoolQ 1 ziya ${Ziya-Coding-34B-v1.0权重路径} 8
    ```

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 ziya ${Ziya-Coding-34B-v1.0权重路径} 8
    ```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)