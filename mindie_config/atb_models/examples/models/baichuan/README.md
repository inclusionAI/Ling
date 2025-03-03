# README

- Baichuan大模型，融合了意图理解、信息检索以及强化学习技术，结合有监督微调与人类意图对齐，在知识问答、文本创作领域表现突出。

- 此代码仓中实现了一套基于NPU硬件的Baichuan推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵

- 此矩阵罗列了各Baichuan模型支持的特性

| 模型及参数量  | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 | Razor Attention |
| ------------- | -------------------------- | --------------------------- | ---- | ---- | --------------- | --------------- | -------- | --------- | ------------ | -------- | ------- | -------------- | --- | ------ | --------------- |
| Baichuan2-7B  | 支持world size 1,2,4,8     | 支持world size 2            | √    | ×    | √               | √               | √        | ×         | ×            | ×        | ×       | √              | √   | ×      | ×               |
| Baichuan2-13B | 支持world size 2,4,8       | 支持world size 2,4          | √    | ×    | ×               | √               | √        | √         | ×            | ×        | ×       | √              | √   | ×      | √               |
| Baichuan-7B   | 支持world size 1,2,4,8     | 支持world size 2            | √    | ×    | √               | √               | ×        | ×         | ×            | ×        | ×       | √              | ×   | ×      | ×               |
| Baichuan-13B  | 支持world size 2,4,8       | 支持world size 2,4          | √    | ×    | ×               | √               | ×        | ×         | ×            | ×        | ×       | √              | ×   | ×      | ×               |


# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                                     |
| llm_path    | 模型仓所在路径。若使用编译好的包，则建议解压后的路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径。Baichuan系列模型的工作脚本所在路径为${llm_path}/examples/models/baichuan                                                                             |
| weight_path | 模型权重路径                                                                                                                                                       |

## 权重
**权重下载**
- [Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main)
- [Baichuan-13B](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat/tree/main)
- [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/tree/main)
- [Baichuan2-13B](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/tree/main)
- 注意事项：
    - 请下载全部权重文件

**权重转换**
- Paged Attention 场景下需要.safetensors 格式的权重，如果没有，参考[此README文件](../../README.md)转换

**量化权重生成**  
基于原始的FP16的权重，生成量化权重。量化导出脚本使用参数介绍如下：  
| 模型          | dst            | src                  | type                      | 备注 |
| ------------- | -------------- | -------------------- | ------------------------- | ---- |
| baichuan2_7b  | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_7b_w8a8         | -    |
| baichuan2_7b  | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_7b_w8a8_kvcache | -    |
| baichuan2_13b | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_13b_w8a8        | -    |
| baichuan2_13b | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_13b_w4a16       | -    |
| baichuan2_13b | 源模型权重路径 | 目标量化权重导出路径 | baichuan2_13b_w8a16       | -    |


trust_remote_code为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。

例如，对于Baichuan2_7b的w8a8格式量化而言，命令如下：  
```
cd ${llm_path}
bash examples/models/baichuan/generate_baichuan2_quant_weights.sh -src "源权重路径" -dst "目标权重路径" -type baichuan2_7b_w8a8 -trust_remote_code
```

- 稀疏量化权重请使用以下指令生成
  - 暂不支持


**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理
### 对话测试
**运行Flash Attention FP16**
- 其余Baichuan模型参考以下运行方式
    - 运行启动脚本
        - 在\${llm_path}目录下执行以下指令
          ```shell
          bash examples/models/baichuan/run_fa.sh ${weight_path} -trust_remote_code
          ```
          - trust_remote_code为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
    - 环境变量说明
        - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
            - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
            - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
            - 各模型支持的核心数参考“特性矩阵”
        - `export MASTER_PORT=20036`
            - 设置卡间通信端口
            - 默认使用20036端口
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
      chat模式（仅支持baichuan2系列）:
      bash examples/models/baichuan/run_pa.sh ${weight_path} chat -trust_remote_code

      非chat模式:
      bash examples/models/baichuan/run_pa.sh ${weight_path} -trust_remote_code
      ```
      - trust_remote_code为可选参数代表是否信任模型权重路径下的自定义代码文件：默认不执行。传入此参数，则信任本地自定义代码文件。
- 环境变量说明
    - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
        - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
        - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
        - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
        - 各模型支持的核心数参考“特性矩阵”
    - `export MASTER_PORT=20036`
        - 设置卡间通信端口
        - 默认使用20036端口
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
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为W8A8量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改W8A8量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a8`
  - 若config.json中无此字段，则新增

**运行Paged Attention W8A16**
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同
  - `${weight_path}`为W8A16量化权重的路径
- 环境变量说明
  - 参见“运行Paged Attention FP16”中的环境变量说明
- 相比于FP16，运行量化时需修改W8A16量化权重`${weight_path}/config.json`中的`quantize`字段，将此字段对应的值修改为`w8a16`
  - 若config.json中无此字段，则新增


**运行KV cache量化**
- 暂不支持

**运行稀疏量化**
- 暂不支持

**运行MOE量化**
- 暂不支持

**运行Razor Attention FP16**
- 开启环境变量
  ```
  export ATB_LLM_RAZOR_ATTENTION_ENABLE=1
  ```
- 运行启动脚本
  - 与“运行Paged Attention FP16”的启动方式相同

## 精度测试
- 参考[此README文件](https://gitee.com/ascend/MindIE-LLM/blob/master/examples/atb_models/tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_7b ${baichuan-7b权重路径} trust_remote_code 4
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_13b ${baichuan-13b权重路径} trust_remote_code 4
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_7b ${baichuan2-7b权重路径} trust_remote_code 4
    bash run.sh pa_fp16 full_BoolQ 1 baichuan2_13b ${baichuan2-13b权重路径} trust_remote_code 4
    ```
- 注意：baichuan-7b和baichuan-13b模型测试时复用baichuan2_7b和baichuan2_13b的model_name
- 运行量化权重时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](https://gitee.com/ascend/MindIE-LLM/blob/master/examples/atb_models/examples/README.md)
- 测试longbench时需开启环境变量ALiBi Mask Free:
```
export IS_ALIBI_MASK_FREE=1
```

## 性能测试
- 支持ALiBi Mask Free。默认关闭，如需开启，请修改当前目录下的run_pa.sh中环境变量如下：
```
export IS_ALIBI_MASK_FREE=1
```
- 参考[此README文件](https://gitee.com/ascend/MindIE-LLM/blob/master/examples/atb_models/tests/modeltest/README.md)
  - 示例
    ```shell
    cd ${llm_path}/tests/modeltest
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export ATB_LLM_BENCHMARK_ENABLE=1
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 baichuan2_7b ${baichuan-7b权重路径} trust_remote_code 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 baichuan2_13b ${baichuan-13b权重路径} trust_remote_code 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 baichuan2_7b ${baichuan2-7b权重路径} trust_remote_code 8
    bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 baichuan2_13b ${baichuan2-13b权重路径} trust_remote_code 8
    ```
- 注意：baichuan-7b和baichuan-13b模型测试时复用baichuan2_7b和baichuan2_13b的model_name
- 运行量化权重时需注意`${weight_path}/config.json`中的`quantize`字段和`torch_dtype`字段是否与权重匹配，参考[此README文件](https://gitee.com/ascend/MindIE-LLM/blob/master/examples/atb_models/examples/README.md)
- 特殊场景说明: 若在性能测试时发现有波动情况，可配置透明大页，提升内存访问性能。该功能请按需开启，对内存占用有一定影响。
```shell
# 性能测试时，可按需开启透明大页
echo always > /sys/kernel/mm/transparent_hugepage/enabled
# 关闭透明大页
echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

## FAQ
- 更多环境变量见[此README文件](../../README.md)
- 对话测试实际执行的Python文件为`${llm_path}/examples/run_fa.py`和`${llm_path}/examples/run_pa.py`；这两个文件的参数说明见[此README文件](../../README.md)