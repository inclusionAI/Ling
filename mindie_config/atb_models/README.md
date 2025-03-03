# README

## 变量解释
| 变量名      | 含义                                   |
| ----------- | -------------------------------------- |
| working_dir | 加速库及模型库下载后放置的目录         |
| cur_dir     | 运行指令或执行脚本时的路径（当前目录） |
| version     | 版本                                   |

## 环境准备
### 依赖版本
- 模型仓代码配套可运行的硬件型号
  - Atlas 800I A2（32GB/64GB显存）
  - Atlas 300I DUO（96GB显存）
- 模型仓代码运行相关配套软件
  - 系统OS
  - 驱动（HDK）
  - CANN
  - Python
  - PTA
  - 开源软件依赖
- 版本配套关系
  - 当前模型仓需基于CANN包8.0版本及以上，Python 3.10，torch 2.1.0进行环境部署与运行

### 1.1 安装HDK

- 详细信息可参见[昇腾社区驱动与固件](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/envdeployment/instg/instg_000018.html)
- 第一次安装时：先安装driver，再安装firmwire，最后执行`reboot`指令重启服务器生效
- 若服务器上已安装驱动固件，进行版本升级时：先安装firmwire，再安装driver，最后执行`reboot`指令重启服务器生效

#### 1.1.1 安装firmwire

- 下载

| 包名                                     |
| ---------------------------------------- |
| Ascend-hdk-*-npu-firmware_${version}.run |

  - 根据芯片型号下载对应的安装包

- 安装
  ```bash
  chmod +x Ascend-hdk-*-npu-firmware_${version}.run
  ./Ascend-hdk-*-npu-firmware_${version}.run --full
  ```

#### 1.1.2 安装driver

- 下载

| cpu     | 包名                                                 |
| ------- | ---------------------------------------------------- |
| aarch64 | Ascend-hdk-*-npu-driver_${version}_linux-aarch64.run |
| x86     | Ascend-hdk-*-npu-driver_${version}_linux-x86-64.run  |
  - 根据CPU架构以及npu型号下载对应的driver

- 安装
  ```bash
  chmod +x Ascend-hdk-*-npu-driver_${version}_*.run
  ./Ascend-hdk-*-npu-driver_${version}_*.run --full
  ```

### 1.2 安装CANN

- 详细信息可参见[昇腾社区CANN软件](https://www.hiascend.com/software/cann)
- 安装顺序：先安装toolkit 再安装kernel

#### 1.2.1 安装toolkit

- 下载

| cpu     | 包名（其中`${version}`为实际版本）                 |
| ------- | ------------------------------------------------ |
| aarch64 | Ascend-cann-toolkit_${version}_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_${version}_linux-x86_64.run  |

- 安装
  ```bash
  # 安装toolkit  以arm为例
  chmod +x Ascend-cann-toolkit_${version}_linux-aarch64.run
  ./Ascend-cann-toolkit_${version}_linux-aarch64.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

#### 1.2.2 安装kernel

- 下载

| 包名                                       |
| ------------------------------------------ |
| Ascend-cann-kernels-*_${version}_linux.run |

  - 根据芯片型号选择对应的安装包

- 安装
  ```bash
  chmod +x Ascend-cann-kernels-*_${version}_linux.run
  ./Ascend-cann-kernels-*_${version}_linux.run --install
  ```

#### 1.2.3 安装加速库
- 下载加速库
  - [下载链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/261918053?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)。

  | 包名（其中`${version}`为实际版本）            |
  | -------------------------------------------- |
  | Ascend-cann-nnal_${version}_linux-aarch64.run |
  | Ascend-cann-nnal_${version}_linux-x86_64.run  |
  | ...                                          |

  - 将文件放置在\${working_dir}路径下

- 安装
    ```shell
    chmod +x Ascend-cann-nnal_*_linux-*.run
    ./Ascend-cann-nnal_*_linux-*.run --install --install-path=${working_dir}
    source ${working_dir}/nnal/atb/set_env.sh
    ```
- 可以使用`uname -a`指令查看服务器是x86还是aarch架构
- 可以使用以下指令查看abi是0还是1
    ```shell
    python -c "import torch; print(torch.compiled_with_cxx11_abi())"
    ```
    - 若输出结果为True表示abi1，False表示abi0

### 1.3 安装PytorchAdapter

先安装torch 再安装torch_npu

#### 1.3.1 安装torch

- 下载

  | 包名                                         |
  | -------------------------------------------- |
  | torch-2.1.0+cpu-cp310-cp310-linux_x86_64.whl |
  | torch-2.1.0-cp310-cp10-linux_aarch64.whl     |
  | ...                                          |

  - 根据所使用的环境中的python版本以及cpu类型，选择对应版本的torch安装包。

- 安装
  ```bash
  # 安装torch 2.1.0 的python 3.10 的arm版本为例
  pip install torch-2.1.0-cp310-cp310-linux_aarch64.whl
  ```

#### 1.3.2 安装torch_npu

[下载PyTorch Adapter](https://www.hiascend.com/developer/download/community/result?module=pt)，安装方法：

| 包名                        |
| --------------------------- |
| pytorch_v2.1.0_py38.tar.gz |
| pytorch_v2.1.0_py39.tar.gz |
| pytorch_v2.1.0_py310.tar.gz |
| ...                         |

- 安装选择与torch版本以及python版本一致的npu_torch版本

```bash
# 安装 torch_npu，以 torch 2.1.0，python 3.10 的版本为例
tar -zxvf pytorch_v2.1.0_py310.tar.gz
pip install torch*_aarch64.whl
```

### 1.4 安装开源软件依赖
| 模型                     | 开源软件依赖文件                                             |
| ------------------------ | ------------------------------------------------------------ |
| 默认依赖                 | [requirement.txt](./requirements/requirements.txt)           |
| Aquila                   | [requirement_aquila.txt](./requirements/models/requirements_aquila.txt) |
| Baichuan                 | [requirement_baichuan.txt](./requirements/models/requirements_baichuan.txt) |
| Bloom                    | [requirements_bloom.txt](./requirements/models/requirements_bloom.txt)      |
| CodeLlama                | [requirements_codellama.txt](./requirements/models/requirements_codellama.txt) |
| Deepseek-Coder           | [requirements_deepseek_coder.txt](./requirements/models/requirements_deepseek_coder.txt) |
| Deepseek-LLM             | [requirements_deepseek_llm.txt](./requirements/models/requirements_deepseek_llm.txt) |
| Deepseek-Moe             | [requirement_deepseek_moe.txt](./requirements/models/requirement_deepseek_moe.txt) |
| GPT-NeoX                 | [requirement_gpt_neox.txt](./requirements/models/requirement_gpt_neox.txt) |
| Internlm                 | [requirement_internlm.txt](./requirements/models/requirements_internlm.txt) |
| Internvl                 | [requirement_internvl.txt](./requirements/models/requirements_internvl.txt) |
| Qwen1                    | [requirements_qwen.txt](./requirements/models/requirements_qwen.txt) |
| Qwen1.5                  | [requirements_qwen1.5.txt](./requirements/models/requirements_qwen1.5.txt) |
| QwenVL                  | [requirements_qwen_vl.txt](./requirements/models/requirements_qwen_vl.txt)           |
| Minigpt4                 | [requirements_minigpt4.txt](./requirements/models/requirements_minigpt4.txt) |
| Llava                    | [requirements_llava.txt](./requirements/models/requirements_llava.txt) |
| Llama                    | [requirements_llama.txt](./requirements/models/requirements_llama.txt) |
| Llama1-33B               | [requirements_llama1_33b.txt](./requirements/models/requirements_llama1_33b.txt) |
| Llama3                   | [requirements_llama3.txt](./requirements/models/requirements_llama3.txt) |
| Mistral-7B-Instruct-v0.2 | [requirements_mistral_7b_instruct_v0.2.txt](./requirements/models/requirements_mistral_7b_instruct_v0.2.txt) |
| Mixtral-8x7B             | [requirements_mixtral_8x7b.txt](./requirements/models/requirement_mixtral_8x7b.txt)|
| Mixtral-8x22B             | [requirement_mixtral_8x22b.txt](./requirements/models/requirement_mixtral_8x22b.txt)|
| Starcoder2               | [requirements_starcoder2.txt](./requirements/models/requirements_starcoder2.txt) |
| Yi                       | [requirements_yi.txt](./requirements/models/requirements_yi.txt)           |
| Vlmo                     | [requirement_vlmo.txt](./requirements/models/requirement_vlmo.txt)           |

- 开源软件依赖请使用下述命令进行安装：
  ```bash
  pip install -r ./requirements/requirements.txt
  pip install -r ./requirements/models/requirements_{模型}.txt
  ```
- 各个模型开源软件依赖请使用上述表格各模型依赖的requirement_{模型}.txt进行安装，若模型未指定开源软件依赖，请使用默认开源软件依赖。

### 1.5 安装模型仓
- 场景一：使用编译好的包进行安装
  - 下载编译好的包
    - [下载链接](https://www.hiascend.com/developer/download/community/result?module=ie+pt+cann)

    | 包名                                                         |
    | ------------------------------------------------------------ |
    | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch1.11.0-abi0.tar.gz |
    | Ascend-mindie-atb-models_1.0.RC1_linux-aarch64_torch2.1.0-abi1.tar.gz |
    | Ascend-mindie-atb-models_1.0.RC1_linux-x86_64_torch1.11.0-abi1.tar.gz |
    | Ascend-mindie-atb-models_1.0.RC1_linux-x86_64_torch2.1.0-abi1.tar.gz |
    | ...                                                          |

  - 将文件放置在\${working_dir}路径下
  - 解压
    ```shell
    cd ${working_dir}
    mkdir MindIE-LLM
    cd MindIE-LLM
    tar -zxvf ../Ascend-mindie-atb-models_*_linux-*_torch*-abi*.tar.gz
    ```
  - 安装atb_llm whl包
    ```
    cd ${working_dir}/MindIE-LLM
    # 首次安装
    pip install atb_llm-0.0.1-py3-none-any.whl
    # 更新
    pip install atb_llm-0.0.1-py3-none-any.whl --force-reinstall
    ```
- 场景二：手动编译模型仓
  - 获取模型仓代码
    ```shell
    git clone https://gitee.com/ascend/MindIE-LLM.git
    ```
  - 切换到目标分支
    ```shell
    cd MindIE-LLM
    git checkout master
    ```
  - 代码编译
    ```shell
    cd examples/atb_models
    bash scripts/build.sh
    cd output/atb_models/
    source set_env.sh
    ```

- 安装 sdk

  ```
  # 若使用编译好的包（即1.5章节的场景一），则执行以下指令
  cd ${working_dir}/MindIE-LLM/examples/models/
  # 若使用gitee上的源码进行编译（即1.5章节的场景二），则执行以下指令
  cd ${working_dir}/MindIE-LLM/examples/atb_models/examples/models/
  # 首次安装
  pip3 install ./atb_speed_sdk/
  # 更新
  pip3 install ./atb_speed_sdk/ --force-reinstall
  ```

### 1.6 安装量化工具msModelSlim (可选)
  - 量化权重需使用该工具生成，具体生成方式详见各模型README文件
  - 工具下载及安装方式见[README](https://gitee.com/ascend/msit/blob/master/msmodelslim/README.md)

### 2.1 开启CPU Performance模式
- 开启CPU Performance模式以提高模型推理性能（首次开启时，根据提示安装依赖）
  ```
  cpupower -c all frequency-set -g performance
  ```
- 备注：
  - 仅ARM服务器适用
  - 指令请在裸机上执行

## 环境变量参考

### CANN、加速库、模型仓的环境变量
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ${working_dir}/nnal/atb/set_env.sh
# 若使用编译好的包（即1.5章节的场景一），则执行以下指令
source ${working_dir}/MindIE-LLM/set_env.sh
# 若使用gitee上的源码进行编译（即1.5章节的场景二），则执行以下指令
source ${working_dir}/MindIE-LLM/examples/atb_models/output/atb_models/set_env.sh
```

- 模型仓日志
  - 打开模型仓日志
    - 推荐使用
      ```shell
      export MINDIE_LOG_TO_STDOUT=1
      export MINDIE_LOG_TO_FILE=1
      export MINDIE_LOG_LEVEL=info
      ```
    - 复杂场景
      ```shell
      export MINDIE_LOG_TO_STDOUT='llm:true;false'
      # 将llm组件打屏设置为true，其他组件设置为false
      export MINDIE_LOG_TO_FILE='llm:true;service:false'
      # 将llm组件落盘设置为true，service组件设置为false
      export MINDIE_LOG_LEVEL='llm:critical;service:debug'
      # 将llm组件日志界别设置为critical，service组件日志界别设置为debug
      ```
    - 旧版本开关（支持到2025.06）    
      ```shell
      export ATB_LOG_TO_FILE=1
      export ATB_LOG_TO_STDOUT=1
      export ATB_LOG_LEVEL=INFO
      export TASK_QUEUE_ENABLE=1
      export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
      ```
  - 关闭模型仓日志
    - 推荐使用
      ```shell
      export MINDIE_LOG_TO_STDOUT=0
      export MINDIE_LOG_TO_FILE=0
      ```
    -旧版本日志开关（支持到2025.06）
      ```shell
      export ATB_LOG_TO_FILE=0
      export ATB_LOG_TO_STDOUT=0
      export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
      ```
  - 日志存放在~/mindie/log/debug下

- 加速库算子库日志
  - 打开加速库算子库日志
    ```shell
    export ASDOPS_LOG_TO_FILE=1 #算子库日志是否输出到文件
    export ASDOPS_LOG_TO_STDOUT=1 #算子库日志是否输出到控制台 
    export ASDOPS_LOG_LEVEL=ERROR #算子库日志级别
    ```
  - 关闭加速库算子库日志
    ```shell
    export ASDOPS_LOG_TO_FILE=0
    export ASDOPS_LOG_TO_STDOUT=0
    ```
  - 日志存放在~/atb/log下

- 注意：
    1、开启日志后会影响推理性能，建议默认关闭；当推理执行报错时，开启日志定位原因
    2、INFO级别日志和ASDOPS_LOG_TO_STDOUT开关同时打开时，会让控制台输出大量打印，请根据需要打开。

### 确定性计算（可选）
- NPU支持确定性计算，即多次运行同样的数据集，结果一致
  - 确定性计算需设置以下环境变量
    ```shell
    export LCCL_DETERMINISTIC=1
    export HCCL_DETERMINISTIC=true
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    ```
  - 关闭确定性计算
    ```shell
    export LCCL_DETERMINISTIC=0
    export HCCL_DETERMINISTIC=false
    export ATB_MATMUL_SHUFFLE_K_ENABLE=1
    ```
- 注意：开启确定性计算会影响性能
- 若使用相同的Prompt构造不同Batch size进行推理时，为了保证结果一致，除了开启以上环境变量外，需关闭通信计算掩盖功能
  ```shell
  export ATB_LLM_LCOC_ENABLE=0
  ```

### 多机推理
- 在性能测试时开启"AIV"提升性能，若有确定性计算需求时建议关闭"AIV"
```shell
   export HCCL_OP_EXPANSION_MODE="AIV"
```
- 若要在运行时中断进程，ctrl+C无效，需要使用pkill结束进程

### 显存分析
- 开启虚拟内存，提高内存碎片利用率
```shell
   export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

### 性能分析
#### Profiling
- 适用于性能分析（例如：查看单token耗时前三的算子）及定位性能问题（例如：算子间有空泡，单算子耗时过长）
- Step 1：开启Profiling环境变量
  ```shell
  export ATB_PROFILING_ENABLE=1
  ```
  - Profiler Level默认使用Level0
- Step 2：执行推理
  - 运行推理指令（此指令和不生成Profiling文件时运行的指令无差异）
  - 注意：生成Profiling文件时，增量token数量不应设置太大，否则会导致Profiling文件过大
- Step 3：下载单卡Profiling文件
  - Profiling文件默认保存在${cur_dir}/profiling目录下
    - 可以通过设置`PROFILING_FILEPATH`环境变量进行修改
    - 多卡运行时${cur_dir}/profiling目录下会有多个文件夹
  - 单卡Profiling文件在${cur_dir}/profiling/\*/PROF\*/mindstudio_profiler_output路径下，名称为msprof_\*.json
- Step 4：查看Profiling文件
  - 将msprof_\*.json文件拖拽入`chrome://tracing`网页中即可查看Profiling文件
- ${cur_dir}/profiling/\*/PROF\*/mindstudio_profiler_output路径下名为op_summary_\*.csv的文件中有单算子更加详细的性能数据
  - 开启`export ATB_PROFILING_TENSOR_ENABLE=1`环境变量后可以在此文件中查看算子执行时的shape信息（注意：开启此环境变量会导致CPU耗时增加，影响整体流水，但对单算子分析影响不大）
  - 获取shape信息需将Profiler Level设置为Level1，可以通过以下环境变量设置
    ```shell
    export PROFILING_LEVEL=Level1
    ```

#### 开启透明大页
- 若Profiling出现快慢进程，导致通信算子耗时长、波动大；模型整网性能测试，同一组测试样例，测试多次性能波动较大；此时可以尝试开启透明大页，提升内存访问性能。
- 开启方式
  ```shell
  echo always > /sys/kernel/mm/transparent_hugepage/enabled
  ```
- 关闭方式
  ```shell
  echo never > /sys/kernel/mm/transparent_hugepage/enabled
  ```
- 备注：
  - 仅ARM服务器适用
  - 指令请在裸机上执行

### 精度分析
#### Dump Tensor
- 适用于定位精度问题
- msit推理工具提供dump tensor以及精度对比的能力
  - 工具下载方式见[README](https://gitee.com/ascend/msit/blob/master/msit/docs/install/README.md)
  - 工具使用方式见[README](https://gitee.com/ascend/msit/blob/master/msit/docs/llm/%E5%B7%A5%E5%85%B7-DUMP%E5%8A%A0%E9%80%9F%E5%BA%93%E6%95%B0%E6%8D%AE%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

#### 溢出检查
- 可以开启以下环境变量，开启后若出现溢出，溢出值会被置为NaN；若不开启此变量，则会对溢出值进行截断
```shell
export INF_NAN_MODE_ENABLE=1
```
- 注意：Atlas 800I A2服务器若需要强制对溢出值进行截断（即`export INF_NAN_MODE_ENABLE=0`），需要额外将`INF_NAN_MODE_FORCE_DISABLE`环境变量设置为1。

## 特性支持矩阵
- 浮点特性

  MindIE LLM 主打高性能推理，考虑到内存开销，当前仅支持 float16 和 bfloat16 的浮点格式。可通过配置模型 config.json 中的 'torch_dtype' 字段进行类型修改。

  | 浮点特性 | 浮点能力 |
  | -------- | -------- |
  | float16 | √ |
  | bfloat16 | √ (仅Atlas 800I A2支持) |

- 量化特性

  MindIE LLM 提供多种量化选择进行推理加速，用户可根据自己的需要进行选择。

  | 量化特性 | per channel | per token | per group |
  | -------- | ----------- | --------- | --------- |
  | w8a8 | √ | × | × |
  | w8a16 | √ | × | × |
  | kv cache int8 | √ | × | × |
  | w8a8 稀疏量化 | √ | × | × |

  | 量化特性 | Atlas 800I A2 | Atlas 300I DUO |
  | -------- | ----------- | --------- |
  | w8a8 | √ | √ |
  | w8a16 | √ | × |
  | kv cache int8 | √ | × |
  | w8a8 稀疏量化 | × | √ |

- 并行特性

  MindIE LLM 提供 TP（Tensor Parallelism）和 EP（Expert Parallelism）两种并行策略。

  | 并行特性 | 并行能力 |
  | ------- | -------- |
  | TP | √ |
  | DP | × |
  | PP | × |
  | EP | √ |

- 长序列特性

  长序列特性的主要要求是对输入文本超长的场景下，模型回答的效果及性能也可以同时得到保障，其中输入的序列长度大多长于 32K，甚至可到达 1M 的级别。其主要的技术难点是对 attention 部分以及 kv cache 的使用方面的优化。在长序列场景下，由 attention 和 kv cache 部分造成的显存消耗会快速的成倍增长。因此对这部分现存的优化便是长序列特性的关键技术点。其中涉及到诸如 kv cache 量化，kv 多头压缩，训短推长等关键算法技术。

  最大序列长度的支持往往受限于多种因素，诸如机器的显存规格，模型的参数量等。若想在长序列场景下兼顾模型的回答效果，请兼顾模型支持的最大有效推理长度，以 llama1-65B 为例，其模型有效推理最大长度为 2048tokens，而部分经过训长推长的模型或训短推长的模型则可以保证在不使用额外算法优化的前提下获得长序列下保持推理精度的能力。具体规格请参考对应模型的官方介绍文档。长序列推理的使能方式与普通模型推理方式一致，仅需要将长序列文本按照正常推理流程传入模型即可。

- 多机特性

  针对于模型参数特别大的场景（如 175B 以上参数量模型）以及长序列场景，用户可选择使用多台服务器组合进行推理以达到较优效果，多机通信目前有两种硬件连接方式，PCIE 建联以及 RDMA 建联。使用多机推理时，需要将包含设备 ip，进程号信息，服务器 ip信息的 json 文件地址传递给底层通信算子。

  多机通信有多种并行策略可以结合，目前主流的并行测类包括 Tensor Parallel（TP，张量并行），Pipeline Parallel（PP，流水并行），Sequence Parallel（SP，序列并行）以及 MoE（Mixture of Experts）类模型特有的 Expert Parallel（EP，专家并行）。通过不同的并行策略的结合，可以最大化整体吞吐量，获得最优性能表现，目前模型默认使用纯TP并行方式执行推理，部分开源模型已经完成多机多卡推理能力构建，其中包括 llama 系列模型（包括基于 llama 实现的模型），deepseek-moe 以及 mixtral-moe。

## 预置模型列表
- [Aquila](./examples/models/aquila/README.md)
- [BaiChuan](./examples/models/baichuan/README.md)
- [bge-large](./examples/models/bge/large-zh-v1.5/README.md)
- [bge-reranker](./examples/models/bge/reranker-large/README.md)
- [BLOOM](./examples/models/bloom/README.md)
- [Bunny](./examples/models/bunny/README.md)
- [ChatGLM2](./examples/models/chatglm/v2_6b/README.md)
- [ChatGLM3](./examples/models/chatglm/v3_6b/README.md)
- [GLM4](./examples/models/chatglm/v4_9b/README.md)
- [Chinese Alpaca](./examples/models/chinese_alpaca/README.md)
- [CodeGeeX2](./examples/models/codegeex/v2_6b/README.md)
- [Codellama](./examples/models/codellama/README.md)
- [Codeshell](./examples/models/codeshell/README.md)
- [DeepSeek Coder](./examples/models/deepseek/README_DeepSeek_Coder.md)
- [DeepSeek V2](./examples/models/deepseekv2/README.md)
- [DeepSeek LLM](./examples/models/deepseek/README_deepseek_llm.md)
- [DeepSeek MOE](./examples/models/deepseek/README_deepseek_moe.md)
- [Gemma](./examples/models/gemma/README.md)
- [GLM4V](./examples/models/glm4v/README.md)
- [GPT2](./examples/models/gpt2/README.md)
- [GPT Neox](./examples/models/gpt_neox/README.md)
- [InternLM](./examples/models/internlm/README.md)
- [InternLM-XComposer2](./examples/models/internlmxcomposer2/README.md)
- [Internvl](./examples/models/internvl/README.md)
- [LLaMa1&2](./examples/models/llama/README.md)
- [LLaMa3](./examples/models/llama3/README.md)
- [LLava](./examples/models/llava/README.md)
- [MiniCPM](./examples/models/minicpm/README.md)
- [MiniCPM-Llama3](./examples/models/minicpm/README.md)
- [MiniCPMV2](./examples/models/minicpmv2/README.md)
- [MiniGPT4](./examples/models/minigpt4/README.md)
- [Mistral](./examples/models/mistral/README.md)
- [Mixtral](./examples/models/mixtral/README.md)
- [Phi3](./examples/models/phi3/README.md)
- [Qwen](./examples/models/qwen/README.md)
- [Qwen VL](./examples/models/qwen_vl/README.md)
- [StarCoder](./examples/models/starcoder/README.md)
- [StarCoder2](./examples/models/starcoder2/README.md)
- [Telechat](./examples/models/telechat/README.md)
- [Vicuna](./examples/models/vicuna/README.md)
- [Vlmo](./examples/models/vlmo/README.md)
- [Yi](./examples/models/yi/README.md)
- [Yi-VL](./examples/models/yivl/README.md)
- [zhinao](./examples/models/zhinao/README.md)
- [Ziya](./examples/models/ziya/README.md)


## 问题定位
- 若遇到推理执行报错，优先打开日志环境变量，并查看算子库和加速库的日志中是否有error级别的告警，基于error信息进一步定位
- 开启日志环境变量后，日志仍没有保存，定位排查点：
  - 1. 确认脚本中没有覆盖日志相关的环境变量
  - 2. 确认服务器内存充足
  - 3. 报错退出有可能导致日志来不及写入文件，可以尝试打开此环境变量`export ASDOPS_LOG_TO_FILE_FLUSH=1`
- 若遇到精度问题，可以dump tensor后进行定位

## 公网地址说明
- 代码涉及公网地址参考[此README文档](./public_address_statement.md)

## 约束
- 使用ATB Models进行推理，模型初始化失败时，请结束进程。
- 使用ATB Models进行推理，权重路径及文件的权限需保证其他用户无写权限