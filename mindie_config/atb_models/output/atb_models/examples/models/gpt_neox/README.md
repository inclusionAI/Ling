# README

- GPT-NeoX-20B 是一个 200 亿参数的自回归语言模型，使用 GPT-NeoX 库在 Pile 上训练。它的架构有意类似于 GPT-3，并且与 GPT-J-6B 的架构几乎相同。其训练数据集包含大量英语文本，反映了该模型的通用性质。
- 此代码仓中实现了一套基于NPU硬件的GPT-NEOX-20B推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
# 特性矩阵

- 此矩阵罗列了GPT-NEOX-20B模型支持的特性

| 模型及参数量           | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI  | 长序列 |
| ---------------------- |----------------------|---------------------------| ---- |-----| --------------- | --------------- | -------- | --------- | --------- | ------------ | -------------------------- | ---- | ------ | ---- |
| GPT-NEOX-20B           | 支持world size 2,4,8   | 支持world size 2, 4        | √    | ×   | √               | √               | ×        | ×             | ×            | ×                          | ×    | ×      | ×    | ×    |

# Paged Attention 推理使用说明

## 路径变量解释

| 变量名         | 含义                                                                                                                  |
|-------------|---------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                     |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径。GPT-NEOX系列模型的工作脚本所在路径为${llm_path}/examples/models/gpt_neox                                                   |
| weight_path | 模型权重路径                                                                                                              |

# 权重下载

- [GPTNeoX-20B](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)
## 权重转换

- 通常情况 Paged Attention 场景下需要.safetensors 格式的权重，如果没有，参考[此README文件](../../README.md)转换
- 但这里我们的GPTNeoX本身就是safetensors 格式权重，PA场景下无需转换。
- 注: huggingface上有safetensors类型权重可直接下载

## 量化权重生成
- 暂不支持
## 操作说明

### 推理

##### 对话测试
- 运行Paged Attention FP16
- 在`${llm_path}`目录下执行以下脚本

```shell
bash examples/models/gpt_neox/run_pa.sh ${weight_path}
```

根据硬件设备不同请参考下表修改run_pa.sh再运行

### run_pa.sh 参数说明

| 参数名称                      | 含义                                  | 800I A2推荐值 | 300I DUO推荐值 |
|---------------------------|-------------------------------------|------------|-------------|
| BIND_CPU                  | 绑定CPU核心开关,默认进行绑核                    | 1          | 1           |
| IS_QUANT                  | 是否启动量化                              | 0          | 0           |
| ASCEND_RT_VISIBLE_DEVICES | 使用的硬件卡号，多个卡间使用逗号相连                  | 根据实际情况设置   | 根据实际情况设置    |
| MASTER_PORT               | 卡间通信端口,通常不用修改，有冲突时再改                |            |             |

### 运行Paged Attention BF16
- 暂不支持
### 运行Paged Attention W8A16
- 暂不支持
### 运行Paged Attention W8A8
- 暂不支持
### 运行Paged Attention BF16
- 暂不支持
### 运行KV cache量化
- 暂不支持
### 运行稀疏量化
- 暂不支持
### 运行MOE量化
- 暂不支持
### 精度测试

- 参考[此README文件](../../../tests/modeltest/README.md)
- 300I DUO服务器上用2卡4芯跑，800I A2服务器上用8卡跑
- GPT-NEOX-20B 上下文长度只支持2048，精度测试需要修改模型权重路径下的 配置文件config.json 中的 "max_position_embeddings" 字段，由 2048 改为 4096 才能运行boolq精度测试。
- 示例
 ```shell
 cd ${llm_path}/tests/modeltest
 export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 bash run.sh pa_fp16 full_BoolQ 1 gptneox_20b ${gptneox-20b权重路径} 8
 ```
注：GPTNeoX为纯英文模型，所以一般只测试BoolQ英文测试集，对于Ceval中文集我们不做测试

## 性能测试

- 参考[此README文件](../../../tests/modeltest/README.md)
- 300I DUO服务器上用2卡4芯跑，800I A2服务器上用8卡跑
- 示例
 ```shell
 cd ${llm_path}/tests/modeltest
 export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 bash run.sh pa_fp16 performance [[2048,2048],[1024,1024],[512,512],[256,256]] 1 gptneox_20b ${gptneox-20b权重路径} 8
 ```

# Flash Attention推理使用说明

#### 张量并行模型切分（仅在模型需要多卡并行时使用）

```shell
cp ${script_path}/modeling_gpt_neox_ascend.py ${model_path}
```

修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_gpt_neox_ascend.GPTNeoXForCausalLM"`

```text
修改`${script_path}/cut_model_and_run.sh`    
将 `input_dir` 修改为模型所在路径 `${model_path}` 
将 `output_dir` 修改为原目录下子目录 `${model_path/part_model}`。模型切分成功后，会自动生成新目录part_model(用户无需新建该文件夹)
将 `world_size_` 修改为期望切分的份数。world_size_=2表示模型切分为2份。

```

目录结构示例建议

```
--model_path
  *.py(模型源文件)
  *.json(模型源文件)
  *.tiktoken(模型源文件)
  *.bin(模型源文件，软链接，部分模型权重为其它格式，如*.safetensors等)
  modeling_gpt_neox_ascend.py(加速库modeling)
  configuration_gpt_neox.py(模型配置文件)
  --part_model(以双卡为例，权重切分成功后文件夹)
    --0
    --1
  ......(其他)
--script_path
  cut_model_and_run.sh
  cut_model_util.py
  main.py
  config.ini
  ......(其他)
```

执行

```shell
cd ${script_path}
bash cut_model_and_run.sh
```

切分所需时间较长，切分完成后，将会打印 'Tensor parallelism weights have been successfully saved.'。

确认${model_path}/part_model/{rank_id}里的config.json中的kv对为

```
"AutoModelForCausalLM": "modeling_gpt_neox_ascend.GPTNeoXForCausalLM"
```
再次执行进行推理
```shell
bash cut_model_and_run.sh ${task_name}
```
- task_name 可选 inference、precision、performance

# CPU高性能模式

可开启CPU Performance模式以提高模型推理性能。

```

cpupower frequency-set -g performance

```

### 执行推理

#### 修改 ${script_path}/config.ini

[config文件配置参考](../atb_speed_sdk/README.md)  
提示：多卡并行推理时，config.ini中model_path路径为part_model父文件夹。例如：

```
# 正确示例：

model_path=../model

# 错误示例：

model_path=../model/part_model
```

#### main.py

提供了demo推理，precision测试，性能测试三种下游任务。  
task_name可选inference、precision、performance。

- 单卡
  修改 ${model_path}里的config.json中的kv对，改成`"AutoModelForCausalLM": "modeling_gpt_neox_ascend.GPTNeoXForCausalLM"`

```shell
python main.py --task ${task_name}
```

注意，由于本模型体量较大，受硬件限制，单卡很可能无法跑起。

- 多卡

```shell
bash cut_model_and_run.sh ${task_name}
```

**注意**
1.docker环境与conda环境有所不同，docker环境中启动模型时需要修改环境变量"ATB_OPERATION_EXECUTE_ASYNC=0"、"TASK_QUEUE_ENABLE=0"，否则可能出现算子下发同步失败。
2.300l DUO暂时不支持lccl，因此在300l DUO上启动模型时需删去环境变量"BACKEND='lccl'"

如

```shell
python main.py --task ${task_name}
```

或

```shell
bash cut_model_and_run.sh ${task_name}
```

如果遇到

```text
Traceback (most recent call last):
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/__init__.py", line 31, in <module>
    import torch_npu.npu
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/__init__.py", line 46, in <module>
    from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/utils.py", line 27, in <module>
    import torch_npu._C
ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block
Segmentation fault (core dumped)
```

则在命令行前加上`LD_PRELOAD=上面的error路径`。如

```shell
LD_PRELOAD=/root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1 python main.py --task ${task_name}  --is_quant ${is_quant}
```

# 附录：

# 精度测试指南

由于gpt-neox是英文模型，一般选用boolq数据集进行精度测试，不过下面也介绍一下使用MMLU数据集进行精度测试。

## 配置说明

安装atb_speed_sdk，参考 [SDK精度测试指南章节](../atb_speed_sdk/README.md)

```shell
cd examples/models/atb_speed_sdk
pip install .
```

## 老版本FA运行脚本

- 单芯

```shell
cd ${script_path}
python main.py --task precision
```

- 多芯  


```shell
cd ${script_path}
bash cut_model_and_run.sh precision
```

结束后在${mmlu_work_dir}/test_result目录下查看测试结果。[双芯结果每个两份，只需看其中一份即可]。

| 文件                        | 用途                   | 
|---------------------------|----------------------| 
| device0.log               | 运行过程日志               |
| cache0.csv                | 结果详情，C列为预期答案，D列为测试答案 |
| result_0_classes_acc.json | 测试数据下按不同维度统计准确率      |
| result_0_subject_acc.json | 测试数据下按不同学科统计准确率      |

**注意：后续重新运行， 需要删除当前目录下生成的test_result文件夹，否则只会读取当前的目录下的测试结果**

# FA性能测试

在功能运行正常的基础下，执行以下步骤进行性能测试

## 按照推理指导,下载模型及配置路径，并安装atb_speed_sdk

## 1. 准备

参考 [SDK性能测试指南精确打点法章节](../atb_speed_sdk/README.md) 进行准备

## 2. 修改配置文件

- 配置config.ini中[performance]属性， 如下：
  ```
  model_name=gpt_neox_20b
  perf_mode=detail
  ```

## 3. 执行测试脚本

- 单芯

```shell
cd ${script_path}
TIMEIT=1 python main.py --task performance
```

- 多芯  
  多卡推理，芯片类型区分为300l DUO、800l A2系列。当在800l A2芯片进行多卡推理时，"cut_model_and_run.sh"脚本需修改环境变量"ATB_USE_TILING_COPY_STREAM=0"。
该环境变量功能是为了解决300l DUO上asynccopy性能慢的问题，与800l A2无关。

```shell
cd ${script_path}
TIMEIT=1 bash cut_model_and_run.sh performance
```

为了不影响正常使用，将`TIMEIT`设置成1来返回具体的性能测试的值，默认是0

### FA性能测试结果

得到性能测试结果csv `performance_test_npu_${model_name}_xxx.csv`

### 结果分析

| 列名                            | 含义         |
|-------------------------------|------------|
| batch_size                    | batch大小    |
| input_seq_len(Encoding)       | 输入长度       |
| output_seq_len(Decoding)	     | 输出长度       |
| ResponseTime(s)	              | 总响应时间      |
| forward_first_token_time(ms)  | 首token推理时长 |
| forward_next_token_time(ms)   | 增量推理时长     |
| pre_next_token_time(ms)	      | 前处理时长      |
| post_next_token_time_post(ms) | 后处理时长      |
