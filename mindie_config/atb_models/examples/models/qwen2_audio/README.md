# README

- [Qwen2-Audio-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct)，Qwen2-Audio可以识别讲话者的情绪，判断音乐的节奏和类型，分辨各种环境声音，甚至能理解混合音频的含义，例如从一段包含警报声、刹车声和引擎声的音频中，推测出可能是交通事故现场。Qwen2-Audio能够接受各种音频信号输入，对语音指令进行音频分析或直接文本回复。


# 特性矩阵
| 模型及参数量    | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态  | 服务化支持模态 |
| --------------- | -------------------------- | -------------------------- | ---- | ------------ | -------------- | -------------- | ------------ |
| Qwen2-Audio-7B-Instruct | 支持world size 1,2,4,8   | 暂不支持          | √    | √            | √              | 文本、音频      | 文本、音频 |


## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；qwen2_audio的工作脚本所在路径为 `${llm_path}/examples/models/qwen2_audio`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| audio_path  | 音频所在路径                                                                     |
| max_batch_size  | 最大bacth数                                                                  |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |


-注意：
max_input_length长度设置可参考模型权重路径下config.json里的max_position_embeddings参数值
## 权重

**权重下载**

- [Qwen2-Audio-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct)


**基础环境变量**

- Python其他第三方库依赖，参考[requirements_qwen2_audio.txt](../../../requirements/models/requirements_qwen2_audio.txt)
- 参考[此README文件](../../../README.md)
- 注意：保证先后顺序，否则qwen2_audio的其余三方依赖会重新安装torch，导致出现别的错误
- transformers版本等于4.45.2
- 模型运行依赖ffmpeg，需要确保安装后运行。安装可参考以下指令
    ```shell
    apt-get update -y && apt-get install ffmpeg
    ```
**服务化推理测试**
- Python第三方库依赖安装完毕之后，在测试服务化之前，需要在\${script_path}目录下执行以下指令
    ```shell
    python librosa_file_util.py --modify_librosa True
    ```
用于修改librosa，保证服务化能正常推理。
- 在所有服务化推理结束之后，可在\${script_path}目录下执行以下指令
    ```shell
    python librosa_file_util.py --restore_librosa True
    ```
将librosa改回至原文件状态。
- 执行以上指令，如果用户权限不够，请切换至root用户执行。

## 推理

### 对话测试


- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} ${audio_path}
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
    export LCCL_ENABLE_FALLBACK=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```

## 精度测试
使用modeltest进行纯模型在VocalSound数据集上的精度测试
- 数据准备
  - 数据集下载 [vocalsound](https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset)  
- 设置环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh 
  source ${llm_path}/set_env.sh 
  ```
- 进入以下目录 MindIE-LLM/examples/atb_models/tests/modeltest
  ```shell
  cd MindIE-LLM/examples/atb_models/tests/modeltest
  ```
- 安装modeltest及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将modeltest/config/model/qwen2_audio.yaml中的model_path的值修改为模型权重的绝对路径。将mm_model.warm_up_audio_path的值改为Vocalsound数据集中任一音频的绝对路径。
  ```yaml
  model_path: /data_mm/weights/Qwen2-Audio-7B-Instruct
  mm_model:
  warm_up_audio_path: ['/data_mm/datasets/vs_release_16k/audio_16k/f0003_0_cough.wav']
  ```
- 将modeltest/config/task/vocalsound.yaml中的model_path修改为subset_npu_gpu_16k文件夹的绝对路径
  ```yaml
  local_dataset_path: /data_mm/datasets/vocalsound_all/subset_npu_gpu_16k
  ```
- 设置可见卡数，修改mm_run.sh文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh vocalsound qwen2_audio
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```
  

## 性能测试
- 在\${llm_path}目录下执行以下指令
- 测试模型侧性能数据，开启环境变量
  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
  ```
- 设置 `max_output_length` 为一个合理的值，确保实际输出文本长度 >= `max_output_length`
- 性能测试时需要在 `${audio_path}` 下仅存放一个音频文件，使用以下命令运行 `run_pa.sh`，会自动输出batchsize为1,2,4,8,16时的吞吐。

```shell
bash ${script_path}/run_pa.sh --performance ${weight_path} ${audio_path} ${max_batch_size} ${max_input_length} ${max_output_length}
```
- 执行推理脚本，性能数据在${script_path}中查看qwen2_audio_performance.csv文件。
