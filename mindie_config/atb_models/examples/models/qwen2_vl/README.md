# README

- Qwen2-VL 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen2-VL 可以以图像、文本、视频作为输入，并以文本作为输出。

## 特性矩阵

| 模型及参数量       | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态  | 服务化支持模态  |
|--------------|----------------------------|-----------------------------|------|--------------|----------------|----------|----------|
| Qwen2-VL-2B-Instruct  | 支持world size 1,2,4,8         | 支持world size 1,2,4,8                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-7B-Instruct  | 支持world size 1,2,4,8         | 支持world size 1,2,4,8                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |
| Qwen2-VL-72B-Instruct | 支持world size 4,8           | 支持world size 4,8                        | √    | √            | √              | 文本、图片、视频 | 文本、图片、视频 |

注意：

- 表中所示支持的world size为建议配置，实际运行时还需考虑单卡的显存上限，以及输入序列长度。
- 推理默认加载BF16权重，如需调整为FP16请将权重路径下config.json文件的`torch_dtype`字段修改为`float16`。
- MindIE Service表示模型支持MindIE服务化部署，多卡服务化推理场景，需要设置环境变量，MASTER_PORT确保未被占用。
  ```shell
  export MASTER_ADDR=localhost
  export MASTER_PORT=5678
  ```
- 服务化以及纯模型场景下处理视频默认FPS=0.5，即两秒抽一帧。
- 当前版本不支持传入多帧图片格式的视频。

## 路径变量解释

| 变量名        | 含义                                                                           |
|------------|------------------------------------------------------------------------------|
| llm_path   | 模型仓路径，若使用模型仓安装包，则该路径为安装包解压后的路径；若使用源码编译，则路径为 `MindIE-LLM/examples/atb_models`  |

## 推理
**权重下载**

- [Qwen2-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/files)
- [Qwen2-VL-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-7B-Instruct/files)
- [Qwen2-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-VL-72B-Instruct/files)

**安装依赖**

- Toolkit, MindIE/ATB, ATB-SPEED等，参考[此README文件](../../../README.md)
- 安装Python其他第三方库依赖，参考[requirements_qwen2_vl.txt](../../../requirements/models/requirements_qwen2_vl.txt)，注意 transformers == 4.46.0
  ```shell
  pip install -r ${llm_path}/requirements/models/requirements_qwen2_vl.txt
  ```

**环境变量说明**
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
  - 以下环境变量与性能和内存优化相关，通常情况下无需修改，详细信息可参考ATB官方文档
    ```shell
    export ATB_LAUNCH_KERNEL_WITH_TILING=1
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048'
    export HCCL_BUFFSIZE=120
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    ```
**运行**
- 修改启动脚本 `${llm_path}/examples/models/qwen2_vl/run_pa.sh`

    - 修改启动脚本中 `model_path` 为本地权重路径。
    - 修改启动脚本中 `max_batch_size` 为 Batch Size。
    - 修改启动脚本中 `max_input_length` 为最大输入长度（需考虑输入图片的分辨率以及视频的长度）。
    - 修改启动脚本中 `max_output_length` 为最大输出长度。
    - 修改启动脚本中 `input_image` 为图片或者视频的本地文件路径（或者修改 `dataset_path` 为数据集的本地文件夹路径）。
    - 修改启动脚本中 `input_text` 为输入prompt。
    - 修改启动脚本中 `shm_name_save_path` 为共享内存的保存路径
- 执行启动脚本 `${llm_path}/examples/models/qwen2_vl/run_pa.sh`
    ```shell
    bash ${llm_path}/examples/models/qwen2_vl/run_pa.sh
    ```
- 其他支持的推理参数请参考 `${llm_path}/examples/models/qwen2_vl/run_pa.py` 文件。

## 精度测试
### TextVQA
使用modeltest进行纯模型在TextVQA数据集上的精度测试
- 数据准备
    - 数据集下载 [textvqa](https://huggingface.co/datasets/maoxx241/textvqa_subset)
    - 保证textvqa_val.jsonl和textvqa_val_annotations.json在同一目录下
    - 将textvqa_val.jsonl文件中所有"image"属性的值改为相应图片的绝对路径
  ```json
  ...
  {
    "image": "/data/textvqa/train_images/003a8ae2ef43b901.jpg",
    "question": "what is the brand of this camera?",
    "question_id": 34602,
    "answer": "dakota"
  }
  ...
  ```
- 设置环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh 
  source ${llm_path}/set_env.sh 
  ```
- 进入以下目录 `${llm_path}/tests/modeltest`
  ```shell
  cd ${llm_path}/tests/modeltest
  ```
- 安装modeltest及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将 `modeltest/config/model/qwen2_vl.yaml` 中的model_path的值修改为模型权重的绝对路径
  ```yaml
  model_path: /data_mm/weights/Qwen2-VL-7B-Instruct
  ```
- 将 `modeltest/config/task/textvqa.yaml` 中的model_path修改为textvqa_val.jsonl文件的绝对路径
  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  ```
- 设置可见卡数，修改 `scripts/mm_run.sh` 文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh textvqa qwen2_vl
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```
### VideoBench
使用modeltest进行纯模型在VideoBench数据集上的精度测试
- 数据准备
  - 数据集下载 [Eval_QA](https://huggingface.co/datasets/maoxx241/videobench_subset) && [Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)
  - 将Eval_QA/目录下的各json文件中的vid_path改为相应图片的绝对路径
  ```json
  ...
  "v_C7yd6yEkxXE_4": {
    "vid_path": "/data_mm/Eval_video/ActivityNet/v_C7yd6yEkxXE.mp4",
  }
  ...
  ```
- 设置环境变量
  ```shell
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  source /usr/local/Ascend/nnal/atb/set_env.sh 
  source ${llm_path}/set_env.sh 
  ```
- 进入以下目录 `${llm_path}/tests/modeltest`
  ```shell
  cd ${llm_path}/tests/modeltest
  ```
- 安装modeltest及其三方依赖
  ```shell
  pip install --upgrade pip
  pip install -e .
  pip install tabulate termcolor 
  ```
- 将 `modeltest/config/model/qwen2_vl.yaml` 中的model_path的值修改为模型权重的绝对路径
  ```yaml
  model_path: /data_mm/weights/Qwen2-VL-7B-Instruct
  ```
- 将 `modeltest/config/task/videobench.yaml` 中的model_path修改为Video-Bench-main/Eval_QA文件的绝对路径，查看EVAL_QA文件夹下的json文件，将subject_mapping中不涉及测试的视频子数据集注释掉（可自行调整），样例如下：
  ```yaml
  local_dataset_path: /data_mm/datasets/VideoBench/Video-Bench-main/Eval_QA
  ...
  subject_mapping:
  # ActivityNet:
  #  name: ActivityNet
  Driving-decision-making:
    name: Driving-decision-making
  ```
- 设置可见卡数，修改 `scripts/mm_run.sh` 文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh videobench qwen2_vl
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```

## 性能测试

- 测试模型侧性能数据，开启环境变量
  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  ```
- 设置 `max_output_length` 为一个合理的值，确保实际输出文本长度 >= `max_output_length`
- 执行`${llm_path}/examples/models/qwen2_vl/run_pa.sh`推理脚本，查看终端输出的性能数据。