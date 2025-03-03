# README

- [Yi-VL](https://github.com/01-ai/Yi/tree/main/VL)，是大型语言模型Yi系列的开源多模态版本，支持如下特性：
    - 多轮文本-图像对话：可以同时将文本和图像作为输入，并产生文本输出。目前支持一图多轮视觉答题。
    - 双语文本支持：支持中英文对话，包括图片中的文字识别。
    - 强大的图像理解能力：擅长分析视觉信息，使其成为从图像中提取、组织和总结信息等任务的高效工具。
    - 更细粒度的图像分辨率：支持更高分辨率的448×448的图像理解。
- 此代码仓中实现了一套基于NPU硬件的Yi-VL推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

## 特性矩阵

Yi-VL模型支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
|-------------|----------------------------|-----------------------------|------|------------------|-----------------|-----|-----|
|  Yi-VL-6B   | 支持world size 1,2,4,8     | 支持world size 1,2,4,8        | √   |  √                   | √              | 文本、图片           | 文本、图片
|  Yi-VL-6B   | 支持world size 4,8     | 支持world size 4,8        | √   |  √                   |√              | 文本、图片           | 文本、图片
- 注：Yi-VL系列服务化暂不支持OpenAI格式请求
# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；yivl的工作脚本所在路径为 `${llm_path}/examples/models/yivl`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| image_path  | 图片所在路径    
| trust_remote_code  | 是否信任本地的可执行文件：默认不执行，传入此参数，则信任                                                                 |
| max_batch_size  | 最大bacth数                                                                  |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |
|open_clip_path| open_clip权重所在路径                                                           |

## 权重

**权重下载**

- [Yi-VL-6B](https://huggingface.co/01-ai/Yi-VL-6B)
- [Yi-VL-34B](https://huggingface.co/01-ai/Yi-VL-34B)


**基础环境变量**

- 参考[此README文件](../../../README.md)
- 确保python环境中的transformers版本>=4.36.2
- 跑精度评测需要安装`pip install open_clip_torch==2.20.0`
- 根据需要执行的推理模式来修改模型配置文件`${weight_path}/config.json`中的`torch_dtype`字段。如果需要执行FP16推理，那么将`torch_dtype`字段修改为`float16`；执行BF16推理，则将`torch_dtype`字段修改为`bfloat16`
- 300I DUO硬件不支持BF16推理，仅支持FP16推理

## 推理

### 对话测试

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run (--trust_remote_code) ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length} -trust_remote_code
    ```
  - trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 对于300I DUO卡而言，若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
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

### TextVQA

- 主要流程参考此[此README文件](../qwen2_vl/README.md)的`TextVQA`章节，
- 模型配置文件`${llm_path}/tests/modeltest/modeltest/config/model/yivl.yaml`设置如下
1. `model_path`的值修改为模型权重的绝对路径
2. `mm_model.warm_up_image_path`修改为用于warm_up的图像绝对路径
- 运行测试命令为
  ```shell
  bash mm_run.sh textvqa yivl
  ```

### CocoTest
#### 方案

使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，得分高者精度更优。

#### 实施

1. 下载[open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，并把下载的权重放在`${open_clip_path}`目录下
   下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入{image_path}目录下

2. GPU上，下载01-AI提供的[工程](https://github.com/01-ai/Yi/tree/main/VL/llava), 在`{llm_path}/examples/models/`目录下，运行如下脚本，得到gpu推理结果，存储在`{script_path}/gpu_coco_predict.json`文件
    ``` shell
    python coco_base_runner.py --model_path ${weight_path} --image_path ${image_path}
    ```

3. NPU 上,在\${script_path}目录下执行以下指令：
   ```bash
   bash ${script_path}/run_pa.sh --precision (--trust_remote_code) ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
   ```
   运行完成后会在{script_path}生成predict_result.json文件存储npu的推理结果

4. 对结果进行评分：分别使用GPU和NPU推理得到的两组图片描述(`gpu_coco_predict.json`, `predict_result.json`)作为输入,执行如下脚本脚本输出评分结果
    ```shell
    python {llm_path}/example/models/clip_score_base_runner.py \ 
    --model_name ViT-H-14
    --model_weights_path {open_clip_path}/open_clip_pytorch_model.bin \ 
    --image_info {gpu_coco_predict.json or predict_result.json} \
    --dataset_path {image_path}
    --device_ids {device_ids}
    ```

## 性能测试

- 运行前设置环境变量`export ATB_LLM_BENCHMARK_ENABLE=1`

- 在 `${image_path}` 下仅存放**1张**图片
    
- 以下命令运行 `run_pa.sh`，会自动输出batchsize为1-10时，输出token长度为256时的吞吐。
  ```shell
  bash ${script_path}/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path} ${max_batch_size} ${max_input_length} ${max_output_length}
  ```

- 测试结果保存在`{script_path}`路径下。

## FAQ
- 更多环境变量见[此README文件](../../README.md)