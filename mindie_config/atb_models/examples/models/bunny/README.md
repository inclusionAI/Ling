# README

- [Bunny](https://github.com/BAAI-DCAI/Bunny),是一系列轻量级但功能强大的多模态模型，具有强大的图像和文本处理能力。在聊天机器人中，Bunny可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，Bunny还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。
- 此代码仓中实现了一套基于NPU硬件的Bunny推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持 Bunny-v1_0-2B-zh，基于 SigLIP 视觉模型 + MLP + Qwen1.5-1.8B 文本模型的多模态推理。

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                    |
| ----------- |-----------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                       |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；bunny的工作脚本所在路径为 `${llm_path}/examples/models/bunny/`                                                           |
| weight_path | 模型权重路径                                                                                                                |
| image_path  | 图片所在路径                                                                                                                |
|open_clip_path| open_clip权重所在路径                                                                                                       |
## 权重

**权重下载**

- [Bunny-v1_0-2B-zh](https://huggingface.co/BAAI/Bunny-v1_0-2B-zh/tree/main)

**基础环境变量**

- 1.python 3.10, torch 2.1.0. Python其他第三方库依赖，参考[requirements_bunny.txt](../../../requirements/models/requirements_bunny_qwen2.txt)
- 2.参考[此README文件](../../../README.md)
- 注意：保证先后顺序，首先安装FrameworkPTAdapter中的2.1.0版本的pytorch和torch_npu，然后再安装其他的python依赖。

## 推理

### 对话测试

**运行Paged Attention FP16**

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} ${image_path}
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
    ```

## 精度测试

#### 方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载[open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，并把下载的权重放在open_clip_path目录下
   下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入{image_path}目录下
   

2. GPU上，在{script_path}/precision目录下，运行脚本python run_coco_gpu.py --model_path ${weight_path} --image_path ${image_path},会在{script_path}/precision目录下生成gpu_coco_predict.json文件存储gpu推理结果

3. NPU 上,在\${llm_path}目录下执行以下指令：
   ```bash
   bash ${script_path}/run_pa.sh --precision ${weight_path} ${image_path} 
   ```
   运行完成后会在{script_path}/precision生成predict_result.json文件存储npu的推理结果

4. 对结果进行评分：分别使用GPU和NPU推理得到的两组图片描述(gpu_coco_predict.json、predict_result.json)作为输入,执行clip_score_bunny.py 脚本输出评分结果
```bash
   python examples/models/bunny/precision/clip_score_bunny.py \ 
   --model_weights_path {open_clip_path}/open_clip_pytorch_model.bin \ 
   --image_info {gpu_coco_predict.json 或 predict_result.json的路径} \
   --dataset_path {iamge_path}
```

   得分高者精度更优。

## 性能测试

性能测试时需要在 `${image_path}` 下仅存放一张图片，使用以下命令运行 `run_pa.sh`，会自动输出batchsize为1-10时，输出token长度为 256时的吞吐。

```shell
bash ${script_path}/run_pa.sh --performance ${weight_path} ${image_path}
```

例如在 MindIE-ATB-Models 根目录，可以运行：

```shell
bash examples/models/bunny/run_pa.sh --performance ${weight_path} ${image_path}
```

可以在 `examples/models/bunny` 路径下找到测试结果。

## FAQ
- 在对话测试或者精度测试时，用户如果需要修改输入input_texts,max_batch_size时，可以修改{script_path}/run_pa.py里的参数，具体可见run_pa.py
- 更多环境变量见[此README文件](../../README.md)