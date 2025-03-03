# README

- [InternVL](https://github.com/OpenGVLab/InternVL)，是一种多模态大模型，具有强大的图像和文本处理能力，通过开源组件缩小与商业多模态模型的差距——GPT-4V的开源替代方案。在聊天机器人中，InternVL可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，InternVL还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。
- 此代码仓中实现了一套基于NPU硬件的InternVL推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持 InternVL-Chat-V1-2，基于 InternViT-6B-448px-V1-2 视觉模型 + MLP + Nous-Hermes-2-Yi-34B 文本模型的多模态推理。
- 支持 InternVL-Chat-V1-5，基于 InternViT-6B-448px-V1-5 视觉模型 + MLP + InternLM2-Chat-20B 文本模型的多模态推理。
- 支持 InternVL2-2B，基于 InternViT-6B 视觉模型 + MLP + InternLM2-Chat-1.8B 文本模型的多模态推理。
- 支持 InternVL2-8B，基于 InternViT-6B 视觉模型 + MLP + InternLM2.5-7B-Chat 文本模型的多模态推理。
- 支持 InternVL2-40B，基于 InternViT-6B 视觉模型 + MLP + Nous-Hermes-2-Yi-34B 文本模型的多模态推理。

## 特性矩阵

- 此矩阵罗列了各InternVL模型支持的特性

| 模型及参数量  | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
| ------------- |--------------------------|---------------------------| ---- | ---- | --------------- | --------------- | -------- |
| InternVL-Chat-V1-2  | 支持world size 4,8         | 支持world size 4,8          | √    | √    | √               | 文本、图片               | 文本、图片        |
| InternVL-Chat-V1-5 | 支持world size 4,8         | 支持world size 4,8          | √    | ×    | √               | 文本、图片               | 文本、图片        |
| InternVL2-2B | 支持world size 1,2,4,8     | 支持world size 1,2,4,8      | √    | ×    | √               | 文本、图片               | 文本、图片        |
| InternVL2-8B | 支持world size 1,2,4,8     | 支持world size 1,2,4,8      | √    | ×    | √               | 文本、图片               | 文本、图片        | 
| InternVL2-40B | 支持world size 4,8         | 支持world size 4,8          | √    | √    | √               | 文本、图片               | 文本、图片        |

注意：
- 当前多模态场景, MindIE Service仅支持MindIE Service、TGI、Triton、vLLM Generate 4种服务化请求格式。
- 表中所示支持的world size为建议配置，实际运行时还需考虑单卡的显存上限，以及输入序列长度。
- 推理默认加载BF16权重，如运行特性矩阵中不支持BF16的模型，请将权重路径下config.json文件的`torch_dtype`字段修改为`float16`。
- MindIE Service表示模型支持MindIE服务化部署，多卡服务化推理场景。
- 若需要在同一环境下拉起多个 MindIE Service 服务，需要设置以下环境变量，确保每个服务之间的 MASTER_PORT 不冲突。该变量的取值范围通常为 [1024, 65535]，建议动态分配一个未占用的端口。
  ```shell
  export MASTER_ADDR=localhost
  export MASTER_PORT=<动态分配的未占用端口号>
  ```
  提示：可以通过以下命令检查某个端口是否被占用：
  ```shell
  netstat -anp | grep <PORT>
  ```

## 路径变量解释

| 变量名      | 含义                                                                                                                    |
| ----------- |-----------------------------------------------------------------------------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                                                                                                       |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；internvl的工作脚本所在路径为 `${llm_path}/examples/models/internvl`                                                      |
| weight_path | 模型权重路径                                                                                                                |
| trust_remote_code | 是否信任本地的可执行文件：默认不执行，传入此参数，则信任                                                                  |
| image_path  | 图片所在路径                                                                                                                |
|open_clip_path| open_clip权重所在路径                                                                                                       |
## 权重

**权重下载**

- [InternVL-Chat-V1-2](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2/tree/main)
- [InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/tree/main)
- [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B/tree/main)
- [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B/tree/main)
- [InternVL2-40B](https://huggingface.co/OpenGVLab/InternVL2-40B/tree/main)

**基础环境变量**

- 1.python 3.10, torch 2.1.0. Python其他第三方库依赖，参考[requirements_internvl.txt](../../../requirements/models/requirements_internvl.txt)
- 2.参考[此README文件](../../../README.md)
- 注意：保证先后顺序，首先安装FrameworkPTAdapter中的2.1.0版本的pytorch和torch_npu，然后再安装其他的python依赖。

## 推理

### 注意事项

- 若使用Atlas 300I DUO卡，**在batchsize > 10的情况下，请使用4卡8芯的机器运行，并确保至少有8个可见NPU核心(TP8)**，否则会造成显存溢出

### 对话测试

**运行Paged Attention FP16**

- 运行启动脚本
  - 在`${llm_path}`目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run (--trust_remote_code) ${weight_path} ${image_path}
    ```
- 环境变量说明 (以下为run_pa.sh启动脚本中配置的环境变量)
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

## 精度测试(v1系列)

#### 方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载[open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，并把下载的权重放在open_clip_path目录下
   下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入`${image_path}`目录下

2. GPU上，在`${script_path}/precision`目录下，运行脚本`python run_coco_gpu.py --model_path ${weight_path} --image_path ${image_path}`,会在`${script_path}/precision`目录下生成gpu_coco_predict.json文件存储gpu推理结果

3. NPU 上,在`${llm_path}`目录下执行以下指令：

   ```bash
   bash ${script_path}/run_pa.sh --precision (--trust_remote_code) ${weight_path} ${image_path} 
   ```
   
   运行完成后会在{script_path}/precision生成predict_result.json文件存储npu的推理结果

4. 对结果进行评分：分别使用GPU和NPU推理得到的两组图片描述(gpu_coco_predict.json、predict_result.json)作为输入,执行clip_score_internvl.py 脚本输出评分结果
   
   ```bash
   python examples/models/internvl/precision/clip_score_internvl.py \ 
   --model_weights_path ${open_clip_path}/open_clip_pytorch_model.bin \ 
   --image_info {gpu_coco_predict.json 或 predict_result.json的路径} \
   --dataset_path ${image_path}
   ```

   得分高者精度更优。

## 精度测试(v2系列)
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
   - 若下载有SSL相关报错，可在命令后加上'-i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com'参数使用阿里源进行下载
- 场景一：运行Internvl2-8B。将modeltest/config/model/internvl.yaml中的model_path的值修改为模型权重的绝对路径，mm_model.warm_up_image_path改为textvqa数据集中任一图片的绝对路径。注意：trust_remote_code为可选参数代表是否信任本地的可执行文件，默认为false。若设置为true，则信任本地可执行文件，此时transformers会执行用户权重路径下的代码文件，这些代码文件的功能的安全性需由用户保证。
  ```yaml
  model_path: /data_mm/weights/InternVL2-8B 
  trust_remote_code: {用户输入的trust_remote_code值}
  mm_model:
    warm_up_image_path: ['/data_mm/datasets/textvqa_val/train_images/003a8ae2ef43b901.jpg']
  ```
- 场景二：运行Internvl2-40B。将modeltest/config/model/internvl.yaml中的model_path的值修改为模型权重的绝对路径，mm_model.warm_up_image_path改为textvqa数据集中任一图片的绝对路径。注意：trust_remote_code为可选参数代表是否信任本地的可执行文件，默认为false。若设置为true，则信任本地可执行文件，此时transformers会执行用户权重路径下的代码文件，这些代码文件的功能的安全性需由用户保证。
  ```yaml
  model_path: /data_mm/weights/InternVL2-40B 
  trust_remote_code: {用户输入的trust_remote_code值}
  mm_model:
    warm_up_image_path: ['/data_mm/datasets/textvqa_val/train_images/003a8ae2ef43b901.jpg']
  ```
- 将modeltest/config/task/textvqa.yaml中的model_path修改为textvqa_val.jsonl文件的绝对路径，以及将requested_max_input_length和requested_max_output_length的值分别改为20000和256
  ```yaml
  local_dataset_path: /data_mm/datasets/textvqa_val/textvqa_val.jsonl
  requested_max_input_length: 20000
  requested_max_output_length: 256
  ```
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
- 设置可见卡数，修改mm_run.sh文件中的ASCEND_RT_VISIBLE_DEVICES。依需求设置单卡或多卡可见。
  ```shell
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  ```
- 运行测试命令
  ```shell
  bash scripts/mm_run.sh textvqa internvl
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```
  

## 性能测试

_性能测试时需要在 `${image_path}` 下仅存放一张图片_

测试模型侧性能数据，需要开启环境变量
  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
  ```
**在${llm_path}目录使用以下命令运行 `run_pa.sh`**，会自动输出batchsize为1-10时，输出token长度为 256时的吞吐。

```shell
bash examples/models/internvl/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path}
```

可以在 `${script_path}` 路径下找到测试结果。

## FAQ
- 在对话测试或者精度测试时，用户如果需要修改输入input_texts,max_batch_size时，可以修改`${script_path}/internvl.py`里的参数，具体可见internvl.py
- 更多环境变量见[此README文件](../../README.md)