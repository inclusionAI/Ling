# README

- Qwen-VL 是阿里云研发的大规模视觉语言模型（Large Vision Language Model, LVLM）。Qwen-VL 可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。



## 特性矩阵

- 此矩阵罗列了Qwen-VL模型支持的特性

| 模型及参数量  | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态 | 服务化支持模态 |
| ------------- | -------------------------- | --------------------------- | ---- | ---- | --------------- | --------------- | -------- |
| Qwen-VL  | 支持world size 1,2,4,8     | 支持world size 1,2,4,8            | √    | √    | √               | 文本、图片               | 文本、图片        |

## 路径变量解释

| 变量名      | 含义 |
| ---------- | ------- |
| llm_path   | 模型仓路径，若使用模型仓安装包，则该路径为安装包解压后的路径；若使用源码编译，则路径为 `MindIE-LLM/examples/atb_models` |
| model_path | 模型所在路径。`${llm_path}/examples/models/qwen_vl` |


## 权重

**权重下载**

- [Qwen-VL](https://modelscope.cn/models/qwen/Qwen-VL/files)

**基础环境变量**

- Toolkit, MindIE/ATB，ATB-SPEED等，参考[此README文件](../../../README.md)
- Python其他第三方库依赖，参考[requirements_qwen_vl.txt](../../../requirements/models/requirements_qwen_vl.txt)

## 推理

**运行Paged Attention FP16**

- 执行启动脚本
  
  在`${llm_path}`目录下, 修改run_pa.sh中`model_path`为本地权重路径，变量`image_path`为图片本地路径，并执行以下指令
  ```shell
  bash ${model_path}/run_pa.sh
  ```
  其他支持的推理参数请参考`${model_path}/run_pa.py`文件。

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

## 精度测试(v1系列)

### 测试方法
使用同样的一组图片与相同的文本输入，分别在GPU和NPU上执行推理，得到两组图片描述。再使用open_clip模型作为裁判，对两组结果分别进行评分，评分越高越好。

### 测试步骤

- 权重和图片下载
    - 下载open_clip的权重[open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)
    - 下载测试图片[CoCotest 数据集](https://cocodataset.org/#download)，随机抽取其中100张图片作为测试数据集
    - 安装open_clip仓库（众多github下载的库可以参照如下方式，快速安装）
    ```shell
    # 在命令行界面中手动克隆 open_clip 仓库，进入克隆下来的 open_clip 目录 pip 安装
    git clone https://github.com/mlfoundations/open_clip.git
    cd open_clip
    pip install -e .
    ```


- 推理得到两组图片描述
    - GPU推理：GPU推理不需要依赖昇腾环境，但推理脚本用到了模型仓的一些公共函数，用户可以直接将`${llm_path}`添加到`PYTHONPATH`环境变量，然后执行推理命令。运行成功后在执行目录下生成 gpu_coco_rst.json文件
    ```shell
    export PYTHONPATH=${llm_path}:$PYTHONPATH
    python run_coco_rst_gpu.py --model_path {qwenvl权重路径} --image_path {测试数据集路径}
    ```
    - NPU推理：执行推理脚本，运行成功后在执行目录下生成 npu_coco_rst.json文件, 注意参数(--dataset_path 测试数据集路径, input_texts 需要NPU与GPU保持一致，默认为 'Generate the caption in English with grounding:')
    ```shell
    bash examples/models/qwen_vl/run_pa.sh
    ```
   

- 评分

   分别使用GPU和NPU推理得到的两组图片描述(gpu_coco_rst.json、npu_coco_rst.json)作为输入,执行clip_score_qwenvl.py 脚本输出评分结果
   在`${llm_path}`目录下执行：
   ```bash
   python examples/models/qwen_vl/precision/clip_score_qwenvl.py \ 
   --model_weights_path {open_clip_pytorch_model.bin 的路径} \ 
   --image_info {gpu_coco_rst.json 或 npu_coco_rst.json 的路径} \
   --dataset_path {测试数据集路径}
   ```

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
- 将modeltest/config/model/qwen_vl.yaml中的model_path的值修改为模型权重的绝对路径，mm_model.warm_up_image_path改为textvqa数据集中任一图片的绝对路径。注意：trust_remote_code为可选参数代表是否信任本地的可执行文件，默认为false。若设置为true，则信任本地可执行文件，此时transformers会执行用户权重路径下的代码文件，这些代码文件的功能的安全性需由用户保证。
  ```yaml
  model_path: /data_mm/weights/Qwen-VL
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
  bash scripts/mm_run.sh textvqa qwen_vl
  ```
- 测试结果保存于以下路径。其下的results/..(一系列文件夹嵌套)/\*\_result.csv中存放着modeltest的测试结果。debug/..(一系列文件夹嵌套)/output\_\*.txt中存储着每一条数据的运行结果，第一项为output文本，第二项为输入infer函数的第一个参数的值，即模型输入。第三项为e2e_time。
  ```shell
  output/$DATE/modeltest/$MODEL_NAME/precision_result/
  ```