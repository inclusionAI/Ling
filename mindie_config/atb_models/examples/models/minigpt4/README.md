# MiniGPT-4

## 目录

- [概述](#概述)
- [特性矩阵](#特性矩阵)
- [环境准备](#环境准备)
    - [路径变量解释](#路径变量解释)
    - [python环境准备](#python环境准备)
    - [其他依赖](#其他依赖)
- [模型文件（源码与权重）准备](#模型文件（源码与权重）准备)
    - [模型文件（源码与权重）下载，以及相应的配置修改](#模型文件（源码与权重）下载，以及相应的配置修改)
    - [图像处理部分的 om 转换与其他的源码修改](#图像处理部分的 om 转换与其他的源码修改)
- [基本推理](#基本推理)
- [测试](#测试)
    - [图像处理时间测试](#图像处理时间测试)
    - [精度测试](#精度测试)
    - [性能测试](#性能测试)
- [附录](#附录)
    - [图像处理部分的om转换](#图像处理部分的om转换)
    - [对源码的其他必要修改](#对源码的其他必要修改)
    - [附加说明](#附加说明)

## 概述

MiniGPT-4 是兼具语言与图像理解能力的多模态模型，使用了先进的大语言模型强化了机器的视觉理解能力。
具体来说，它结合了大语言模型 Vicuna 和视觉编码器 BLIP-2，具备强大的新型视觉语言能力。

## 特性矩阵

- 此矩阵罗列了 minigpt4 模型支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|----------------------------|----------------------------|------|------------------|-----------------|-----------------|---------|-----------|---------|-----------|--------------------------|-----|--------|---|
| minigpt4-7B   | 支持world size 1,2,4,8     | 支持world size 2,4           | √   | ×                    | √              | ×              | ×       | ×          | ×      | ×                        | ×   | ×      | ×   | ×     |

- 此模型仓已适配的模型版本
    - [MiniGPT-4 GitHub仓](https://github.com/Vision-CAIR/MiniGPT-4)

## 环境准备

### 路径变量解释

| 变量名         | 含义                    |  
|-------------|-----------------------|
| work_space  | 主工作目录                 |
| model_path  | 开源权重等必要材料放置在此目录       | 

### python环境准备

参见 `../../../requirements/models/requirements_minigpt4.txt`

```bash
pip install -r requirements_minigpt4.txt
```

此外，还需要安装 `aclruntime` 和 `ais_bench` 这两个三方件（为了支持 om 格式的模型）。请参考
[这里](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench) ，下载并安装。

### 其他依赖

其他依赖具备一般性，请参考[此README文件](../../../README.md)

## 模型文件（源码与权重）准备
本教程的权重文件和线性层是基于vicuna-7b的，也可以使用Llama2-7B的配套权重。

### 模型文件（源码与权重）下载，以及相应的配置修改

1. 下载 [MiniGPT-4 的源码](https://github.com/Vision-CAIR/MiniGPT-4)。

   下载所得的目录 `MiniGPT-4-main` 即为主工作目录 `${work_space}`。

2. 下载
   [MiniGPT-4 线性层的权重 pretrained_minigpt4_7b.pth](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing)
   。

   下载完成后，保存到路径`${model_path}/weights_linear/`下。

   须修改配置文件`${work_space}/eval_configs/minigpt4_eval.yaml`中关于此路径的配置。

   line 8
      ```yaml
      ckpt: "${model_path}/weights_linear/pretrained_minigpt4_7b.pth"
      ```

3. 下载 [大语言模型 Vicuna-7b 的权重](https://hf-mirror.com//Vision-CAIR/vicuna-7b/tree/main)。

   下载完成后，保存到路径`${model_path}/weights_language/`下。

   须修改配置文件`${work_space}/minigpt4/configs/models/minigpt4_vicuna0.yaml`中关于此路径的配置。

   line 18
      ```yaml
      llama_model: "${model_path}/weights_language/"
      ```

4. 下载处理图像所需的
   [VIT 的权重 eva_vit_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
   、
   [Qformer 的权重 blip2_pretrained_flant5xxl.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth)
   以及 [Bert(bert-base-uncased) 的 Tokenizer](https://hf-mirror.com//bert-base-uncased)。

   下载完成后，保存到路径`${model_path}/weights_image/`下。完成后，此路径下的全部文件应是如此：

   ```bash
   eva_vit_g.pth
   blip2_pretrained_flant5xxl.pth
   bert-base-uncased
     config.json
     tokenizer_config.json
     vocab.txt
   ```

   须进行的配置修改如下：

    1. `./om_trans/eva_vit_model.py`

       line 55
       ```python
       encoder_config = BertConfig.from_pretrained("${model_path}/weights_image/bert-base-uncased")
       ```

    2. `${work_space}/minigpt4/models/eva_vit.py`，

       line 433
       ```python
       state_dict = torch.load("${model_path}/weights_image/eva_vit_g.pth", map_location="cpu")
       ```

    3. `${work_space}/minigpt4/models/minigpt4.py`

       line 28
       ```python
       q_former_model = "${model_path}/weights_image/blip2_pretrained_flant5xxl.pth"
       ```

       line 150
       ```python
       q_former_model = cfg.get("q_former_model", "${model_path}/weights_image/blip2_pretrained_flant5xxl.pth")
       ```

       line 89
       ```python
       encoder_config = BertConfig.from_pretrained("${model_path}/weights_image/bert-base-uncased")
       ```

### 图像处理部分的 om 转换与其他的源码修改

见[附录](#附录)。

## 基本推理

1. 修改 `${model_path}/weights_language/config.json`（让 llama 知道我们的输入是 embeds 而非 ids）

   line 24

    ```json
   "skip_word_embedding": true
    ```

3. 进入`./predict/`，将 `${work_space}`, `${model_path}` 填入 `run_predict.sh`

   line 10, 11

    ```bash
    minigpt_dir="${work_space}"
    LLM_model_path="${model_path}/weights_language"
    ```

   运行此脚本，参考

    ```bash
    bash run_predict.sh
    ```

## 测试

### 图像处理时间测试结果

将图像处理部分转换为 om 模型后，图像处理时间约为0.018s；GPU图像处理时间约为1.185s

### 精度测试

#### 方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。
再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载
   [open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，

   下载
   [测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入一个文件夹，

   建议都放置到`./precision`下。

2. 收集推理结果。
    1. GPU 上：收集脚本参考 `./precision/run_predict_walk_dir_GPU.py`，
       将其放到`${work_space}`目录下执行，注意脚本传参（主要是`--image-path`和`--output-path`）。
    2. NPU 上：类似基本推理，只需增加一个参数（图片文件夹的路径）即可
       ```bash
       bash run_predict.sh 图片文件夹的路径
       ```
   收集的结果应是类似 `./precision/GPU_NPU_result_example.json` 的形式。

3. 对结果进行评分：执行脚本 `./precision/clip_score_minigpt4.py`，参考命令：
   ```bash
   python clip_score_minigpt4.py --image_info GPU_NPU_result_example.json（这个替换成你的实际路径）
   ```
   得分高者精度更优。

### 性能测试

#### 方案

我们基于 `../../../examples/models/llama/run_fa.sh`，略微修改运行逻辑，得到我们的性能测试脚本 `./performance/run_performance.sh`。

#### 实施

1. 修改 `${model_path}/weights_language/config.json`（让 llama 仍走完整的计算逻辑）

   line 24

    ```json
   "skip_word_embedding": false
    ```

2. 将 `${model_path}` 填入 `./performance/run_performance.sh`

   line 8

    ```bash
    LLM_model_path="${model_path}/weights_language"
    ```

   并按需设置测试参数（参考 line 10-20）。此脚本支持自动摸高。

   运行此脚本，参考

    ```bash
    bash run_performance.sh
    ```

## 附录

### 图像处理部分的 om 转换

#### 概述

MiniGPT-4 的图像处理部分的逻辑是固定的，且在每次推理中只执行一次，比较适合转换为 om 离线模型，以提高运行性能。

整个过程分为三步。

第一步，使用 `torch.onnx.export` 把需要转换的计算逻辑制作成一个 onnx 格式的模型。

第二步，使用昇腾 ATC 工具将上述 onnx 模型转换为 om 模型。

第三步，修改 MiniGPT-4 源码，接入转换所得的 om 模型。

#### onnx 转换

1. 首先，识别出图像处理部分的逻辑。即原始代码中`minigpt4.py`的第 125 行的
   `image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)`
   及其配套代码。将这一部分单独写成一个文件（即`./om_trans/eva_vit_model.py`）。
   将它拷贝到`${work_space}/minigpt4/models`目录下。

2. 基于这部分代码，使用 `torch.onnx.export` 将相应的权重转换为 onnx 格式，详见 `./om_trans/onnx_model_export.py`。
   运行该文件，即可得到 onnx 模型。
   参考运行命令:
   ```bash
   python onnx_model_export.py --onnx-model-dir onnx模型的输出路径 --image-path ${work_space}/examples_v2/office.jpg
   ```
   提示：`onnx_model_export.py`脚本需要 import `${work_space}/minigpt4` 下的模块，
   为确保能 import 成功，可以 cd 到 `${work_space}` 下再运行此脚本，也可以把 `${work_space}` 加入 `PYTHONPATH`。

#### om 转换

om 转换需使用昇腾 ATC 工具，参考
[这里](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000005.html)

1. 环境准备：安装 CANN 并 source 相应的环境变量；

2. 模型转换：参考快速入门中 onnx 网络模型转换成离线模型的章节，或参考执行下面的转换命令
   （要进入到已转换好的 onnx 模型目录中去执行上述命令，否则会找不到权重文件）：
   ```bash
   atc --model=eva_vit_g.onnx --framework=5 --output=${output_path}/eva_vit_g --soc_version="$soc_version"(按实际) --input_shape="input:1,3,224,224"
   ```
   转换完成后，将所得的 om 模型保存到路径`${model_path}/weights_image/eva_vit_g.om`。

#### 接入转换所得的om模型

1. 将 `./om_trans/image_encoder.py` 拷贝到 `${work_space}/minigpt4/models` 目录下。

2. 修改 `${work_space}/minigpt4/models/minigpt_base.py` 文件，具体如下：

    1. 导入图像 om 模型推理类

       line 13
       ```python
       from minigpt4.models.image_encoder import ImageEncoderOM
       ```

    2. 新增如下代码，初始化加载 om 模型

       line 40
       ```python
       self.image_encoder = ImageEncoderOM("${model_path}/weights_image/eva_vit_g.om", device_8bit)
       ```

    3. 删除原来的图像处理代码

       line 51
       ```python
       self.visual_encoder, self.ln_vision = self.init_vision_encoder(
           vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit
       )
       ```

    4. 修正（源码多写了一次`.model`）

       line 312
       ```python
       if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
           embeds = self.llama_model.base_model.model.embed_tokens(token_ids)
       ```


3. 修改 `${work_space}/minigpt4/models/minigpt4.py` 文件，具体如下：

    1. 在原文件的第 62 行和 70 行，将`self.visual_encoder.num_features`修改为 VisionTransformer 类的入参 embed_dim 的固定值 1408.

       line 62
        ```python
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408, freeze_qformer)
        ```

       line 70
        ```python
        img_f_dim = 1408 * 4
        ```

    2. 图像 embedding 的计算不再走原始逻辑，改用转换后的 om 模型进行计算

       line 125
        ```python
        image_embeds = torch.tensor(self.image_encoder.image_encoder_om.infer(image.cpu().numpy())[0]).to(device)
        ```

### 对源码的其他必要修改

1. 修改 `${work_space}/minigpt4/models/base_model.py` 文件，具体如下：

   修改的目的是改用来自昇腾模型库的 llama_model。

    1. 删除不必要的三方件引入（训练才需要）

       删除 line 17
        ```python
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_int8_training,
        )
        ```

    2. 改用来自昇腾模型库的 LlamaForCausalLM 类

       将 line 26
        ```python
        from minigpt4.models.modeling_llama import LlamaForCausalLM
        ```
       替换为
        ```python
        from atb_llm.models.llama.causal_llama import LlamaForCausalLM
        from atb_llm.runner import ModelRunner
        ```

    3. 重写 `init_llm(...)` 方法

       重写 line 171 的方法为
        ```python
        def init_llm(cls, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                     **lora_kargs):
            logging.info('Loading LLAMA')
            llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)
            llama_tokenizer.pad_token = "$$"

            rank = int(os.getenv("RANK", "0"))
            world_size = int(os.getenv("WORLD_SIZE", "1"))
            llama_model_runner = ModelRunner(llama_model_path, rank=rank, world_size=world_size,
                                             is_flash_causal_lm=False)
            llama_model_runner.load_weights()
            llama_model = llama_model_runner.model
            for name, param in llama_model.named_parameters():
                param.requires_grad = False

            logging.info('Loading LLAMA Done')
            return llama_model, llama_tokenizer
        ```

3. 修改 `${work_space}/minigpt4/datasets/data_utils.py` 文件，具体如下：

   删除不必要的三方件引入及其使用。

   删除 line 18, 19, 29
   ```python
   import decord
   
   from decord import VideoReader
   
   decord.bridge.set_bridge("torch")
   ```

4. 修改 `${work_space}/eval_configs/minigpt4_eval.yaml` 文件，具体如下：

   由于无法使用 CUDA 的 8 位优化器，需将`low_resource`参数值设置为`False`。

   修改 line 6
   ```yaml
   low_resource: False
   ```
### 附加说明

1. 开启同步参数 ASCEND_LAUNCH_BLOCKING=1。
2. Vicuna-7b权重加载比较慢，Llama2-7B的权重加载更快，而且模型能力更强，推荐使用Llama2-7B。