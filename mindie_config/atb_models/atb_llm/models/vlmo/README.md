[TOC]

# Vlmo模型-推理指导

# 概述

VLMo 是由微软提出的一种多模态 Transformer 模型，Mixture-of-Modality-Experts (MOME)，即混合多模态专家。VLMo 相当于是一个混合专家 Transformer 模型。预训练完成后，使用时既可以是双塔结构实现高效的图像文本检索，又可以是单塔结构成为分类任务的多模态编码器。

- 参考实现：

  ```
  https://github.com/microsoft/unilm/tree/master/vlmo
  ```

# 快速上手

## 获取源码及依赖

### 1. 环境部署

#### 1.1 安装PytorchAdapter
#### 1.1.1 安装requirements依赖包

| 包名               | 推荐版本   |  
|-----------------|--------|
| transformers     | 4.37.0  |
| decorator        | 5.1.1  |
| sympy            | 1.12.1 |
| scipy            | 1.12.0 |
| attrs            | 23.1.0 |
| sentencepiece    | 0.1.99 |
| pytorch_lightning| 1.5.5  |
| Pillow| 10.2.0 |
| tqdm |4.53.0|
| ipdb |0.13.7|
| einops| 0.3.0|
| pyarrow |14.0.1|
| sacred |0.8.5|
| pandas |2.2.0|
| timm |0.4.12|
| torchmetrics| 0.7.3|
| fairscale |0.4.0|
| numpy |1.26.4|
| opencv-python |4.9.0.80|
| opencv-python-headless| 4.9.0.80|
| psutil |5.9.8|
| torchvision |0.16.2|

##### 1.1.2 安装atd_speed
cd /MindIE-LLM/examples/atb_models/examples/models/
pip install ./atb_speed_sdk/

##### 1.1.3 安装torch
安装方法：

| 包名                                              |
|-------------------------------------------------|
| torch-*+cpu-cp38-cp38-linux_x86_64.whl      |
| torch-*+cpu-cp39-cp39-linux_x86_64.whl      |
| torch-*-cp38-cp38-manylinux2014_aarch64.whl |
| torch-*-cp39-cp39-manylinux2014_aarch64.whl |
| ...                                             |

根据所使用python版本，以及CPU架构，选择对应的包

```bash
# 以安装torch-*-cp39-cp39-manylinux2014_aarch64.whl包为例
pip install torch-*-cp39-cp39-manylinux2014_aarch64.whl
```

##### 1.1.4 安装torch_npu

安装方法：

| 包名                         |
|----------------------------|
| pytorch_v*_py38.tar.gz |
| pytorch_v*_py39.tar.gz |
| ...                        |

选择安装与torch版本以及python版本一致的torch_npu版本

```bash
# 安装torch_npu，以torch*对应的python3.9的aarch64版本为例
tar -zxvf pytorch_v*_py39.tar.gz
pip install torch*_aarch64.whl
```




##### 注：安装完毕部分依赖后可能会修改torch版本，请确保运行时torch版本为所需版本



### 2. 安装依赖

#### 路径变量解释

| 变量名                 | 含义                                                                   |  
|---------------------|----------------------------------------------------------------------|
| model_download_path | 开源权重放置目录                                                             | 
| data_download_path|  数据集放置目录
| llm_path            | 加速库及模型库下载后放置目录                                                       |
| model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |

#### 2.1 推理环境准备

1. 下载代码，通过git工具将vlmo代码下载至本地 `${model_path}` 中。
   ```
   git clone https://github.com/microsoft/unilm.git
   ```
   
2. 下载模型权重，放置到自定义`${model_download_path}` (请查看 README.md 下载链接中'Configs'页签下所需测试集的'finetuned weight')
   ```
   https://github.com/microsoft/unilm/tree/master/vlmo
   ```
   分类任务请使用 VQAv2数据集进行评估，检索任务情使用 COCO 数据集进行评估\
   以VQAv2为例，下载 vlmo_base_patch16_480_vqa.pt 将其放在 `${model_download_path}` 目录中。

3. 下载数据集，同上url下载相应数据集。(请查看 DATA.md 下载指定测试集的数据，并整理成所需目录结构)放置`${data_download_path}`目录\
   以VQAv2为例，将文件按照文档说明整理为如下格式：
   ```
      `${data_download_path}`
      ├── train2014            
      │   ├── COCO_train2014_000000000009.jpg                
      |   └── ...
      ├── val2014              
      |   ├── COCO_val2014_000000000042.jpg
      |   └── ...  
      ├── test2015              
      |   ├── COCO_test2015_000000000001.jpg
      |   └── ...         
      ├── v2_OpenEnded_mscoco_train2014_questions.json
      ├── v2_OpenEnded_mscoco_val2014_questions.json
      ├── v2_OpenEnded_mscoco_test2015_questions.json
      ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
      ├── v2_mscoco_train2014_annotations.json
      └── v2_mscoco_val2014_annotations.json
   ```
   在 `${model_path}`/unilm/vlmo 目录下新建文件 makearrow.py 内容如下：
   ```python
   from vlmo.utils.write_vqa import make_arrow
   make_arrow('{data_download_path}', '{data_download_path}/vqa_arrow')
   ```
   
   对于VQA v2数据集，vlmo的write_vqa脚本不会生成分类结果与答案的映射关系，需要在`${model_path}`/unilm/vlmo/vlmo/utils/write_vqa.py 中最下方手动添加代码进行输出。

   ```python
       # 注意行对齐
       with open(os.path.join(dataset_root, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))
   ```

   执行该脚本，将会在 vqa_arrow文件夹下生成相应的二进制数据集文件：
   ```shell
   python makearrow.py
   ```
   生成目录结构如下：
   ```
      `${data_download_path}`W
        arrow
          ├── vqav2_val.arrow
          ├── vqav2_trainable_val.arrow
          ├── vqav2_train.arrow
          ├── vqav2_test.arrow
          ├── vqav2_test-dev.arrow
          ├── vqav2_test.arrow
          └── answer2label.txt
   ```


4. 下载Bert 词表
   ```
   https://huggingface.co/google-bert/bert-base-uncased/tree/main
   ```
   在Files and versions 页签中找到 vocab.txt 下载后放入 `${model_download_path}` 中备用。
   
### 拷贝文件

### 准备

#### 1. 将大模型加速库中 vlmo 相关的 文件替换至 model_path 中的指定路径

```shell
cd ${llm_path}/pytorch/examples/vlmo/
cp multiway_transformer.py ${model_path}/unilm/vlmo/vlmo/modules
cp vlmo_module.py ${model_path}/unilm/vlmo/vlmo/modules
cp objectives.py ${model_path}/unilm/vlmo/vlmo/modules
cp vlmo_utils.py ${model_path}/unilm/vlmo/vlmo/modules
cp vlmo_file_check.py ${model_path}/unilm/vlmo/vlmo/modules
cp run_ascend_vqa.py ${model_path}/unilm/vlmo/
cp run_ascend_vqa.sh ${model_path}/unilm/vlmo/
cp cut_model_util.py ${model_path}/unilm/vlmo/
cp cut_ascend_vqa.py ${model_path}/unilm/vlmo/
cp cut_model_and_run.sh ${model_path}/unilm/vlmo/
cp vlmo_ascend_utils.py ${model_path}/unilm/vlmo/
```

#### 2.修改配置

以VQA v2 task_finetune_vqa_base_image480 微调评估为例。\
打开 `${model_path}`/unilm/vlmo/run_ascend_vqa.sh \
修改 `<Finetuned_VLMo_WEIGHT>`  为 `${model_download_path}`；修改 `<CONFIG_NAME>` 为 task_finetune_vqa_base_image480

打开 `${model_path}`/unilm/vlmo/run_ascend_vqa.py \
修改 `VQA_ARROW_DIR`  路径为 '`${data_download_path}`/arrow' ；修改 `<BERT_VOCAB>` 为 '`${model_download_path}`/vocab.txt'
修改 DEVICE_ID 后的值可选择在哪张卡上运行

##### 修改双芯推理配置
打开 `${model_path}/unilm/vlmo/cut_model_and_run.sh` 修改input_path为`${model_download_path}`;修改 `CONFIG_NAME` 后的值为 task_finetune_vqa_base_image480
打开 `${model_path}/unilm/vlmo/cut_ascend_vqa.py` \
修改 `VQA_ARROW_DIR`  路径为 '`${data_download_path}`/arrow' ；修改 `<BERT_VOCAB>` 为 '`${model_download_path}`/vocab.txt'。
修改 DEVICE_ID 后的值可选择在哪张卡上运行
 
# CPU高性能模式

可开启CPU Performance模式以提高模型推理性能。

```
cpupower frequency-set -g performance
```

### 执行推理

#### run_ascend_vqa.sh

用于执行已经基于VQA v2 数据集微调好的权重，执行图片分类任务。输入为一张图片以及一个问题，推理结果为一个特征值，通过分类器可将其从3129个备选答案中选出一个结果。

```shell
bash run_ascend_vqa.sh
```
若执行推理时遇到性能下降问题，可以尝试清除所有进程再重新执行推理。
```shell
pkill python
```
或更换npu编号重新执行推理

### 执行双芯推理
#### cut_model_and_run.sh
第一次执行为切分权重，第二次执行为进行双芯推理。
```shell
bash cut_model_and_run.sh
```

### 获取精度、性能数据：
模型运行完毕后，会在日志中打印出accuracy（正确率/精度数据）mean of cost（平均耗时/性能数据）作为对比参考
#### FAQ

### 

1. ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block  

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

则可取消run_inf_ascend_*.sh 脚本中的注释，修改为报错中相应的路径。如

```shell
LD_PRELOAD=/root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1:$LD_PRELOAD
```