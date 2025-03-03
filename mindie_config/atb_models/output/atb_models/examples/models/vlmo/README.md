# README

- [VLMo(Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts.)](https://github.com/microsoft/unilm/tree/master/vlmo)是由微软提出的一种多模态 Transformer 模型，Mixture-of-Modality-Experts (MOME)，即混合多模态专家。VLMo 相当于是一个混合专家 Transformer 模型。预训练完成后，使用时既可以是双塔结构实现高效的图像文本检索，又可以是单塔结构成为分类任务的多模态编码器。

- 此代码仓中实现了一套基于NPU硬件的VLMO推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了各VLMO模型支持的特性

| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|----------------------------|---------------------------|------|------------------|-----------------|-----------------|---------|-----------|---------|-----------|--------------------------|-----|--------|---|
|  VLMO    | 支持world size 1,2     | 支持world size 1,2          | √   | ×                    | √              | ×              | ×       | ×          | ×      | ×                        | ×   | ×      | ×   | ×     |

- 此模型仓已适配的模型版本
  - [VLMO系列](https://github.com/microsoft/unilm/tree/master/vlmo)

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；VLMO的文件所在路径为`${llm_path}/atb_llm/models/vlmo`                            |
| weight_path | 模型权重路径                            |

## 权重
**权重下载**
- [VLMO](https://github.com/microsoft/unilm/tree/master/vlmo)
- 请查看 README.md 下载链接中'Configs'页签下所需测试集的'finetuned weight'
- 分类任务请使用 VQAv2数据集进行评估，检索任务情使用 COCO 数据集进行评估\
   以VQAv2为例，下载 vlmo_base_patch16_480_vqa.pt

**权重转换**
- 不涉及

**量化权重生成**
- 不涉及


**基础环境变量**
- 参考[此README文件](../../../README.md)

## 推理

### 测试
**运行Flash Attention FP16**
- VLMO模型参考以下运行方式
  - 安装依赖包 pip install 包名==版本号
    ```shell
    | 包名               | 推荐版本   |  
    |-----------------|--------|
    | transformers     | 4.33.1 | 
    | decorator        | 5.1.1  |
    | sympy            | 1.11.1 |
    | scipy            | 1.11.3 |
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
    | scipy |1.12.0|
    | opencv-python |4.9.0.80|
    | opencv-python-headless| 4.9.0.80|
    | psutil |5.9.8|
    | torchvision |0.16.2|
    如torchvision版本安装失败，则说明需要从Huawei源下载，需要将pip源修改为华为源http://cmc-cd-mirror.rnd.huawei.com/pypi/simple/
    ```
  - 安装torch
    - `根据所使用python版本，以及CPU架构，选择对应的包`
      ```bash
      # 以安装torch-*-cp39-cp39-manylinux2014_aarch64.whl包为例
      pip install torch-*-cp39-cp39-manylinux2014_aarch64.whl
      ```
  - 安装torch_npu
    - `选择安装与torch版本以及python版本一致的torch_npu版本`
      ```bash
      # 安装torch_npu，以torch*对应的python3.9的aarch64版本为例
      tar -zxvf pytorch_v*_py39.tar.gz
      pip install torch*_aarch64.whl
      ```
  - 路径变量解释
    | 变量名                 | 含义                                                                   |  
    |---------------------|----------------------------------------------------------------------|
    | model_download_path | 开源权重放置目录                                                             | 
    | data_download_path|  数据集放置目录
    | llm_path            | 加速库及模型库下载后放置目录                                                       |
    | model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |
  - 环境准备
    - 下载代码，通过git工具将vlmo代码下载至本地 `${model_path}` 中
      ```
      git clone https://github.com/microsoft/unilm.git
      ```  
    - 下载模型权重，放置到自定义`${model_download_path}` 下载方式参考上文模型权重下载
    - 下载数据集(请查看 DATA.md 下载指定测试集的数据，并整理成所需目录结构)放置`${data_download_path}`目录\
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
      ```python
        #对于VQA v2数据集，vlmo的write_vqa脚本不会生成分类结果与答案的映射关系，需要在`${model_path}`/unilm/vlmo/vlmo/utils/write_vqa.py 中最下方手动添加代码进行输出。

      
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
    - 下载Bert 词表
      ```
      https://huggingface.co/google-bert/bert-base-uncased/tree/main
      ```
      在Files and versions 页签中找到 vocab.txt 下载后放入 `${model_download_path}` 中备用。  
    - 拷贝文件
      - 将大模型加速库中 vlmo 相关的 文件替换至 model_path 中的指定路径
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
      - 修改配置
        以VQA v2 task_finetune_vqa_base_image480 微调评估为例。\
        打开 `${model_path}`/unilm/vlmo/run_ascend_vqa.sh \
        修改 `<Finetuned_VLMo_WEIGHT>`  为 `${model_download_path}`；修改 `<CONFIG_NAME>` 为 task_finetune_vqa_base_image480

        打开 `${model_path}`/unilm/vlmo/run_ascend_vqa.py \
        修改 `VQA_ARROW_DIR`  路径为 '`${data_download_path}`/arrow' ；修改 `<BERT_VOCAB>` 为 '`${model_download_path}`/vocab.txt'
        修改 DEVICE_ID 后的值可选择在哪张卡上运行
    - 执行推理
      - 单芯推理 run_ascend_vqa.sh
        ```shell
        bash run_ascend_vqa.sh
        ``` 
      - 双芯推理 cut_model_and_run.sh
        - 修改双芯推理配置
          打开 `${model_path}/unilm/vlmo/cut_model_and_run.sh` 修改input_path为`${model_download_path}`;修改 `CONFIG_NAME` 后的值为 task_finetune_vqa_base_image480
          打开 `${model_path}/unilm/vlmo/cut_ascend_vqa.py` \
          修改 `VQA_ARROW_DIR`  路径为 '`${data_download_path}`/arrow' ；修改 `<BERT_VOCAB>` 为 '`${model_download_path}`/vocab.txt'。
          修改 DEVICE_ID 后的值可选择在哪张卡上运行
        - 第一次执行为切分权重，第二次执行为进行双芯推理。
        ```shell
        bash cut_model_and_run.sh
        ```


            
**运行Flash Attention BF16**
- 暂不支持

**运行Flash Attention W8A8**
- 运行启动脚本
- 暂不支持

**运行Flash Attention W8A16**
- 暂不支持

**运行Paged Attention FP16**
- 暂不支持

**运行Paged Attention BF16**
- 暂不支持

**运行Paged Attention W8A8**
- 暂不支持

**运行Paged Attention W8A16**
- 暂不支持

**运行KV cache量化**
- 待补充

**运行稀疏量化**
- 暂不支持

**运行MOE量化**
- 待补充

## 精度测试
模型运行完毕后，会在日志中打印出accuracy（正确率/精度数据）mean of cost（平均耗时/性能数据）作为对比参考

## 性能测试
模型运行完毕后，会在日志中打印出accuracy（正确率/精度数据）mean of cost（平均耗时/性能数据）作为对比参考

## FAQ
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
