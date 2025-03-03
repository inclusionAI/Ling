# README

- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)是面向图文理解的端侧多模态大模型系列。该系列模型接受图像和文本输入，并提供高质量的文本输出.
- 此代码仓中实现了一套基于NPU硬件的MiniCPM-V推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
| 模型及参数量      | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service |纯模型支持模态  | 服务化支持模态 |
|-------------|--------------------------| --------------------------- | ---- |------------|-----------------|---------|------------|
| minicpmv2-2b| ×                        | 支持world size 1             | √    | ×          | ×               | 文本、图片 | 当前模型不支持服务化 |

# 使用说明

## 路径变量解释

| 变量名               | 含义                                                                                                                                                             |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| working_dir       | 加速库及模型库下载后放置的目录                                                                                                                                                |
| llm_path          | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models`                                          |
| script_path       | 脚本所在路径；minicpmv2的工作脚本所在路径为 `${llm_path}/examples/models/minicpmv2`                                                                                               |
| weight_path       | 模型权重路径                                                                                                                                                         |
| image_path        | 图片所在路径                                                                                                                                                         |
| max_batch_size    | 最大bacth数                                                                                                                                                       |
| max_input_length  | 多模态模型的最大embedding长度。若要使用非默认值，请同步修改${llm_path}/examples/atb_models/atb_llm/models/minicpmv2/flash_causal_minicpmv2.py中prepare_prefill_token方法内的局部变量max_inp_length |
| max_output_length | 生成的最大token数                                                                                                                                                    |
| open_clip_path    | open_clip权重所在路径                                                                                                                                                |


## 权重

**权重下载**

- [MiniCPM-V-2](https://huggingface.co/openbmb/MiniCPM-V-2/tree/main)

**模型文件拷贝**

将权重目录中的resampler.py拷贝到${llm_path}/examples/atb_models/atb_llm/models/minicpmv2目录下

**基础环境变量**

-1.Python其他第三方库依赖，参考[requirements_minicpmv2.txt](../../../requirements/models/requirements_minicpmv2.txt)

-2.参考[此README文件](../../../README.md)

-3.执行以下指令获取timm依赖路径`${Location}`
    ```shell
    pip show timm
    ```
    
-4.修改`${Location}/timm/layers/pos_embed.py`，在第45行后添加代码

    ```python
    import torch_npu
    torch_npu.npu_format_cast_(posemb, 0)
    posemb = posemb.contiguous()
    ```
    
-注意：保证先后顺序，否则minicpmv2的其余三方依赖会重新安装torch，导致出现别的错误


## 推理

### 对话测试

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_pa.sh --run ${weight_path} ${image_path} -trust_remote_code
    ```
- 环境变量说明
  - `export ASCEND_RT_VISIBLE_DEVICES=0`
    - 当前模型暂时仅支持单卡推理
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
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

#### 方案

我们采用的精度测试方案是这样的：使用同样的一组图片，分别在 GPU 和 NPU 上执行推理，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载[open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，并把下载的权重放在open_clip_path目录下
   
2. 下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入{image_path}目录下
   

3. GPU上，将run_coco_gpu.py由{script_path}/precision目录拷贝至{llm_path}/examples/atb_models，运行脚本
   ```bash
   cd {llm_path}/examples/atb_models
   python run_coco_gpu.py --model_path ${weight_path} --image_path ${image_path} --trust_remote_code
   ```
   会在{script_path}/precision目录下生成gpu_coco_predict.json文件存储gpu推理结果

4. NPU 上,在\${llm_path}目录下执行以下指令：
   ```bash
   bash ${script_path}/run_pa.sh --precision ${weight_path} ${image_path} -trust_remote_code
   ```
   运行完成后会在{script_path}生成predict_result.json文件存储npu的推理结果

5. 对结果进行评分：分别使用GPU和NPU推理得到的两组图片描述(gpu_coco_predict.json、predict_result.json)作为输入,将clip_score_minicpmv2.py由交付件的{script_path}/precision目录拷贝至{llm_path}/examples/atb_models，在GPU执行clip_score_minicpmv2.py 脚本输出评分结果
```bash
   cd {llm_path}/examples/atb_models
   python clip_score_minicpmv2.py \ 
   --model_weights_path {open_clip_path}/open_clip_pytorch_model.bin \ 
   --image_info {gpu_coco_predict.json 或 predict_result.json的路径} \
   --dataset_path {iamge_path}
```

   得分高者精度更优。

## 性能测试

性能测试时需要在 `${image_path}` 下仅存放一张图片，使用以下命令运行 `run_pa.sh`，会自动输出batchsize为1-10时，输出token长度为 256时的吞吐。

测试性能时，需要导入环境变量:
```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
```
```shell
bash ${script_path}/run_pa.sh --performance ${weight_path} ${image_path} -trust_remote_code
```

可以在 `examples/models/minicpmv2` 路径下找到测试结果。

## FAQ
- 在对话测试或者精度测试时，用户如果需要修改输入input_texts,max_batch_size等时，可以修改{script_path}/minicpmv2.py里的参数，具体可见minicpmv2.py
- 更多环境变量见[此README文件](../../README.md)