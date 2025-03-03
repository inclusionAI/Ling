# README

- [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)，是一种多模态大模型，具有强大的图像和文本处理能力，通过开源组件缩小与商业多模态模型的差距——GPT-4V的开源替代方案。在聊天机器人中，InternLM-XComposer可以通过解析用户的文字输入，结合图像信息，生成更加生动、准确的回复。 此外，InternLM-XComposer还可以根据用户的图像输入，提供相关的文本信息，实现更加智能化的交互。
- 此代码仓中实现了一套基于NPU硬件的InternLM-XComposer推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。
- 支持 InternLM-XComposer2-VL-7B，基于 OpenAI 的 Clip-vit-large-patch14-336 视觉模型 + MLP + InternLM2-Chat-20B 文本模型的多模态推理。
- 支持 InternLM-XComposer2-4khd-7B, 基于 OpenAI 的 Clip-vit-large-patch14-336 视觉模型 + MLP + InternLM2-Chat-20B 文本模型的多模态推理。

# 特性矩阵
- 此矩阵罗列了 InternLM-XC2 系列模型支持的特性

| 模型及参数量    | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | 800I A2 BF16 | MindIE Service | 纯模型支持模态  | 服务化支持模态 |
| --------------- | -------------------------- | -------------------------- | ---- | ------------ | -------------- | -------------- | ------------ |
| InternLM-XComposer2-VL-7B | 支持world size 1   | 支持world size 1          | √    | ×            | ×              | 文本、图片      | 当前模型不支持服务化 |
| InternLM-XComposer2-4KHD-7B | 支持world size 1 | 支持world size 1          | √    | ×            | ×              | 文本、图片      | 当前模型不支持服务化 |

# 使用说明

## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；工作脚本所在路径为 `${llm_path}/examples/models/internlmxcomposer2`                                                                        |
| weight_path | 文本模型权重路径                                                   |
| vit_path  | 视觉模型所在路径                                                    |
| image_path  | 图片所在路径                                                      |
| open_clip_path | open_clip权重所在路径                                          |
| trust_remote_code | 是否信任本地可执行文件。默认不执行。若传入此参数，则信任本地可执行文件，本模型执行时需要传入该参数。|
## 权重

**权重下载**

- [视觉模型vit](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main)，请提前下载该视觉模型权重，并存放于 `vit_path` 目录
- [InternLM-XComposer2-VL-7B](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b/tree/main)
- [InternLM-XComposer2-4KHD-7B](https://hf-mirror.com/internlm/internlm-xcomposer2-4khd-7b/tree/main)

**权重转safetensor**

模型只支持safetensor格式权重，需要将bin格式权重转为safetensor格式，参考[此README文件](../../README.md)

**稀疏量化权重生成**

由于当前lora特性不支持量化且vit模型也不支持量化，稀疏量化功能只能作用于基础LLM模型，以下是 `InternLM-XComposer2-VL-7B` 量化步骤：

1. 拷贝一份权重，将其记作 `$weight_path_copy`
2. 由于lora特性不支持量化，需要修改文件 `$weight_path_copy/modeling_internlm2.py`，将 `MLP` 和 `Attention` 中使用到 `lora` 计算的方法删除 `im_mask`，例如：
```python
# 原始代码
qkv_states = self.wqkv(hidden_states, im_mask)
# 修改为
qkv_states = self.wqkv(hidden_states)
```
1. 当前量化工具暂不支持多模态，需要修改 /usr/local/Ascend/ascend-toolkit/latest/tools/msmodelslim/pytorch/llm_ptq/llm_ptq_tools/quant_tools.py 文件
  里的 `rollback_names_process` 函数加上 `nn.Conv2d`, 改成如下：
  ```python
  if isinstance(module, (nn.Linear,nn.Conv2d,  nn.modules.linear.NonDynamicallyQuantizableLinear))
  ```
  改完后设置CANN环境变量：source /usr/local/Ascend/ascend-toolkit/set_env.sh

1. 在 `$llm_path` 目录下执行稀疏量化权重生成步骤1：
  ```python
  python examples/models/internlmxcomposer2/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --w_bit 4 --a_bit 8 --calib_file ${llm_path}/examples/convert/model_slim/teacher_qualification.jsonl --fraction 0.011 --co_sparse True (--trust_remote_code)
  ```

1. 在 `$llm_path` 执行量化权重切分及压缩
  > 运行前需要确保压缩工具编译过
  >
  > `cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/msmodelslim/pytorch/weight_compression/compress_graph`
  >
  > `bash build.sh /usr/local/Ascend/ascend-toolkit/latest`
  ```shell
  torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径} (--trust_remote_code)
  ```
    - TP数为tensor parallel并行个数
    - 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    - 示例
      ```shell
        torchrun --nproc_per_node 1 -m examples.convert.model_slim.sparse_compressor --model_path internlm-xc2-vl-7b_w8a8s --save_directory internlm-xc2-vl-7b_w8a8sc
      ```

**W8A8量化权重生成**
步骤分为4步，其中步骤1、2、3同**稀疏量化权重生成**，步骤4为：
```shell
python examples/models/internlmxcomposer2/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --w_bit 8 --a_bit 8 --calib_file ${llm_path}/examples/convert/model_slim/teacher_qualification.jsonl --device_type npu (--trust_remote_code)
```
  
## 推理前准备

**环境准备**

- torchvision版本需与torch版本配套，请参考[此txt文件](../../../requirements/models/requirements_internlmxcomposer2.txt)
- 测试前请先source环境变量，包括CANN、ATB、MINDIE

**运行 Mindie Paged Attention FP16前修改配置**

- 修改模型权重路径下 `config.json` 文件中的 `torch_dtype` 为 float16：
  ```shell
  {
    ...
    "torch_dtype": "float16",
  }
  ```

- 运行前请使用本地视觉模型，修改视觉模型从本地路径获取，设置 `${llm_path}/atb_llm/models/internlmxcomposer2/buildmlp/build_mlp.py`、`${llm_path}/atb_llm/models/internlmxcomposer2/buildmlp/build_mlp_4k.py` 中 `vision_tower` 为 `${vit_path}`

- 增加lora适配文件：adapter_config.json 和 lora_adapter.json，放在对应文本模型权重路径下：
  ```shell
  # InternLM-XComposer2-VL-7B 的 adapter_config.json 文件内容为：
  {
    "lora_alpha": 256,
    "r": 256
  }

  # InternLM-XComposer2-4KHD-7B 的 adapter_config.json 文件内容为：
  {
    "rank_pattern":{
      "attention.wqkv": 8
    },
    "alpha_pattern":{
      "attention.wqkv": 16
    },
    "lora_alpha": 256,
    "r": 256
  }

  # 配置lora权重路径，InternLM-XComposer2-VL-7B和InternLM-XComposer2-4khd-7B的lora权重都融合在文本模型权重中，因此，这里配置文本模型权重实际路径 $weight_path 即可。
  # lora_adapter.json文件内容为：
  {"internlmxc2":"$weight_path"}
  ```

**运行Torch_npu FP16前配置**
- 运行 torch_npu 推理前应设置 `${weight_path}/build_mlp.py` 文件中的 vision_tower 为 `${vit_path}`
- 运行InternLM-XComposer2-4khd-7B模型的Torch_npu结果时，请修改为 `eager` 模式，修改config.json文件为：
  ```shell
  {
    ...
    "attn_implementation": "eager",
  }
  ```

## 精度测试

#### 方案

精度测试方案：使用同样的一组图片，分别执行 Torch_npu 和 加速库 路线，得到两组图片描述。 再使用 open_clip 模型作为裁判，对两组结果分别进行评分，以判断优劣。

#### 实施

1. 下载[open_clip 的权重 open_clip_pytorch_model.bin](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)，并把下载的权重放在open_clip_path目录下
   下载[测试图片（CoCotest 数据集）](https://cocodataset.org/#download)并随机抽取其中100张图片放入 `${image_path}` 目录下

2. Torch_npu 路线，在 `${llm_path}` 目录下，运行脚本
   ```bash
   # 注意：300I Duo场景下，InternLM-XComposer2-4KHD-7B 模型需修改 `${weight_path}/modeling_internlm_xcomposer2.py` 文件中 `img2emb` 函数中的 `image` 变量为 `image.cpu()` ，以转换到cpu侧执行
   python ${script_path}/precision/run_coco_gpu.py --model_path ${weight_path} --image_path ${image_path} (--trust_remote_code)
   ```
   会在当前 `${llm_path}` 目录下生成torch_npu_coco_predict.json文件存储torch_npu推理结果

3. 加速库 路线,在 `${llm_path}` 目录下执行以下指令：
   ```bash
   bash ${script_path}/run_pa.sh --precision (--trust_remote_code) ${weight_path} ${image_path}
   ```

   运行完成后会在 `${script_path}` 目录生成predict_result.json文件存储加速库路线的推理结果

4. 对结果进行评分：分别使用GPU和NPU推理得到的两组图片描述(torch_npu_coco_predict.json、predict_result.json)作为输入,执行clip_score_internlmxcomposer2.py 脚本输出评分结果，在 `${llm_path}` 目录下执行：
```bash
   python examples/models/internlmxcomposer2/precision/clip_score_internlmxcomposer2.py \ 
   --model_weights_path ${open_clip_path}/open_clip_pytorch_model.bin \ 
   --image_info {gpu_coco_predict.json 或 predict_result.json的路径} \
   --dataset_path ${image_path}
```


## 性能测试

性能测试时需要在 `${image_path}` 下仅存放一张图片，使用以下命令运行 `run_pa.sh`，会自动输出batchsize为1，输出token长度为 256 时的吞吐。

测试模型侧性能数据，开启环境变量
  ```shell
  export ATB_LLM_BENCHMARK_ENABLE=1
  export ATB_LLM_BENCHMARK_FILEPATH=${script_path}/benchmark.csv
  ```

```shell
bash ${script_path}/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path}
```

例如在 MindIE-ATB-Models 根目录，可以运行：

```shell
bash examples/models/internlmxcomposer2/run_pa.sh --performance (--trust_remote_code) ${weight_path} ${image_path}
```

可以在 `examples/models/internlmxcomposer2/internlmxcomposer2_performance.csv` 文件中找到测试结果。

## FAQ
- 在精度测试和性能测试时，用户如果需要修改输入prompt，max_batch_size，max_output_length时，可以修改{script_path}/run_pa.sh里的可修改配置
- 更多环境变量见[此README文件](../../README.md)