# GLM4-9B 模型推理指导 <!-- omit in toc -->

# 概述

- GLM4 是智谱AI和清华大学 KEG 实验室联合发布的对话预训练模型。GLM4-9B 是 [GLM4]((https://github.com/THUDM/GLM-4)) 系列中的开源模型，在保留了前三代模型对话流畅、部署门槛低等众多优秀特性的基础上，GLM4-9B 有更强大的基础模型、更完整的功能支持、和更全面的开源序列。
- 此代码仓中实现了一套基于NPU硬件的GLM4-9B-chat推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了GLM4-9B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化  | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|-----|-----|
| GLM4-9B-chat    | 支持world size 1,2,4,8 | 支持world size 1,2,4 | 是   | 是   | 否              | 是              | 是      | 否      | 否           | 是 | 否     | 是     | 否 | 是 |

- 此模型仓已适配的模型版本
  - [GLM4-9B-chat](https://huggingface.co/THUDM/glm-4-9b-chat/tree/main)
  - 注：GLM4-9B-chat 推荐使用commit id为 `e824789b14985787c181beaac940d103b2516cb0` 的模型仓版本
  - 注：transformer需要安装固定的4.42.4版本


# 使用说明

- 参考[此README文件](../../chatglm/v2_6b/README.md)

- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。

## 量化权重导出
量化权重可通过msmodelslim（昇腾压缩加速工具）实现。

### 环境准备
环境配置可参考msmodelslim官网：https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0002.html

### 导出w8a8量化权重
通过`${llm_path}/examples/models/chatglm/v4_9b/quant_glm4_w8a8.sh`文件导出模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：
```shell
export INF_NAN_MODE_ENABLE=0
bash quant_glm4_w8a8.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -trust_remote_code
```

导出量化权重后应生成`quant_model_weight_w8a8.safetensors`和`quant_model_description_w8a8.json`两个文件。

注：

1.quant_glm4_w8a8.sh文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行脚本生成量化权重时，会在生成的权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，当前仅支持`w8a8`。

3.执行完以上步骤后，执行量化模型只需要替换权重路径。

4.如果生成权重时遇到`OpenBLAS Warning: Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP = 1 option`，可通过设置`export OMP_NUM_THREADS=1`来关闭多线程规避。

### 导出稀疏量化权重
- 稀疏量化权重请使用以下指令生成

  校准数据集从 [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/e84444333b6d434ea7b0/) 获取，解压后，使用解压目录下的 `CEval/val/Other/civil_servant.jsonl` 作为校准数据集。
    > 运行前需要确保压缩工具编译过
    >
    > `cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/msmodelslim/pytorch/weight_compression/compress_graph`
    >
    > `bash build.sh /usr/local/Ascend/ascend-toolkit/latest`

    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

    cd ${llm_path}/examples/models/chatglm/v4_9b
    bash convert_quant_weights.sh ${浮点权重路径} ${W8A8S量化权重路径} ${校准数据集路径} ${TP_Size} ${指定生成量化权重的卡号(使用单卡单芯)} ${指定生成稀疏权重的卡号(根据TP数来选择几卡几芯)} -trust_remote_code
    # 例如：bash convert_quant_weights.sh /home/data/glm-4-9b-chat /home/data/glm4_w8a8 /home/data/CEval/val/Other/civil_servant.jsonl 2 0 0,1 -trust_remote_code

## 运行操作说明
- 参考[此README文件](../../chatglm/v2_6b/README.md)
### 开启多头自适应压缩特性
 ```shell
  # 需开启环境变量
  export ATB_LLM_RAZOR_ATTENTION_ROPE=1
  export ATB_LLM_RAZOR_ATTENTION_ENABLE=1
 ```

## 精度测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## FAQ
- `import torch_npu`遇到`xxx/libgomp.so.1: cannot allocate memory in static TLS block`报错，可通过配置`LD_PRELOAD`解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`