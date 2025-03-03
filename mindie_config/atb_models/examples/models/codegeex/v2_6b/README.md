# CodeGeeX2-6B 模型推理指导 <!-- omit in toc -->

# 概述

- [CodeGeeX2-6B](https://github.com/THUDM/CodeGeeX2) 是多语言代码生成模型 [CodeGeeX](https://github.com/THUDM/CodeGeeX) ([KDD’23](https://arxiv.org/abs/2303.17568)) 的第二代模型。不同于一代 CodeGeeX（完全在国产华为昇腾芯片平台训练） ，CodeGeeX2 是基于 [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B) 架构加入代码预训练实现，得益于 ChatGLM2 的更优性能，CodeGeeX2 在多项指标上取得性能提升（+107% > CodeGeeX；仅60亿参数即超过150亿参数的 StarCoder-15B 近10%）。
- 此代码仓中实现了一套基于NPU硬件的CodeGeeX2推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了CodeGeeX2-6B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|-----|-----|
| CodeGeeX2-6B    | 支持world size 1,2,4,8 | 支持world size 1,2,4  | 是   | 否   | 否              | 是              | 是      | 否     | 否           | 否       | 否     | 是     | 是  | 否 |

- 此模型仓已适配的模型版本
  - [CodeGeeX2-6B](https://huggingface.co/THUDM/codegeex2-6b/tree/main)


# 使用说明

- 执行推理前需要将权重目录下的config.json中的`torch_dtype`改为`"float16"`
- 除了“量化权重导出”章节，其余均参考[此README文件](../../chatglm/v2_6b/README.md)
- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认不执行。传入此参数，则信任本地可执行文件。

## 量化权重导出
量化权重可通过msmodelslim（昇腾压缩加速工具）实现。

### 环境准备
环境配置可参考msmodelslim官网：https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0002.html

### 导出量化权重
通过`${llm_path}/examples/models/codegeex/v2_6b/quant_codegeex_w8a8.sh`文件导出模型的量化权重（注意量化权重不要和浮点权重放在同一个目录下）：
```shell
bash quant_codegeex_w8a8.sh -src ${浮点权重路径} -dst ${量化权重保存路径} -calib_file ${校准数据集} -trust_remote_code
```
校准数据集从[此处](https://storage.cloud.google.com/boolq/dev.jsonl)获取。

导出量化权重后应生成`quant_model_weight_w8a8.safetensors`和`quant_model_description_w8a8.json`两个文件。

注：

1.quant_codegeex_w8a8.sh文件中已配置好较优的量化策略，导出量化权重时可直接使用，也可修改为其它策略。

2.执行脚本生成量化权重时，会在生成的权重路径的config.json文件中添加(或修改)`quantize`字段，值为相应量化方式，当前仅支持`w8a8`。

3.执行完以上步骤后，执行量化模型只需要替换权重路径。

4.如果生成权重时遇到`OpenBLAS Warning: Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP = 1 option`，可通过设置`export OMP_NUM_THREADS=1`来关闭多线程规避。


## 精度测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## FAQ
- `import torch_npu`遇到`xxx/libgomp.so.1: cannot allocate memory in static TLS block`报错，可通过配置`LD_PRELOAD`解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`