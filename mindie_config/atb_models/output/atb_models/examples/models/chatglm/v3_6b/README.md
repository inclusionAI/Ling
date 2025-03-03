# ChatGLM3-6B 模型推理指导 <!-- omit in toc -->

# 概述

- ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的对话预训练模型。ChatGLM3-6B 是 [ChatGLM3]((https://github.com/THUDM/ChatGLM3)) 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 有更强大的基础模型、更完整的功能支持、和更全面的开源序列。
- 此代码仓中实现了一套基于NPU硬件的ChatGLM3-6B推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了ChatGLM3-6B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|-----|-----|
| ChatGLM3-6B    | 支持world size 1,2,4,8 | 支持world size 1,2,4 | 是   | 否   | 否              | 是              | 是      | 否     | 否           | 否 | 否     | 是     | 否 | 否 |
| ChatGLM3-6B-32K    | 支持world size 1,2,4,8 | 支持world size 1,2,4 | 是   | 否   | 否              | 是              | 否      | 否     | 否           | 否 | 否     | 是     | 否 | 否 |

- 此模型仓已适配的模型版本
  - [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b/tree/main)
  - [ChatGLM3-6B-32K](https://huggingface.co/THUDM/chatglm3-6b-32k/tree/main)
  - 注：ChatGLM3-6B 推荐使用commit id为 `a5ba5501eb873d40d48bd0983bd2a8dd006bb838` 的模型仓版本


# 使用说明

- 参考[此README文件](../../chatglm/v2_6b/README.md)

## 精度测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../../tests/modeltest/README.md)

## FAQ
- `import torch_npu`遇到`xxx/libgomp.so.1: cannot allocate memory in static TLS block`报错，可通过配置`LD_PRELOAD`解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`