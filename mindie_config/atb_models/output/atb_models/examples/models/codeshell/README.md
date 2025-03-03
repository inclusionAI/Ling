# CodeShell-7B 模型推理指导 <!-- omit in toc -->

# 概述

- [CodeShell-7B](https://github.com/WisdomShell/codeshell)是北京大学知识计算实验室联合四川天府银行AI团队研发的多语言代码大模型基座。它拥有70亿参数，经过对五千亿Tokens的训练，并具有8192的上下文窗口长度。CodeShell在权威的代码评估Benchmark（HumanEval与MBPP）上取得了同等规模最好的性能。这个项目为多语言代码处理和理解提供了有力的工具。
- 此代码仓中实现了一套基于NPU硬件的CodeShell推理模型。配合加速库使用，旨在NPU上获得极致的推理性能。

# 特性矩阵
- 此矩阵罗列了CodeShell-7B模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI | 长序列 |
|-------------|-------------------------|-------------------------|------|------|-----------------|-----------------|---------|--------------|----------|--------|--------|-----|-----|-----|
| CodeShell-7B    | 支持world size 1,2,4,8  | 支持world size 1,2      | 是   | 否   | 否              | 是              | 否       | 否          | 否           | 否       | 否     | 否    | 否  | 否 |

- 此模型仓已适配的模型版本
  - [CodeShell-7B](https://huggingface.co/WisdomShell/CodeShell-7B/tree/main)


# 使用说明

- 执行推理前需要将权重目录下的config.json中的`torch_dtype`改为`"float16"`
- 修改config.json中的`model_type`改为`"codeshell"`


## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## FAQ
- `import torch_npu`遇到`xxx/libgomp.so.1: cannot allocate memory in static TLS block`报错，可通过配置`LD_PRELOAD`解决。
  - 示例：`export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`