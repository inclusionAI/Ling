#!/bin/bash

# 定义模型检查点和保存目录
onnx_directory="$1"
om_directory="$2"
soc_version=$(python -c "import torch;import torch_npu;print(torch.npu.get_device_name())")

# 检查是否输入了转换精度参数
if [ -z "$3" ]; then
    precision_mode=mixed_float16
else
    precision_mode="$3"
fi

# 检查ONNX模型是否存在
if [ -f "$onnx_directory/model.onnx" ]; then
    echo "Prepared convert ONNX model to OM model."
else
    echo "Error: Unable to find ONNX model."
    exit 1
fi

# 使用ATC命令对ONNX模型进行转换或优化
cd "$onnx_directory" || exit
atc --model="model.onnx" \
    --framework=5 \
    --output="$om_directory/bge-reranker-large" \
    --soc_version="$soc_version" \
    --input_shape="input_ids:-1,-1;attention_mask:-1,-1" \
    --precision_mode_v2="$precision_mode" \
    --modify_mixlist="ops_info.json"

# 检查ATC命令是否执行成功
# shellcheck disable=SC2181
if [ $? -eq 0 ]; then
    echo "Model conversion with ATC successful."
else
    echo "Error: Failed to convert model with ATC."
    exit 1
fi
