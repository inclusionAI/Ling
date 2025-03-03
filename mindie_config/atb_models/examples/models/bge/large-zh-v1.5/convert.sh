#!/bin/bash  
  
# 定义模型检查点和保存目录  
model_checkpoint="$1" 
save_directory="$model_checkpoint"
soc_version=$(python -c "import torch;import torch_npu;print(torch.npu.get_device_name())")

precision_mode=allow_mix_precision

# 确保当前模型路径下没有同名的model.onnx文件
if [ -f "$save_directory/model.onnx" ]; then
    echo "Error: model.onnx already exists in the current path"
    exit 1
fi

# 使用Python脚本加载并导出模型到ONNX  
python -c "  
from optimum.onnxruntime import ORTModelForFeatureExtraction
from atb_llm.models.base.model_utils import safe_from_pretrained
  
ort_model = safe_from_pretrained(ORTModelForFeatureExtraction, '$model_checkpoint', export=True, from_transformers=True)
ort_model.save_pretrained('$save_directory')  
"  
  
# 检查ONNX模型是否成功保存  
if [ -f "$save_directory/model.onnx" ]; then  
    echo "ONNX model successfully saved at $save_directory/model.onnx"  
else  
    echo "Error: Failed to save ONNX model."  
    exit 1  
fi  


# 使用ATC命令对ONNX模型进行转换或优化  
atc --model=$save_directory/model.onnx --framework=5 --output=$save_directory/bge-large-zh --soc_version="$soc_version" --input_shape="input_ids:-1,-1;attention_mask:-1,-1;token_type_ids:-1,-1" --precision_mode="$precision_mode"
  
# 检查ATC命令是否执行成功  
if [ $? -eq 0 ]; then  
    echo "Model conversion with ATC successful."  
else  
    echo "Error: Failed to convert model with ATC."  
    exit 1  
fi