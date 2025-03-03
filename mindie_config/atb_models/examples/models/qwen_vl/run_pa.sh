#!/bin/bash
export BIND_CPU=1
export RESERVED_MEMORY_GB=3
export ASCEND_RT_VISIBLE_DEVICES=0
export MASTER_PORT=20030
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

model_path="/data/Qwen-VL"
image_path=""
atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
base_cmd="torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT \
    -m examples.models.qwen_vl.run_pa \
    --model_path $model_path \
    --input_image ${image_path} \
    --input_text 'Generate the caption in English with grounding:' \
    --trust_remote_code "
run_cmd="${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${model_path} ]];then
    eval "${run_cmd}"
fi
