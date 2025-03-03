#!/bin/bash
export BIND_CPU=1
export RESERVED_MEMORY_GB=0
export ASCEND_RT_VISIBLE_DEVICES=0
export MASTER_PORT=20031
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

# 开启确定性计算
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0

model_path=""
max_batch_size=1
max_input_length=4096 #输入长视频或者较大分辨率图片时，需要设置较大的值，以便支持更长的输入序列
max_output_length=80
input_image=""
dataset_path=""
input_text="Explain the details in the image."
shm_name_save_path="./shm_name.txt"

benchmark_options="ATB_LLM_BENCHMARK_ENABLE=1"
atb_options="ATB_PROFILING_ENABLE=0 ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"

base_cmd="torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT \
    -m examples.models.qwen2_vl.run_pa \
    --model_path $model_path \
    --shm_name_save_path $shm_name_save_path \
    --max_input_length $max_input_length \
    --max_output_length $max_output_length \
    --max_batch_size $max_batch_size \
    --input_image $input_image \
    --input_text '${input_text}'"
run_cmd="${benchmark_options} ${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${model_path} ]];then
    eval "${run_cmd}"
fi