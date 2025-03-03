#!/bin/bash
input_dir="./vlmo/"
output_dir="${input_dir}/part_model"
world_size=2
CONFIG_NAME=

max_seqence_length=4096
use_launch_kernel_with_tiling=1
atb_operation_execute_async=1
task_queue_enable=1
LONG_SEQ_ENABLE=0
# ATB_LOG_TO_STDOUT=1 ATB_LOG_LEVEL=INFO TASK_QUEUE_ENABLE=0 ASDOPS_LOG_TO_STDOUT=1 ASDOPS_LOG_LEVEL=INFO
atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_OP_BASE_FFTS_MODE_ENABLE=1 HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
start_cmd="torchrun --nproc_per_node 2 --master_port 20001 cut_ascend_vqa.py with "${CONFIG_NAME}"  load_path="${input_dir}" test_only=True"
run_cmd="${atb_options} ${atb_async_options} ${start_cmd}"
# ${atb_options} ${atb_async_options}
echo "**********************float model**********************"
if [[ -d "${output_dir}" ]];then
    echo "Weight directory exists, running......"
    eval "${run_cmd}"
else
    echo "Cut model weights ......"
    python ./cut_model_util.py   --input_path $input_dir --output_path $output_dir --world_size $world_size
fi 
