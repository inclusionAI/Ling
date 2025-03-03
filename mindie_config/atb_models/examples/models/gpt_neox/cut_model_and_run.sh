#!/bin/bash
input_dir="/home/lfy/LM_trans/gptneox20b/model"
output_dir="/home/lfy/LM_trans/gptneox20b/model/part_model"
world_size_=2
task_name=${1-inference}

export PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048'
atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 HCCL_OP_BASE_FFTS_MODE_ENABLE=1 HCCL_BUFFSIZE=110"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
options_cmd="${atb_options} ${atb_async_options}"

if [[ -d "${output_dir}" ]];then
    echo "**********************The gpt-neox-20b part model exists, Now begin to run ...**********************"
    env $options_cmd torchrun --nproc_per_node $world_size_ --master_port 20001 main.py --task $task_name
else
    echo "The gpt-neox-20b part model is not exists, Now begin to cut ..."
    python ./cut_model_util.py --input_path $input_dir --output_path $output_dir --world_size $world_size_
fi