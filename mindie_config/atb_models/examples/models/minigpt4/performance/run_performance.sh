#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

# 此脚本的手动配置参数
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20030

LLM_model_path="${model_path}/weights_language"

case=256
bsz_base=600

#case=512
#bsz_base=300

#case=1024
#bsz_base=150

#case=2048
#bsz_base=75

# 注册 python 环境变量
export PYTHONPATH="${ATB_SPEED_HOME_PATH}:${PYTHONPATH}"

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export HCCL_BUFFSIZE=120
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_SIZE=0

for ((bsz = bsz_base; bsz < bsz_base + 20; bsz++)); do
    extra_param=""
    extra_param="${extra_param} --max_position_embeddings $((case * 2))
                                --max_input_length $case
                                --max_output_length $case
                                --batch_size $bsz"

    echo ${extra_param}

    if [ "$TP_WORLD_SIZE" == "1" ]; then
        python -m examples.run_fa --model_path $LLM_model_path $extra_param
    else
        world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) + 1))
        torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_fa --model_path $LLM_model_path $extra_param
    fi
    if [ $? -ne 0 ]; then
        exit
    fi
done
