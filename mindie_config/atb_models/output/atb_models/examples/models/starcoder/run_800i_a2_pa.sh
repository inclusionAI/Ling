#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12343

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export LCCL_ENABLE_FALLBACK=1

extra_param=""
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_pa --input_text "def print_hello_world()" --model_path "$1" $extra_param --max_output_length 256
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_pa --input_text "def print_hello_world()" --model_path "$1" $extra_param --max_output_length 256
fi
