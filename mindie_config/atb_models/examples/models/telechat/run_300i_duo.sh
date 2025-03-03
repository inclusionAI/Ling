#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TP_WORLD_SIZE=2
export MASTER_PORT=12888
export PYTHONPATH=${llm_path}:$PYTHONPATH

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1

if [ -z "$1" ]; then
    echo "Error: model path is required."
    exit 1
fi

model_path="$1"
shift

trust_remote_code="False"
# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -trust_remote_code)
            trust_remote_code="True"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$trust_remote_code" == "True" ]; then
    extra_param="--trust_remote_code"
fi


torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.run_pa --model_path $model_path --max_batch_size 1 --max_input_length 256 --max_output_length 256 --block_size 96 $extra_param