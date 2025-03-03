#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=20036

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export LCCL_ENABLE_FALLBACK=1

if [ -z "$1" ]; then
    echo "Error: model path is required."
    exit 1
fi

model_path="$1"
shift

is_chat_model="False"
trust_remote_code="False"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        chat)
            is_chat_model="True"
            ;;
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

world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

extra_param=""
if [[ "$is_chat_model" == "True" ]]; then
    extra_param+=" --is_chat_model"
fi
if [ "$trust_remote_code" == "True" ]; then
    extra_param+=" --trust_remote_code"
    echo "INFO: current trust_remote_code: True"
fi

if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_pa --model_path "$model_path" $extra_param
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_pa --model_path "$model_path" $extra_param
fi
