#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20030

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export INF_NAN_MODE_ENABLE=0

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

world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_pa --model_path "$model_path" $extra_param
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_pa --model_path "$model_path" $extra_param
fi