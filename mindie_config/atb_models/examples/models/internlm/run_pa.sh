#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=20036

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export INF_NAN_MODE_ENABLE=0

extra_param=""
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [[ "$2" == "chat" ]]; then
    extra_param+=" --is_chat_model"
fi

if [ "$TP_WORLD_SIZE" == "1" ]; then
    python -m examples.run_pa --model_path "$1" $extra_param
else
    torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.run_pa --model_path "$1" $extra_param
fi