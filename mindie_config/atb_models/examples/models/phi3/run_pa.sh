#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20030
TP_WORLD_SIZE=1

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_SIZE=0

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
    param="--trust_remote_code"
fi

extra_param="--is_chat_model"
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ "${TP_WORLD_SIZE}" == "1" ]; then
    python -m examples.run_pa --model_path "${model_path}" "${extra_param}" $param
else
    torchrun --nproc_per_node "${world_size}" --master_port "${MASTER_PORT}" -m examples.run_pa --model_path "${model_path}" "${extra_param}" $param
fi