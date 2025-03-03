#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# shellcheck disable=SC2148

export BIND_CPU=1
export RESERVED_MEMORY_GB=3
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MASTER_PORT=20050
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
DATE=$(date "+%Y-%m-%d")

export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0

DATASET=$1
if [ -z "$1" ]
then
    echo "arg1: DATASET is required"
    exit 1
fi 

# 定义允许的数据集名称列表
VALID_DATASETS=("textvqa" "videobench" "vocalsound")

# 检查DATASET是否在允许的模型名称列表中
if [[ " ${VALID_DATASETS[@]} " =~ " ${DATASET} " ]]; then
    echo "$DATASET is supported!"
else
    echo "Unsupported dataset : $DATASET!"
    exit 1
fi

MODEL_NAME=$2
if [ -z "$2" ]
then
    echo "arg2: MODEL_NAME is required"
    exit 1
fi 

# 定义允许的模型名称列表
VALID_MODELS=(
    "cogvlm2"
    "internvl"
    "qwen_vl"
    "qwen2_audio"
    "qwen2_vl"
    "yivl"
)

# 检查MODEL_NAME是否在允许的模型名称列表中
if [[ " ${VALID_MODELS[@]} " =~ " ${MODEL_NAME} " ]]; then
    echo "$MODEL_NAME is running!"
else
    echo "Unsupported model : $MODEL_NAME!"
    exit 1
fi


atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
base_cmd="
    torchrun \
    --nproc_per_node $TP_WORLD_SIZE \
    --master_port $MASTER_PORT \
    -m modeltest \
    --model_config_path modeltest/config/model/$MODEL_NAME.yaml \
    --task_config_path modeltest/config/task/$DATASET.yaml \
    --output_dir output/$DATE/modeltest/$MODEL_NAME/precision_result/ \
    --tp $TP_WORLD_SIZE \
    --batch_size 1 \
    --lcoc_disable \
    --save_debug_enable
    "

run_cmd="${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${MODEL_NAME} ]];then
    eval "${run_cmd}"
fi
