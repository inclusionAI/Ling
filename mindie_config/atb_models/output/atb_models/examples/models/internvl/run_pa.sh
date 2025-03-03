#!/bin/bash
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20036

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=1
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1

IMAGE_PATH=/data/acltransformer_testdata/images/internvl
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [ $# -eq 0 ]; then
        echo "Error: require parameter. Please refer to README."
        exit 1
fi


if [[ ! -z "$1" ]];then
    RUN_OPTION=$1
    echo "[RUN_OPTION]: $RUN_OPTION"
    shift
fi

if [[ "$1" == "--trust_remote_code" ]]; then
    TRUST_REMOTE_CODE="--trust_remote_code"
    echo "[TRUST_REMOTE_CODE]: true"
    shift
else
    TRUST_REMOTE_CODE=""
fi

if [[ ! -z "$1" ]];then
    MODEL_PATH=$1
    echo "[MODEL_PATH]: $MODEL_PATH"
    shift
fi

if [[ ! -z "$1" ]];then
    IMAGE_PATH=$1
    echo "[IMAGE_PATH]: $IMAGE_PATH"
    shift
fi

case "${RUN_OPTION}" in
    "--run")
    extra_param=""
    extra_param="${extra_param} --model_path $MODEL_PATH
                                --image_or_video_path $IMAGE_PATH"
    extra_param="${extra_param} $TRUST_REMOTE_CODE"
    if [ "$TP_WORLD_SIZE" == "1" ]; then
        python -m examples.models.internvl.internvl $extra_param
    else
        torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.internvl.internvl $extra_param
    fi
    ;;
     "--performance")
        for bsz in {1..10}; do
            extra_param=""
            extra_param="${extra_param} --model_path $MODEL_PATH
                                        --image_or_video_path $IMAGE_PATH
                                        --max_batch_size $bsz
                                        --max_output_length 256"
            extra_param="${extra_param} $TRUST_REMOTE_CODE"
            if [ "$TP_WORLD_SIZE" == "1" ]; then
                python -m examples.models.internvl.internvl $extra_param --performance
            else
                torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.internvl.internvl $extra_param --performance
            fi
                done
            
            ;;
    "--precision")
    extra_param=""
    extra_param="${extra_param} --model_path $MODEL_PATH
                                --image_or_video_path $IMAGE_PATH"
    extra_param="${extra_param} $TRUST_REMOTE_CODE"
    if [ "$TP_WORLD_SIZE" == "1" ]; then
        python -m examples.models.internvl.internvl $extra_param --prediction_result
    else
        torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.internvl.internvl $extra_param --prediction_result
    fi
    ;;
esac