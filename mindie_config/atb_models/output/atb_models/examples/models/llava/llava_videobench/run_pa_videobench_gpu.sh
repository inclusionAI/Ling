# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# 参数配置以及启动指令的说明见同级目录下的README.md文件
export MASTER_PORT=20034
export CUDA_VISIBLE_DEVICES='1,2'

world_size=$(($(echo "${CUDA_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))

if [[ ! -z "$1" ]];then
    DATASET_NAME=$1
    echo "[DATASET_NAME]: $DATASET_NAME"
    shift
fi

if [[ ! -z "$1" ]];then
   EVAL_QA_ROOT=$1
   echo "[EVAL_QA_ROOT]: $EVAL_QA_ROOT"
   shift
fi

if [[ ! -z "$1" ]];then
    EVAL_VIDEO_ROOT=$1
    echo "[EVAL_VIDEO_ROOT]: $EVAL_VIDEO_ROOT"
    shift
fi
if [[ ! -z "$1" ]];then
   MODEL_PATH=$1
   echo "[MODEL_PATH]: $MODEL_PATH"
   shift
fi

if [[ ! -z "$1" ]];then
   CHAT_CONVERSATION_OUTPUT_FOLDER=$1
   echo "[CHAT_CONVERSATION_OUTPUT_FOLDER]: $CHAT_CONVERSATION_OUTPUT_FOLDER"
   shift
fi

extra_param="${extra_param} --model_path $MODEL_PATH
                           --dataset_name $DATASET_NAME
                           --Eval_QA_root $EVAL_QA_ROOT
                           --Eval_Video_root $EVAL_VIDEO_ROOT
                           --chat_conversation_output_folder $CHAT_CONVERSATION_OUTPUT_FOLDER"
torchrun --nproc_per_node $world_size --master_port $MASTER_PORT -m examples.models.llava.llava_videobench.llava_gpu $extra_param
