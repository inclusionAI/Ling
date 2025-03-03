#!/bin/bash
export BIND_CPU=1
export RESERVED_MEMORY_GB=3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=12347
export TP_WORLD_SIZE=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) +1))
model_path=""
is_chat_model="false"
# embedding model. example: gte-qwen2
is_embedding_model="false"
trust_remote_code="false"

function usage(){
    echo "$0 pls. use '-m|--model-path' input model path"
    exit 1
}

if [[ $# -eq 0 ]];then
        usage
fi

GETOP_ARGS=$(getopt -o m:c:t:e:r: -al model-path:,is-chat-model:,chat_template:,is_embedding_model:,trust_remote_code: -- "$@")
eval set -- "${GETOP_ARGS}"
while [ -n "$1" ]
do
    case "$1" in
        -m|--model-path) model_path=$2;shift 2;;
        -c|--is-chat-model) is_chat_model=$2;shift 2;;
        -t|--chat_template) chat_template=$2;shift 2;;
        -e|--is_embedding_model) is_embedding_model=$2;shift 2;;
        -r|--trust_remote_code) trust_remote_code=$2;shift 2;;
        --) shift;break;;
        *) usage;break;;
    esac
done

atb_options="ATB_LAUNCH_KERNEL_WITH_TILING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF='max_split_size_mb:2048' HCCL_BUFFSIZE=120 ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1"
atb_async_options="ATB_OPERATION_EXECUTE_ASYNC=1 TASK_QUEUE_ENABLE=1"
base_cmd="torchrun --nproc_per_node $TP_WORLD_SIZE --master_port $MASTER_PORT -m examples.run_pa --model_path $model_path"
if [[ ${is_chat_model} = "true" ]];then
    base_cmd="${base_cmd} --is_chat_model --chat_template \"${chat_template}\""
fi
if [[ ${trust_remote_code} = "true" ]];then
    base_cmd="${base_cmd} --trust_remote_code"
fi
run_cmd="${atb_options} ${atb_async_options} ${base_cmd}"

if [[ -n ${model_path} && "${is_embedding_model}" == "true" ]];then
    echo "INFO: For gte model, only output an embedding tensor."
    ${base_cmd}  --max_output_length 1 --is_embedding_model
elif [[ -n ${model_path} ]]; then
    eval "${run_cmd}"
fi