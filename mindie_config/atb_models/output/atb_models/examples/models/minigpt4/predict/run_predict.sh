#!/bin/bash

# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

# 此脚本的手动配置参数
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=20030

flag_save_log=false

minigpt_dir="${work_space}"
LLM_model_path="${model_path}/weights_language"

min_length=1
max_new_tokens=300
stop_words_ids="[[835],[2277,29937]]" # 注意里面不能有空格
do_sample=False
num_beams=1
top_p=0.9
temperature=0.1
repetition_penalty=1.05
length_penalty=1

# 此脚本的自动配置参数
world_size=$(($(echo "${ASCEND_RT_VISIBLE_DEVICES}" | grep -o , | wc -l) + 1))
cur_time=$(date +"%Y-%m-%d_%H-%M-%S")
inputs_embeds_dir="inputs_embeds_dir_${cur_time}"
results_save_path="results_save_path_${cur_time}.json"

# 注册 python 环境变量
export PYTHONPATH="${minigpt_dir}:${PYTHONPATH}"
export PYTHONPATH="${ATB_SPEED_HOME_PATH}:${PYTHONPATH}"

# step 1/2: 图文输入 → LLM_inputs（单进程，调用 minigpt）
params_1=""
params_1="${params_1} --cfg_path ${minigpt_dir}/eval_configs/minigpt4_eval.yaml"
if [ "${1}" != "" ]; then
    params_1="${params_1} --image_path ${1}"
else
    params_1="${params_1} --image_path ${minigpt_dir}/examples_v2/office.jpg"
fi
params_1="${params_1} --inputs_embeds_dir ${inputs_embeds_dir}"
python -m make_embeds ${params_1}
if [ $? -ne 0 ]; then
    exit
fi

# 以下环境变量与性能和内存优化相关，通常情况下无需修改
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_CONVERT_NCHW_TO_ND=1
export HCCL_BUFFSIZE=120
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_SIZE=0

# 打印加速库、算子库日志
if [ "${flag_save_log}" = true ]; then
    export ATB_LOG_LEVEL=INFO
    export ATB_LOG_TO_STDOUT=1
    export ATB_LOG_TO_FILE=1

    export ASDOPS_LOG_LEVEL=INFO
    export ASDOPS_LOG_TO_STDOUT=1
    export ASDOPS_LOG_TO_FILE=1

    export TASK_QUEUE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=0
    export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1
fi

# step 2/2: LLM_inputs → 文字输出（可多进程，调用 llama）
params_2=""

params_2="${params_2} --model_path ${LLM_model_path}
                      --results_save_path ${results_save_path}"

params_2="${params_2} --inputs_embeds_dir ${inputs_embeds_dir}
                      --min_length ${min_length}
                      --max_output_length ${max_new_tokens}
                      --stop_words_ids ${stop_words_ids}
                      --do_sample ${do_sample}
                      --num_beams ${num_beams}
                      --top_p ${top_p}
                      --temperature ${temperature}
                      --repetition_penalty ${repetition_penalty}
                      --length_penalty ${length_penalty}"

if [ "${TP_WORLD_SIZE}" == "1" ]; then
    python -m examples.run_fa ${params_2}
else
    torchrun --nproc_per_node ${world_size} --master_port ${MASTER_PORT} -m examples.run_fa ${params_2}
fi

