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
SCRIPT_DIR=$(cd $(dirname $0); pwd)
BASE_DIR=$(cd $SCRIPT_DIR/base/; pwd)
TESTS_DIR=$(cd $SCRIPT_DIR/core/; pwd)
DATASET_DIR=$(cd $SCRIPT_DIR/dataset/; pwd)
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

test_mode="performance"
model_type="pa"
model_name=""
weight_dir=""
data_type="fp16"
chip_num=0
dataset="CEval"
shot=-1
batch_size=0
case_pair="[]"
max_position_embedding=-1
time_limit=0
batch_range="[1,2000]"
res_lst_str=""
is_multinode=0
local_world_size=0
rank_id_start=0
input_text_or_file=""
is_chat_instruct_model="False"
context_length=0
kw_args=""
prefill_batch_size=0
trust_remote_code=0

function accumulate_res()
{
    utils_file="utils.py"
    utils_path="${BASE_DIR}/${utils_file}"
    if [[ ! -e "$utils_path" ]];then
        echo "utils file $utils_path is not found."
        exit 0
    fi
    python3 -m base.utils \
    --mode "$1" \
    --res_list "$res_lst_str"
}

function fn_prepare()
{
    if [ "$hardware_type" == "NPU" ]; then
        if [ -z "$ASCEND_HOME_PATH" ];then
            echo "env ASCEND_HOME_PATH not exists, fail"
            exit 0
        fi
        if [ -z "$ATB_HOME_PATH" ];then
            echo "env ATB_HOME_PATH not exists, fail"
            exit 0
        fi
    fi

    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
    atb_models_home_path="$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")"
    export PYTHONPATH="$atb_models_home_path:$PYTHONPATH"
    
    IFS="_"
    read -ra parts <<< "$1"
    model_type="${parts[0]}"
    echo "INFO: current model_type: $model_type"
    if [ "$model_type" == "pa" ]; then
        data_type="${parts[1]}"
        echo "INFO: current data_type: $data_type"
    fi

    test_mode="$2"
    if [ "$test_modes" == "full_HumanEval_X" ]; then
        tar -zxf "${DATASET_DIR}/full/HumanEval_X/go/evaluation/vendor.tar.gz" -C "${DATASET_DIR}/full/HumanEval_X/go/evaluation"
        export GOFLAGS=-mod=vendor
    fi

    if ! [[ "$test_mode" == performance* || "$test_mode" == "precision_single" ]]; then
        test_mode="$(echo "$2" | cut -d'_' -f1)"
        dataset="$(echo "$2" | cut -d'_' -f2-)"
        echo "INFO: current test_mode: $test_mode"
        echo "INFO: current dataset: $dataset"
        export MODELTEST_DATASET_SPECIFIED="$dataset"
    fi

    if [ "$MODELTEST_LOG_TO_FILE" = "1" ] && [ -z "$MODELTEST_LOG_FILENAME" ]; then
        logfilename="${test_mode}_${TIMESTAMP}.log"
        export MODELTEST_LOG_FILENAME="${logfilename}"
    fi

    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
    if [[ "$test_mode" == performance* ]]; then
        export ATB_LLM_BENCHMARK_ENABLE=1
        export ATB_LLM_BENCHMARK_FILEPATH="${SCRIPT_DIR}/benchmark.csv"
    else
        export LCCL_DETERMINISTIC=1
        export HCCL_DETERMINISTIC=true
        export ATB_MATMUL_SHUFFLE_K_ENABLE=0
        export ATB_LLM_LCOC_ENABLE=1
    fi
}

function fn_run_single()
{
    test_file="${model_name}_test.py"
    test_path="${TESTS_DIR}/${test_file}"
    if [[ ! -e "$test_path" ]];then
        echo "model test file $test_path is not found."
        exit 0
    fi
    
    if [ "$chip_num" == 0 ]; then
        code_line=$(grep -A 1 "def get_chip_num(self):" "${test_path}" | tail -n 1)
        if [ -z "$code_line" ]; then
            echo "Warning: get_chip_num() not overwrite in '$test_file', use chip_num 1"
            chip_num=1
        else
            chip_num=$(echo "$code_line" | awk -F 'return ' '{print $2}')
            if ! [[ "$chip_num" =~ ^[1-9]+$ ]]; then
                echo "Error: return value of get_chip_num() in '$test_file' is not a digit."
                exit 1
            fi
        fi
    fi

    if  [ "$hardware_type" == "NPU" ]; then
        if [ "$is_multinode" == 0 ]; then
            if ! [ -n "$ASCEND_RT_VISIBLE_DEVICES" ]; then
                devices=""
                for ((i=0; i<chip_num-1; i++)); do
                    devices+="$i,"
                done
                devices+="$((chip_num-1))"
                export ASCEND_RT_VISIBLE_DEVICES="$devices"
            fi
            if [ -n "$RANKTABLEFILE" ]; then
                echo "ERROR: Using single node mode, RANKTABLEFILE has been set as"
                cat "$RANKTABLEFILE"
                echo "ERROR: Please unset RANKTABLEFILE and try again"
            fi
            random_port=$(( RANDOM  % 9999 + 10001 ))
            torchrun --nproc_per_node "$chip_num" --master_port $random_port "$test_path" \
            --model_type "$model_type" \
            --data_type "$data_type" \
            --test_mode "$test_mode" \
            --batch_size "$batch_size" \
            --model_name "$model_name" \
            --weight_dir "$weight_dir" \
            --dataset_name "$dataset" \
            --shot "$shot" \
            --hardware_type $hardware_type \
            --case_pair "$case_pair" \
            --time_limit "$time_limit" \
            --max_position_embedding "$max_position_embedding" \
            --input_text_or_file "$input_text_or_file" \
            --is_chat_instruct_model "$is_chat_instruct_model" \
            --context_length "$context_length" \
            --lora_data_path "$lora_data_path" \
            --kw_args "$kw_args" \
            --prefill_batch_size "$prefill_batch_size" \
            --trust_remote_code $trust_remote_code
            wait
        else
            MASTER_ADDR=$master_addr
            GPUS_PER_NODE=$local_world_size
            MASTER_PORT=12345
            NNODES=$node_num
            NODE_RANK=$node_rank
            WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

            python3 ./utils/params_checker.py \
            "$rank_table_file" \
            "$world_size" \
            "$node_num" \
            "$master_addr"

            torchrun \
            --nproc_per_node $GPUS_PER_NODE \
            --master_port $MASTER_PORT \
            --nnodes $NNODES \
            --node_rank $NODE_RANK \
            --master_addr $MASTER_ADDR \
            "$test_path" \
            --model_type "$model_type" \
            --data_type "$data_type" \
            --test_mode "$test_mode" \
            --batch_size "$batch_size" \
            --model_name "$model_name" \
            --weight_dir "$weight_dir" \
            --dataset_name "$dataset" \
            --shot "$shot" \
            --hardware_type $hardware_type \
            --case_pair "$case_pair" \
            --time_limit "$time_limit" \
            --max_position_embedding "$max_position_embedding" \
            --input_text_or_file "$input_text_or_file" \
            --is_chat_instruct_model "$is_chat_instruct_model" \
            --context_length "$context_length" \
            --prefill_batch_size "$prefill_batch_size" \
            --trust_remote_code $trust_remote_code
        fi
    else
        if ! [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            world_size_str=$(seq -s, 0 $((chip_num-1)))
            export CUDA_VISIBLE_DEVICES=$world_size_str
        fi
        export WORLD_SIZE=$chip_num
        echo "using cuda device $CUDA_VISIBLE_DEVICES"
        python3 "$test_path" \
        --model_type "$model_type" \
        --data_type "$data_type" \
        --test_mode "$test_mode" \
        --batch_size "$batch_size" \
        --model_name "$model_name" \
        --weight_dir "$weight_dir" \
        --dataset_name "$dataset" \
        --shot "$shot" \
        --hardware_type $hardware_type \
        --case_pair "$case_pair" \
        --time_limit "$time_limit" \
        --max_position_embedding "$max_position_embedding" \
        --input_text_or_file "$input_text_or_file" \
        --is_chat_instruct_model "$is_chat_instruct_model" \
        --context_length "$context_length" \
        --trust_remote_code $trust_remote_code
    fi

    if [ $? -ne 0 ]; then
        echo "something wrong marked for CI"
        if [ "$test_modes" == "performance" ]; then
            echo "performance test end marked for CI"
        else
            echo "precision test end marked for CI"
        fi
    fi
}

function fn_run_maxbs
{
    case_pair_lst=()
    batch_range_lst=()

    if ! [[ $case_pair =~ ^\[\[[0-9]+,[0-9]+\](,\[[0-9]+,[0-9]+\])*\]$ ]]; then
        echo "Incorrect case_pair format"
        exit 1
    else
        pattern='\[[0-9]+,[0-9]+\]'
        while [[ $case_pair =~ $pattern ]]; do
            case_pair_lst+=("${BASH_REMATCH[0]}")
            case_pair=${case_pair#*"${BASH_REMATCH[0]}"}
        done
    fi
    if [[ $batch_range =~ ^\[\[[0-9]+,[0-9]+\](,\[[0-9]+,[0-9]+\])*\]$ ]]; then
        while read -r batch; do
            batch_range_lst+=("$batch")
        done < <(echo "$batch_range" | grep -o '[0-9]\+')
    fi

    len_case_pair_lst=${#case_pair_lst[@]}
    len_batch_range_lst=${#batch_range_lst[@]}
    declare -a maxbs_lst
    declare -a maxbs_res_lst
    for ((i=0;i<$len_case_pair_lst;i++)); do
        maxbs_lst[i]=0
        maxbs_res_lst[i]=""
    done
    if ! (( len_batch_range_lst == len_case_pair_lst * 2 )); then
        echo "num of case_pair and batch_range not match"
        exit 1
    fi
    round=0
    for case_pair in "${case_pair_lst[@]}"; do
        rounds_res_lst=()
        case_pair=$case_pair
        left_bound=${batch_range_lst[0]}
        right_bound=${batch_range_lst[1]}
        batch_range_lst=("${batch_range_lst[@]:2}")
        while (( $left_bound<=$right_bound ))
        do
            sum=$(expr $left_bound + $right_bound)
            batch_size=$(expr $sum / 2)
            echo "INFO: left_bound: $left_bound, right_bound: $right_bound, curr_bs: $batch_size, case_pair: $case_pair"
            fn_run_single
            lines=()
            while IFS= read -r line; do
                lines+=("$line")
            done < "maxbs.txt"
            rounds_res_lst+=("${lines[2]}")
            if [[ "${lines[0]}" == "1" || "${lines[1]}" == "0" ]]; then
                right_bound=$(expr $batch_size - 1)
            else
                left_bound=$(expr $batch_size + 1)
                maxbs_lst[$(($round))]=$batch_size
                maxbs_res_lst[$(($round))]="${lines[2]}"
            fi
        done
        res_lst_str=$(IFS=','; echo "${rounds_res_lst[*]}")
        accumulate_res "round"
        round=$((round + 1))
    done

    for ((i=0; i<$len_case_pair_lst; i++)); do
        echo "${case_pair_lst[i]}: ${maxbs_lst[i]}"
    done
    res_lst_str=$(IFS=','; echo "${maxbs_res_lst[*]}")
    accumulate_res "final"
}


function fn_run
{
    if [ "$test_modes" == "performance_maxbs" ]; then
        fn_run_maxbs
    else
        fn_run_single
    fi
}

function fn_main()
{
    hardware_type=$(python -c "
import torch
if torch.cuda.is_available():
    hardware_type = 'GPU'
else:
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            hardware_type = 'NPU'
    except ImportError:
        hardware_type = 'None'
print(hardware_type)
")

    if [[ "$hardware_type" =~ "NPU" ]]; then
        hardware_type="NPU"
        echo "INFO: Detected Ascend NPU"
    elif [[ "$hardware_type" =~ "GPU" ]]; then
        hardware_type="GPU"
        echo "INFO: Detected NVIDIA GPU"
    else
        echo "Error: No GPU or NPU detected"
        exit 1
    fi

    if [ $# -eq 0 ]; then
        echo "Error: require parameter. Please refer to README."
        exit 1
    fi

    model_type=$1
    case "$model_type" in
        fa|pa_fp16|pa_bf16)
            echo "INFO: current model_type: $model_type"
            ;;
        *)
            echo "ERROR: invalid model_type, only support fa, pa_fp16, pa_bf16"
            ;;
    esac
    test_modes=$2
    case "$test_modes" in
        performance|performance_maxbs|performance_single|precision_single|simplified_GSM8K|simplified_TruthfulQA|full_CEval|full_GSM8K|full_MMLU|full_TruthfulQA| \
        full_BoolQ|full_HumanEval|full_HumanEval_X|full_LongBench|full_LongBench-E|full_CMMLU|full_NeedleBench|lite_BoolQ|lite_GSM8K)
            echo "INFO: current test_mode: $test_modes"
            ;;
        *)
            echo "ERROR: invalid test_mode, only support performance, performance_maxbs, performance_single, precision_single, simplified_GSM8K, simplified_TruthfulQA, \
            full_CEval, full_GSM8K, full_MMLU, full_TruthfulQA, full_BoolQ, full_HumanEval, full_HumanEval_X, full_LongBench, full_LongBench-E, full_CMMLU, full_NeedleBench, lite_BoolQ, lite_GSM8K"
            exit 1
            ;;
    esac

    if [[ "$test_modes" == performance* || "$test_modes" == "precision_single" ]]; then
        case_pair=$3
        echo "INFO: current case_pair: $case_pair"
        shift
    fi

    if [[ "$test_modes" == "full_CEval" || "$test_modes" == "full_MMLU" || "$test_modes" == "full_CMMLU" ]]; then
        shot=$3
        echo "INFO: use shot: $shot"
        shift
    fi

    if [[ "$test_modes" == "full_NeedleBench" ]]; then
        context_length=$3
        echo "INFO: use context length: $context_length"
        shift
    fi

    if [ "$test_modes" == "performance_maxbs" ]; then
        if [[ "$3" =~ ^\[.*\]$ ]]; then
            batch_range=$3
            echo "INFO: current batch_range: $batch_range"
            shift
        fi
        time_limit=$3
        echo "INFO: current time_limit: $time_limit"
    else 
        if [[ "$test_modes" == "performance_single" || "$test_modes" == "precision_single" ]]; then
            input_text_or_file="$3"
            echo "INFO: current input_text_or_file: $input_text_or_file"
            shift
        fi
        batch_size=$3
        echo "INFO: current batch_size: $batch_size"
    fi

    if [[ "$test_modes" == "performance" && "$4" =~ ^[0-9]+$ ]]; then
        prefill_batch_size=$4
        echo "INFO: current prefill_batch_size: $prefill_batch_size"
        shift
    fi

    model_name=$4

    if [[ "$5" == "chat" || "$5" == "base" || "$5" == "instruct" ]]; then
        is_chat_instruct_model=$5
        echo "INFO: current use $model_name version: $is_chat_instruct_model"
        shift
    fi

    if [[ "$5" == "lora" ]]; then
        lora_data_path=$6
        echo "INFO: current use $model_name lora data path: $lora_data_path"
        shift
        shift
    fi

    weight_dir=$5
    echo "INFO: current model_name: $model_name"
    echo "INFO: current weight_dir: $weight_dir"

    fn_prepare "$model_type" "$test_modes"

    if [[ "$6" == "trust_remote_code" ]]; then
        trust_remote_code=1
        echo "INFO: current trust_remote_code: True"
        shift
    else
        echo "INFO: current trust_remote_code: False"
    fi

    if [[ "$6" == *"encrypt"* ]]; then
        kw_args=$6
        echo "kw_args is: $kw_args"
        shift
    fi


    if ! [[ "$6" =~ ^[1-9]+$ ]]; then
        is_multinode=1
        rank_table_file=$6
        echo "INFO: current using multiple node, use rank table file: $rank_table_file"
        export RANKTABLEFILE=$rank_table_file
        export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        shift
        if ! [[ "$6" =~ ^[1-9]+$ ]]; then
            echo "ERROR: world_size should be a digit"
            exit 1
        fi
        world_size=$6
        chip_num=$world_size
        export WORLD_SIZE=$world_size
        echo "INFO: current world_size: $world_size"
        shift
        if ! [[ "$6" =~ ^[1-9]+$ ]]; then
            echo "ERROR: node_num should be a digit"
            exit 1
        fi
        node_num=$6
        echo "INFO: current node_num: $node_num"
        if ! (( world_size % node_num == 0 )); then
            echo "ERROR: only support world_size is able to be divided by node_num"
            exit 1
        fi
        local_world_size=$((world_size / node_num))
        echo "INFO: current local_world_size: $local_world_size"
        shift
        if ! [[ "$6" =~ ^[0-9]+$ ]]; then
            echo "ERROR: rank_id_start should be a digit"
            exit 1
        fi
        rank_id_start=$6
        echo "INFO: current rank_id_start: $rank_id_start"
        temp=$world_size-$rank_id_start
        node_rank=$((node_num - temp / local_world_size))
        echo "INFO: current node_rank: $node_rank"
        shift
        master_addr=$6
        echo "INFO: current master_addr: $master_addr"
    else
        chip_num=$6
        echo "INFO: current using single node, use input chip_num $chip_num"
    fi
    
    if [ $# -ge 7 ]; then
        if ! [[ "$7" =~ ^[0-9]+$ ]]; then
            echo "Error: input max_position_embedding or max_seq_len is not a digit."
            exit 1
        fi
        max_position_embedding=$7
        echo "INFO: use input max_position_embedding or max_seq_len $max_position_embedding"
    fi
    fn_run
}

fn_main "$@"
