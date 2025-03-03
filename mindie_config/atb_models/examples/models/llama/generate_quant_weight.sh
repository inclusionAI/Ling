#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model tpye and quant type. Support llama2_7b_w8a8, llama2_13b_w8a8, llama2_70b_w8a8, llama2_7b_w8a8s, llama2_13b_w8a8s, llama1_65b/llama2_70b_w8a16."
    echo "-use_kvcache_quant  Whether to use kvcache int8 quant. Default value is false."
    echo "-trust_remote_code  Whether to trust local executable files. Default value is false."
}

use_kvcache_quant=False
trust_remote_code="False"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -src)
            if [[ -n "$2" ]]; then
                src="$2"
                shift
            else
                echo "Error: -src requires a non-empty argument."
                exit 1
            fi
            ;;
        -dst)
            if [[ -n "$2" ]]; then
                dst="$2"
                shift
            else
                echo "Error: -dst requires a non-empty argument."
                exit 1
            fi
            ;;
        -type)
            if [[ -n "$2" ]]; then
                type="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -use_kvcache_quant)
            if [[ -n "$2" ]]; then
                use_kvcache_quant="$2"
                shift
            else
                echo "Error: -use_kvcache_quant requires a non-empty argument."
                exit 1
            fi
            ;;
        -trust_remote_code)
            trust_remote_code="True"
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# 参数校验
if [[ -z "$src" ]]; then
    echo "Error: Missing required option: -src"
    show_help
    exit 1
fi

if [[ -z "$dst" ]]; then
    echo "Error: Missing required option: -dst"
    show_help
    exit 1
fi

if [[ -z "$type" ]]; then
    echo "Error: Missing required option: -type"
    show_help
    exit 1
fi

# 设置环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 进入运行路径
cd ${ATB_SPEED_HOME_PATH}

param=""

get_down_proj_disable_name() {
    config_file="$src/config.json"
    num_layers=$(jq '.num_hidden_layers' "$config_file")
    for ((i=0; i<num_layers; i++)); do
        disable_names="$disable_names model.layers.$i.mlp.down_proj"
    done
    echo "$disable_names"
}

case "$type" in
    llama2_7b_w8a8)
        disable_names=$(get_down_proj_disable_name)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type cpu --disable_level L0 --anti_method m1 --act_method 1 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    llama2_13b_w8a8)
        disable_names=$(get_down_proj_disable_name)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type cpu --disable_level L0 --anti_method m2 --act_method 1 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    llama2_70b_w8a8)
        disable_names=$(get_down_proj_disable_name)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type npu --w_bit 8 --a_bit 8 --disable_level L5 --device_type npu --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    llama2_7b_w8a8s)
        disable_names=$(get_down_proj_disable_name)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/teacher_qualification.jsonl --disable_names $disable_names --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    llama2_13b_w8a8s)
        disable_names=$(get_down_proj_disable_name)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/teacher_qualification.jsonl --disable_names $disable_names --w_bit 4 --a_bit 8 --fraction 0.011 --co_sparse True --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    llama1_33b_w8a8s)
        disable_names=$(get_down_proj_disable_name)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --act_method 2 --do_smooth True --use_sigma True --is_lowbit True --co_sparse True --w_bit 4 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    llama1_65b/llama2_70b_w8a16)
        param='--calib_file= --w_bit 8 --a_bit 16 --act_method 3 --tokenizer_args {"padding_side":"left","pad_token":"<unk>"}'
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

if [ "$trust_remote_code" == "True" ]; then
    param="$param --trust_remote_code"
fi

python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param
