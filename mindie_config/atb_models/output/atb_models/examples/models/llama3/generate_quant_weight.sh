#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model tpye and quant type. Support llama3.1_8b_w8a8, llama3.1_70b_instruct_fp16_w8a8, llama3.1_70b_instruct_bf16_w8a8."
    echo "-use_kvcache_quant  Whether to use kvcache int8 quant. Default value is false."
    echo "-use_fa_quant Whether to use attention quant. Default value is false"
}

use_kvcache_quant=False
use_fa_quant=False

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
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
            ;;
        -use_fa_quant)
            if [[ -n "$2" ]]; then
                use_fa_quant="$2"
                shift
            else
                echo "Error: -type requires a non-empty argument."
                exit 1
            fi
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
    local num_layer=$1
    local disable_names=""
    for ((i=0; i<$num_layer; i++)); do
        disable_names="$disable_names model.layers.$i.mlp.down_proj"
    done
    disable_names="$disable_names lm_head"
    echo "$disable_names"
}

get_llama3_1_70b_fp16_disable_name() {
    local num_layer=$1
    local disable_names=""
    for ((i=0; i<$num_layer; i++)); do
        disable_names="$disable_names model.layers.$i.mlp.down_proj"
    done   
    for ((i=0; i<5; i++)); do
    {
        disable_names="$disable_names model.layers.$i.self_attn.q_proj"
        disable_names="$disable_names model.layers.$i.self_attn.k_proj"
        disable_names="$disable_names model.layers.$i.self_attn.v_proj"
        disable_names="$disable_names model.layers.$i.self_attn.o_proj"
        disable_names="$disable_names model.layers.$i.mlp.gate_proj"
        disable_names="$disable_names model.layers.$i.mlp.up_proj"
    }
    done
    disable_names="$disable_names lm_head"
    echo "$disable_names"
}

get_llama3_70b_disable_name() {
    local num_layer=$1
    local disable_names=""
    for ((i=0; i<5; i++)); do
    {
        disable_names="$disable_names model.layers.$i.mlp.down_proj"
        disable_names="$disable_names model.layers.$i.self_attn.q_proj"
        disable_names="$disable_names model.layers.$i.self_attn.k_proj"
        disable_names="$disable_names model.layers.$i.self_attn.v_proj"
        disable_names="$disable_names model.layers.$i.self_attn.o_proj"
        disable_names="$disable_names model.layers.$i.mlp.gate_proj"
        disable_names="$disable_names model.layers.$i.mlp.up_proj"
    }
    done
    disable_names="$disable_names lm_head"
    echo "$disable_names"
}

case "$type" in
    llama3.1_8b_w8a8)
        disable_names=$(get_down_proj_disable_name 32)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type npu --disable_level L0 --anti_method m1 --act_method 1 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    llama3.1_70b_instruct_fp16_w8a8)
        disable_names=$(get_llama3_1_70b_fp16_disable_name 80)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type npu --disable_level L0 --anti_method m3 --act_method 3 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    llama3.1_70b_instruct_bf16_w8a8)
        disable_names=$(get_down_proj_disable_name 80)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type npu --disable_level L5 --anti_method m3 --act_method 3 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant --use_fa_quant $use_fa_quant --fa_amp 0"
        ;;
    llama3_70b_w8a16)
        disable_names=$(get_llama3_70b_disable_name 80)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/convert/model_slim/boolq.jsonl --disable_names $disable_names --device_type npu --a_bit 16 --w_sym False --mm_tensor False --anti_method m3 --act_method 3 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"} --use_kvcache_quant $use_kvcache_quant"
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param
