#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model tpye and quant type. Support deepseekv2_w8a8_dynamic, deepseekv2_lite_w8a8_dynamic, deepseekv2_w8a16, deepseekv2_lite_w8a16."
}

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

get_disable_name() {
    local num_layer=$1
    local disable_names=""
    for ((i=0; i<$num_layer; i++)); do
        disable_names="$disable_names model.layers.$i.self_attn.kv_b_proj"
    done
    echo "$disable_names"
}

case "$type" in
    deepseekv2_lite_w8a16)
        disable_names=$(get_disable_name 27)
        param="--disable_names $disable_names --part_file_size 5 --device_type cpu --act_method 1 --w_bit 8 --a_bit 16 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    deepseekv2_lite_w8a8_dynamic)
        disable_names=$(get_disable_name 27)
        param="--disable_names $disable_names --part_file_size 5 --device_type cpu --act_method 1 --w_bit 8 --a_bit 8  --is_dynamic True --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    deepseekv2_w8a16)
        disable_names=$(get_disable_name 60)
        param="--disable_names $disable_names --part_file_size 5 --device_type cpu --act_method 2 --w_bit 8 --a_bit 16 --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    deepseekv2_w8a8_dynamic)
        disable_names=$(get_disable_name 60)
        param="--disable_names $disable_names --part_file_size 5 --device_type cpu --act_method 2 --w_bit 8 --a_bit 8  --is_dynamic True --tokenizer_args {\"padding_side\":\"left\",\"pad_token\":\"<unk>\"}"
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst --calib_file "" $param