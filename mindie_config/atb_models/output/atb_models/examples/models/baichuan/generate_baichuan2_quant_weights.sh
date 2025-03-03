#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model type and quant type. Support quant type: baichuan2_7b_w8a8 | baichuan2_7b_w8a8_kvcache | baichuan2_13b_w8a8 | baichuan2_13b_w4a16 | baichuan2_13b_w8a16"
    echo "-trust_remote_code  Whether to trust local executable files. Default value is false."
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
    local num_layer=$1
    local disable_names=""
    for ((i=0; i<$num_layer; i++)); do
        disable_names="$disable_names model.layers.$i.mlp.down_proj"
    done
    echo "$disable_names"
}

case "$type" in
    baichuan2_7b_w8a8)
        disable_names=$(get_down_proj_disable_name 32)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/models/baichuan/quant_baichuan2_7b_w8a8_data.jsonl \
               --disable_names $disable_names --device_type cpu --w_bit 8 --a_bit 8 \
               --disable_level L0 --anti_method m2 --act_method 3 \
               --tokenizer_args {\"padding_side\":\"left\"} \
               --mm_tensor False \
               --disable_last_linear False"
        ;;
    baichuan2_7b_w8a8_kvcache)
        disable_names=$(get_down_proj_disable_name 32)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/models/baichuan/quant_baichuan2_7b_w8a8_kvcache_data.jsonl \
               --disable_names $disable_names --device_type cpu --w_bit 8 --a_bit 8 \
               --disable_level L0 --anti_method m2 --act_method 3 \
               --tokenizer_args {\"padding_side\":\"left\"} \
               --mm_tensor False \
               --disable_last_linear False"
        ;;
    baichuan2_13b_w8a8)
        disable_names=$(get_down_proj_disable_name 40)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/models/baichuan/quant_baichuan2_13b_w8a8_data.jsonl \
               --disable_names $disable_names --device_type cpu --w_bit 8 --a_bit 8 \
               --disable_level L0 --anti_method m2 --act_method 3 \
               --mm_tensor False \
               --disable_last_linear False"
        ;;
    baichuan2_13b_w4a16)
        disable_names=$(get_down_proj_disable_name 40)
        param="--calib_file $ATB_SPEED_HOME_PATH/examples/models/baichuan/quant_baichuan2_13b_w4a16_data.jsonl \
               --disable_names $disable_names --device_type cpu --w_bit 4 --a_bit 16 \
               --disable_level L0 --anti_method m3 --act_method 1 \
               --disable_last_linear False \
               --open_outlier False \
               --is_lowbit True \
               --mm_tensor False \
               --open_outlier False"
        ;;
    baichuan2_13b_w8a16)
        disable_names=$(get_down_proj_disable_name 40)
        param="--disable_names $disable_names --device_type cpu --w_bit 8 --a_bit 16 \
               --disable_level L0 --act_method 3 \
               --disable_last_linear False"
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

if [ "$trust_remote_code" == "True" ]; then
    param="$param --trust_remote_code"
    echo "INFO: current trust_remote_code: True"
fi

python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param