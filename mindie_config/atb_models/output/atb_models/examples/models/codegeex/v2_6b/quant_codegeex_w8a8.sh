#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
}

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
        -calib_file)
            if [[ -n "$2" ]]; then
                calib_file="$2"
                shift
            else
                echo "Error: -calib_file requires a non-empty argument."
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

if [[ -z "$calib_file" ]]; then
    echo "Error: Missing required option: -calib_file"
    show_help
    exit 1
fi

# 设置环境变量
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# 进入运行路径
cd ${ATB_SPEED_HOME_PATH}

disable_names="transformer.encoder.layers.0.self_attention.query_key_value transformer.encoder.layers.0.mlp.dense_4h_to_h transformer.encoder.layers.1.self_attention.query_key_value transformer.encoder.layers.1.mlp.dense_h_to_4h transformer.encoder.layers.1.mlp.dense_4h_to_h transformer.encoder.layers.2.self_attention.query_key_value transformer.encoder.layers.2.mlp.dense_h_to_4h transformer.encoder.layers.2.mlp.dense_4h_to_h transformer.encoder.layers.3.self_attention.query_key_value transformer.encoder.layers.4.self_attention.query_key_value transformer.encoder.layers.5.self_attention.query_key_value transformer.encoder.layers.6.self_attention.query_key_value transformer.encoder.layers.7.self_attention.query_key_value transformer.encoder.layers.8.self_attention.query_key_value transformer.encoder.layers.9.self_attention.query_key_value transformer.encoder.layers.11.self_attention.query_key_value transformer.encoder.layers.17.mlp.dense_4h_to_h transformer.encoder.layers.23.mlp.dense_4h_to_h transformer.encoder.layers.27.mlp.dense_4h_to_h transformer.output_layer"

param="--disable_names $disable_names --w_bit 8 --a_bit 8 --device_type cpu --act_method 3 --w_sym True --mm_tensor False"

if [ "$trust_remote_code" == "True" ]; then
    param="$param --trust_remote_code"
fi

python -m examples.convert.model_slim.quantifier --model_path $src --save_directory $dst $param --calib_file ${calib_file}