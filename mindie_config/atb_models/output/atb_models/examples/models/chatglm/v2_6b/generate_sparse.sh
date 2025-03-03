#!/bin/bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

disable_names="transformer.encoder.layers.0.mlp.dense_4h_to_h transformer.encoder.layers.1.self_attention.query_key_value transformer.encoder.layers.1.self_attention.dense transformer.encoder.layers.1.mlp.dense_h_to_4h transformer.encoder.layers.1.mlp.dense_4h_to_h transformer.encoder.layers.2.self_attention.query_key_value transformer.encoder.layers.2.self_attention.dense transformer.encoder.layers.2.mlp.dense_h_to_4h transformer.encoder.layers.2.mlp.dense_4h_to_h transformer.encoder.layers.3.self_attention.query_key_value transformer.encoder.layers.3.self_attention.dense transformer.encoder.layers.4.self_attention.query_key_value transformer.encoder.layers.4.self_attention.dense transformer.encoder.layers.5.self_attention.query_key_value transformer.encoder.layers.5.self_attention.dense transformer.encoder.layers.6.self_attention.query_key_value transformer.encoder.layers.6.self_attention.dense transformer.encoder.layers.7.self_attention.query_key_value transformer.encoder.layers.7.self_attention.dense transformer.encoder.layers.8.self_attention.query_key_value transformer.encoder.layers.8.self_attention.dense transformer.encoder.layers.9.self_attention.query_key_value transformer.encoder.layers.9.self_attention.dense transformer.encoder.layers.11.self_attention.query_key_value transformer.encoder.layers.11.self_attention.dense transformer.encoder.layers.14.self_attention.query_key_value transformer.encoder.layers.14.self_attention.dense transformer.encoder.layers.19.self_attention.query_key_value transformer.encoder.layers.19.self_attention.dense transformer.encoder.layers.20.mlp.dense_4h_to_h transformer.encoder.layers.27.mlp.dense_4h_to_h transformer.output_layer"

weight_path=$1
shift
w8a8s_weight_path=$1
shift
w8a8sc_weight_path=${w8a8s_weight_path}/compress
calib_data=$1
shift
tp_size=$1
shift

trust_remote_code="False"
# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -trust_remote_code)
            trust_remote_code="True"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$trust_remote_code" == "True" ]; then
    extra_param="--trust_remote_code"
fi

cd "${ATB_SPEED_HOME_PATH}"

python -m examples.convert.model_slim.quantifier --model_path "${weight_path}" --save_directory "${w8a8s_weight_path}" --calib_file ${calib_data} --disable_names ${disable_names} --device_type npu --is_lowbit True --w_bit 4 --a_bit 8 $extra_param

torchrun --nproc_per_node "$tp_size" -m examples.convert.model_slim.sparse_compressor --model_path "${w8a8s_weight_path}" --save_directory "${w8a8sc_weight_path}" $extra_param

cp "$weight_path"/modeling_chatglm.py "$w8a8sc_weight_path"/