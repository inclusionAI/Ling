#!/bin/bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

num_layers=40
disable_names=()
qkv_names=()
up_names=()

for layer in $(seq 0$((num_layers-1))); do
    disable_names+=("transformer.encoder.layers.${layer}.mlp.dense_4h_to_h")
    qkv_names+=("transformer.encoder.layers.${layer}.self_attention.query_key_value")
    up_names+=("transformer.encoder.layers.${layer}.mlp.dense_h_to_4h")
done
disable_names+=("${qkv_names[@]}")
disable_names+=("${up_names[@]}")

weight_path=$1
shift
w8a8s_weight_path=$1
shift
w8a8sc_weight_path=${w8a8s_weight_path}/compress
calib_data=$1
shift
tp_size=$1
shift
device_0=$1
shift
device_1=$1
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

ASCEND_RT_VISIBLE_DEVICES=${device_0} python -m examples.convert.model_slim.quantifier --model_path "${weight_path}" --save_directory "${w8a8s_weight_path}" --calib_file ${calib_data} --disable_names ${disable_names} --device_type npu --is_lowbit True --w_bit 4 --a_bit 8 $extra_param

ASCEND_RT_VISIBLE_DEVICES=${device_1} torchrun --nproc_per_node "$tp_size" -m examples.convert.model_slim.sparse_compressor --model_path "${w8a8s_weight_path}" --save_directory "${w8a8sc_weight_path}" $extra_param
