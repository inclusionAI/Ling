#!/bin/bash

# 参数提示
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "-h -help  Show help message and exit."
    echo "-src FLOAT_WEIGHT_PATH  Specify source float weight path."
    echo "-dst QUANT_WEIGHT_PATH  Specify target quant weight path."
    echo "-type TYPE  Specify model tpye and quant type. Support llava_w8a16."
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

param=""

get_a16_disable_name() {
    for ((i=0; i<24; i++)); do
    {
        disable_names="$disable_names vision_tower.vision_model.encoder.layers.$i.self_attn.k_proj"
        disable_names="$disable_names vision_tower.vision_model.encoder.layers.$i.self_attn.q_proj"
        disable_names="$disable_names vision_tower.vision_model.encoder.layers.$i.self_attn.v_proj"
        disable_names="$disable_names vision_tower.vision_model.encoder.layers.$i.self_attn.out_proj"
        disable_names="$disable_names vision_tower.vision_model.encoder.layers.$i.mlp.fc1"
        disable_names="$disable_names vision_tower.vision_model.encoder.layers.$i.mlp.fc2"
    }
    done
    disable_names="$disable_names multi_modal_projector.linear_1"
    disable_names="$disable_names multi_modal_projector.linear_2"
    disable_names="$disable_names vision_tower.vision_model.embeddings.patch_embedding"
    echo "$disable_names"
}

case "$type" in
    llava_w8a16)
        disable_names=$(get_a16_disable_name)
        param="--disable_names $disable_names --w_bit 8 --a_bit 16 --act_method 3"
        ;;
    *)
        echo "Unknown type: $type"
        show_help
        exit 1
        ;;
esac

python -m examples.models.llava.quantifier --model_path $src --save_directory $dst $param --calib_file ""
