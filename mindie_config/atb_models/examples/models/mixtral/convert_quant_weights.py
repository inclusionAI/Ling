# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.


import torch
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

from atb_llm.models.mixtral.config_mixtral import MixtralConfig
from atb_llm.utils.env import ENV
from atb_llm.models.base import model_utils
from examples.convert.model_slim.quantifier import parse_arguments


def get_calib_dataset(tokenizer_local, calib_list, device):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer_local(calib_data, return_tensors='pt')
        calib_dataset.append([
            inputs.data['input_ids'].to(device),
            inputs.data['attention_mask'].to(device)
        ])
    return calib_dataset


if __name__ == "__main__":
    args = parse_arguments()
    rank = ENV.rank
    config = MixtralConfig.from_pretrained(args.model_path)
    disable_names = []
    mixtral_layers = config.num_hidden_layers
    disable_idx_lst = list(range(mixtral_layers))
    for layer_index in disable_idx_lst:
        gate_name = f"model.layers.{layer_index}.block_sparse_moe.gate"
        disable_names.append(gate_name)
    disable_names.append('lm_head')

    tokenizer = model_utils.safe_get_tokenizer_from_pretrained(args.model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model_utils.safe_get_model_from_pretrained(
        model_path=args.model_path,
        device_map="auto",
        trust_remote_code=True if args.trust_remote_code else False,
        torch_dtype=torch.bfloat16
    ).eval()

    calib_set = [
        "Where is the capital of Cbhina?",
        "Please make a poem:",
        "I want to learn python, how should I learn it?",
        "Please help me write a job report on large model inference optimization:",
        "What are the most worth visiting scenic spots in China?"
    ]
    dataset_calib = get_calib_dataset(tokenizer, calib_set, model.device)

    quant_config = QuantConfig(
        a_bit=8,
        w_bit=8,
        disable_names=disable_names,
        dev_type='npu',
        pr=1.0,
        w_sym=True,
        mm_tensor=False,
        is_dynamic=True
    )
    calibrator = Calibrator(model, quant_config, calib_data=[], disable_level='L0')
    calibrator.run()

    calibrator.save(args.save_directory, save_type=["safe_tensor", "numpy"])