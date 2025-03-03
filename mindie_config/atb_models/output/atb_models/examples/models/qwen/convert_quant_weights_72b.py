# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig

from atb_llm.utils.env import ENV
from atb_llm.models.qwen2.config_qwen2 import Qwen2Config
from examples.convert.model_slim.get_calibration_dataset import load_jsonl
from examples.convert.model_slim.quantifier import parse_arguments, Quantifier
from examples.convert.convert_utils import copy_tokenizer_files, modify_config


if __name__ == "__main__":
    args = parse_arguments()

    rank = ENV.rank

    config = Qwen2Config.from_pretrained(args.model_path)

    disable_names = []
    if args.a_bit != 16:
        # W8A16没有回退层
        num_layers = config.num_hidden_layers
        disable_names = [f"model.layers.{layer}.mlp.down_proj" for layer in range(num_layers)]
        disable_names.append("lm_head")

    anti_outlier_config = None
    if args.anti_method:
        anti_outlier_config = AntiOutlierConfig(anti_method=args.anti_method)

    quant_config = QuantConfig(
        a_bit=args.a_bit,
        w_bit=args.w_bit,
        disable_names=disable_names,
        act_method=args.act_method,
        mm_tensor=False,
        dev_type=args.device_type,
        dev_id=rank,
        pr=1.0,
        fraction=args.fraction,
        co_sparse=args.co_sparse,
        do_smooth=args.do_smooth,
        use_sigma=args.use_sigma,
        sigma_factor=args.sigma_factor,
        is_lowbit=args.is_lowbit,
        use_kvcache_quant=args.use_kvcache_quant
    )

    # 默认无校准数据集
    calibration_dataset = None
    # 若存在calib_file，则使用calib_file作为校准数据集
    if args.calib_file:
        calibration_dataset = load_jsonl(args.calib_file)
    calibration_dataset = calibration_dataset
    quant_weight_generator = Quantifier(args.model_path, quant_config, anti_outlier_config, args.device_type)
    quant_weight_generator.tokenizer.pad_token_id = 0

    tokenized_data = None
    if calibration_dataset is not None:
        tokenized_data = quant_weight_generator.get_tokenized_data(calibration_dataset)

    quant_weight_generator.convert(tokenized_data, args.save_directory, args.disable_level)
    quant_type = f"w{args.w_bit}a{args.a_bit}"
    # 为适配工具稀疏量化传入w_bit=4,a_bit=8暂时修改quant_type
    if args.w_bit == 4 and args.a_bit == 8 and args.co_sparse:
        quant_type = "w8a8s"
    modify_config(
        args.model_path, args.save_directory, config.torch_dtype,
        quant_type, args
    )
    copy_tokenizer_files(args.model_path, args.save_directory)