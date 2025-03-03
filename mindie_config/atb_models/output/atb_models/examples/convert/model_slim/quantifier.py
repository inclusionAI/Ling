# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import json

import torch
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

from atb_llm.utils.file_utils import MAX_PATH_LENGTH
from atb_llm.utils.env import ENV
from atb_llm.utils.argument_utils import ArgumentParser, StringArgumentValidator, MAX_KEY_LENGTH, MAX_JSON_LENGTH
from atb_llm.models.base import model_utils
from examples.convert.convert_utils import copy_tokenizer_files, modify_config
from examples.convert.model_slim.get_calibration_dataset import load_jsonl


CPU = "cpu"
NPU = "npu"


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be `True` or `False`")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help="model and tokenizer path",
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--save_directory', type=str,
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_PATH_LENGTH))
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument(
        '--calib_texts',
        type=str,
        nargs='+',
        default=None)
    parser.add_argument(
        '--calib_file',
        type=str,
        help='A jsonl file contains calibration data.',
        default=f"{os.path.join(os.path.dirname(__file__), 'teacher_qualification.jsonl')}")
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--disable_names', type=str, nargs='+', default=None)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=CPU)
    parser.add_argument('--fraction', type=float, default=0.01)
    parser.add_argument("--act_method", type=int, choices=[1, 2, 3], default=1,
                        help=" `1`: `MinMax`, `2`: `Histogram`, `3`: `Auto`")
    parser.add_argument('--co_sparse', type=cmd_bool, default=False)
    parser.add_argument('--anti_method', type=str, default='')
    parser.add_argument('--disable_level', type=str, default='L0')
    parser.add_argument('--do_smooth', type=cmd_bool, default=False)
    parser.add_argument('--use_sigma', type=cmd_bool, default=False)
    parser.add_argument('--use_reduce_quant', type=cmd_bool, default=False)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--sigma_factor', type=float, default=3.0)
    parser.add_argument('--is_lowbit', type=cmd_bool, default=False)
    parser.add_argument('--mm_tensor', type=cmd_bool, default=True)
    parser.add_argument('--w_sym', type=cmd_bool, default=True)
    parser.add_argument('--use_kvcache_quant', type=cmd_bool, default=False)
    parser.add_argument('--use_fa_quant', type=cmd_bool, default=False)
    parser.add_argument('--fa_amp', type=int, default=0)
    parser.add_argument('--open_outlier', type=cmd_bool, default=True)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--is_dynamic', type=cmd_bool, default=False)
    parser.add_argument('--input_ids_name', type=str, default='input_ids',
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_KEY_LENGTH))
    parser.add_argument('--attention_mask_name', type=str, default='attention_mask',
                        validator=StringArgumentValidator(min_length=1, max_length=MAX_KEY_LENGTH))
    parser.add_argument('--tokenizer_args', type=str, default='{}',
                        validator=StringArgumentValidator(min_length=2, max_length=MAX_JSON_LENGTH))
    parser.add_argument('--disable_last_linear', type=cmd_bool, default=True)
    parser.add_argument('--trust_remote_code', action='store_true')
    return parser.parse_args()


class Quantifier:
    def __init__(self, model_path_or_name, quant_config=None,
                 anti_outlier_config=None, device_type='cpu', **kwargs):
        self.device_type = device_type

        self.quant_config = quant_config
        self.anti_outlier_config = anti_outlier_config
        self.model_path_or_name = model_path_or_name

        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.config = self.get_config(**kwargs)
        self.model = self.get_model(**kwargs)
        self.tokenizer = self.get_tokenizer(**kwargs)

    def get_config(self, **kwargs):
        return model_utils.safe_get_config_from_pretrained(
            self.model_path_or_name, trust_remote_code=self.trust_remote_code)

    def get_model(self, **kwargs):
        device_map = CPU if self.device_type == CPU else "auto"
        dtype = self.config.torch_dtype if self.device_type == NPU else torch.float32
        return model_utils.safe_get_model_from_pretrained(
            self.model_path_or_name,
            low_cpu_mem_usage=True, torch_dtype=dtype,
            device_map=device_map,
            use_safetensors=True, trust_remote_code=self.trust_remote_code
        )

    def get_tokenizer(self, **kwargs):
        tokenizer_args = kwargs.get("tokenizer_args", {})
        return model_utils.safe_get_tokenizer_from_pretrained(
            self.model_path_or_name, use_fast=False,
            trust_remote_code=self.trust_remote_code,
            legacy=False, **tokenizer_args
        )

    def get_tokenized_data(self, input_texts,
                           input_ids_name='input_ids',
                           attention_mask_name='attention_mask'):
        tokenized_data = []
        for input_text in input_texts:
            inputs = self.tokenizer(input_text, return_tensors='pt', padding=True).to(self.device_type)
            tokenized_data.append(
                [inputs.data[input_ids_name], inputs.data[attention_mask_name]])
        return tokenized_data

    def convert(self, tokenized_data, save_path, disable_level, part_file_size=None):
        if self.device_type == NPU:
            # 避免在线编译算子，使用二进制编译的算子
            torch.npu.set_compile_mode(jit_compile=False)

        if self.anti_outlier_config is not None:
            anti_outlier = AntiOutlier(self.model, calib_data=tokenized_data, cfg=self.anti_outlier_config)
            anti_outlier.process()

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        calibrator = Calibrator(self.model, self.quant_config, calib_data=tokenized_data, disable_level=disable_level)
        calibrator.run()
        calibrator.save(save_path, save_type=["safe_tensor"], part_file_size=part_file_size)


class QuantPipeline:
    def __init__(self, context):
        self.context = context
        self.rank = ENV.rank
        self.quantifier = self.get_quantifier()

    def get_quant_conf(self, **kwargs):
        quant_conf_param = {
            "w_bit": self.context.w_bit,
            "a_bit": self.context.a_bit,
            "disable_names": self.context.disable_names,
            "dev_type": self.context.device_type,
            "dev_id": self.rank,
            "act_method": self.context.act_method,
            "w_sym": self.context.w_sym,
            "mm_tensor": False,
            "co_sparse": self.context.co_sparse,
            "fraction": self.context.fraction,
            "sigma_factor": self.context.sigma_factor,
            "use_sigma": self.context.use_sigma,
            "is_lowbit": self.context.is_lowbit,
            "do_smooth": self.context.do_smooth,
            "open_outlier": self.context.open_outlier,
            "group_size": self.context.group_size,
            "use_kvcache_quant": self.context.use_kvcache_quant,
            "is_dynamic": self.context.is_dynamic,
            "disable_last_linear": self.context.disable_last_linear,
        }
        quant_conf_param.update(kwargs)

        quant_conf = QuantConfig(**quant_conf_param)
        if self.context.use_fa_quant:
            quant_conf = quant_conf.fa_quant(fa_amp=self.context.fa_amp)

        return quant_conf

    def get_anti_outlier_conf(self, **kwargs):
        anti_outlier_conf = None
        if self.context.anti_method == 'm3':
            anti_outlier_conf = AntiOutlierConfig(a_bit=self.context.a_bit, w_bit=self.context.w_bit, 
                anti_method=self.context.anti_method, w_sym=self.context.w_sym,
                dev_type=self.context.device_type, dev_id=self.rank)
        elif self.context.anti_method:
            anti_outlier_conf = AntiOutlierConfig(
                anti_method=self.context.anti_method,
                dev_type=self.context.device_type, 
                dev_id=self.rank
                )
        return anti_outlier_conf

    def get_quantifier(self):
        quant_conf = self.get_quant_conf()
        anti_outlier_conf = self.get_anti_outlier_conf()

        quantifier = Quantifier(
            self.context.model_path, quant_conf, anti_outlier_conf,
            device_type=self.context.device_type, tokenizer_args=json.loads(self.context.tokenizer_args),
            trust_remote_code=self.context.trust_remote_code,
        )

        return quantifier

    def get_tokenized_calib_data(self, **kwargs):
        tokenized_calib_data = None
        calib_file = self.context.calib_file
        calib_texts = load_jsonl(calib_file) if calib_file else self.context.calib_texts
        if calib_texts is not None:
            tokenized_calib_data = self.quantifier.get_tokenized_data(
                calib_texts,
                input_ids_name=self.context.input_ids_name,
                attention_mask_name=self.context.attention_mask_name
            )
        return tokenized_calib_data

    def append_quant_conf(self, **kwargs):
        quant_type = f"w{self.context.w_bit}a{self.context.a_bit}"
        is_sparse_compress = self.context.w_bit == 4 and \
            self.context.a_bit == 8 and (self.context.co_sparse or self.context.is_lowbit)
        if is_sparse_compress:
            quant_type = "w8a8s"
        is_w8a8_dynamic = self.context.w_bit == 8 and self.context.a_bit == 8 and self.context.is_dynamic
        if is_w8a8_dynamic:
            quant_type = "w8a8_dynamic"
        auto_config = model_utils.safe_get_config_from_pretrained(
            self.context.model_path, trust_remote_code=self.context.trust_remote_code)
        modify_config(self.context.model_path, self.context.save_directory, auto_config.torch_dtype,
                    quant_type, self.context)

    def run(self, **kwargs):
        tokenized_calib_data = self.get_tokenized_calib_data()
        self.quantifier.convert(
            tokenized_calib_data, self.context.save_directory,
            self.context.disable_level, part_file_size=self.context.part_file_size)

        self.append_quant_conf()
        copy_tokenizer_files(self.context.model_path, self.context.save_directory)


if __name__ == '__main__':
    args = parse_arguments()
    QuantPipeline(args).run()
