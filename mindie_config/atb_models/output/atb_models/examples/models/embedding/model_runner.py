# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

import torch
import torch_npu
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from transformers.tokenization_utils_base import BatchEncoding

from atb_llm.models.base.model_utils import safe_from_pretrained
from atb_llm.utils import file_utils


if TYPE_CHECKING:
    from optimum.onnxruntime import ORTModel
    from ais_bench.infer.interface import InferSession


TRUE = "true"
FALSE = "false"


def check_import(exception: Optional[Exception]) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if exception:
                raise RuntimeError("unsupported model type") from exception
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_auto_model_cls(architectures: Union[str, List[str]]) -> str:
    architecture = architectures[0] if isinstance(architectures, list) else architectures
    match = re.search(r"([A-Z][a-z]+)$", architecture)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"unsupported architecture: {architecture}, please check config.json.")


def modify_config(config: PretrainedConfig, model_name_or_path: Union[str, os.PathLike]) -> None:
    auto_map = {
        "Model": "AutoModel",
        "Classification": "AutoModelForSequenceClassification"
    }
    architecture_type = config.model_type.replace("-", "_")
    model_name_or_path = file_utils.standardize_path(model_name_or_path, check_link=False)
    file_utils.check_path_permission(model_name_or_path)
    if f"modeling_{architecture_type}.py" not in os.listdir(model_name_or_path):
        modeling_file_path = os.path.join(os.path.dirname(__file__), architecture_type)
    else:
        modeling_file_path = model_name_or_path
    config_file = os.path.join(model_name_or_path, "config.json")
    with file_utils.safe_open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["_name_or_path"] = model_name_or_path
    config_dict["auto_map"] = {
        auto_map.get(get_auto_model_cls(arch)): f"{modeling_file_path}--modeling_{architecture_type}.{arch}"
        for arch in config.architectures
    }
    with file_utils.safe_open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


class FloatModel:

    try:
        from transformers import AutoModel, AutoModelForSequenceClassification
    except ImportError as e:
        exception = e
    else:
        exception = None
        architectures_map = {
            "Model": {
                "cls": AutoModel,
                "output": BaseModelOutputWithPoolingAndCrossAttentions
            },
            "Classification": {
                "cls": AutoModelForSequenceClassification,
                "output": SequenceClassifierOutput
            }
        }

    @classmethod
    @check_import(exception)
    def get_model_ins(
            cls,
            model_name_or_path: Union[str, os.PathLike],
            trust_remote_code: bool,
            torch_dtype: torch.dtype,
            device: torch.device,
            architecture: str
    ) -> PreTrainedModel:
        return safe_from_pretrained(
            cls.architectures_map.get(architecture).get("cls"),
            model_name_or_path,
            trust_remote_code=trust_remote_code
        ).to(torch_dtype).to(device).eval()

    @classmethod
    @check_import(exception)
    @torch.no_grad()
    def forward(
            cls,
            inputs: BatchEncoding,
            model: PreTrainedModel,
            device: torch.device,
            architecture: str
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        return model(**inputs, return_dict=True)


class ONNXModel:

    try:
        from optimum.onnxruntime import ORTModelForCustomTasks, ORTModelForSequenceClassification
    except ImportError as e:
        exception = e
    else:
        exception = None
        architectures_map = {
            "Model": {
                "cls": ORTModelForCustomTasks,
                "output": BaseModelOutputWithPoolingAndCrossAttentions
            },
            "Classification": {
                "cls": ORTModelForSequenceClassification,
                "output": SequenceClassifierOutput
            }
        }

    @classmethod
    @check_import(exception)
    def get_model_ins(
            cls,
            model_name_or_path: Union[str, os.PathLike],
            trust_remote_code: bool,
            torch_dtype: torch.dtype,
            device: torch.device,
            architecture: str
    ) -> "ORTModel":
        return safe_from_pretrained(
            cls.architectures_map.get(architecture).get("cls").from_pretrained,
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        ).to(device)

    @classmethod
    @check_import(exception)
    @torch.inference_mode()
    def forward(
            cls,
            inputs: BatchEncoding,
            model: "ORTModel",
            device: torch.device,
            architecture: str
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        return model(**inputs)


class OMModel:

    try:
        from ais_bench.infer.interface import InferSession
    except ImportError as e:
        exception = e
    else:
        exception = None
        architectures_map = {
            "Model": {
                "cls": InferSession,
                "output": BaseModelOutputWithPoolingAndCrossAttentions
            },
            "Classification": {
                "cls": InferSession,
                "output": SequenceClassifierOutput
            }
        }

    @classmethod
    @check_import(exception)
    def get_model_ins(
            cls,
            model_name_or_path: Union[str, os.PathLike],
            trust_remote_code: bool,
            torch_dtype: torch.dtype,
            device: torch.device,
            architecture: str
    ) -> "InferSession":
        om_model_file = next(iter([file for file in os.listdir(model_name_or_path) if file.endswith(".om")]), "")
        return cls.architectures_map.get(architecture).get("cls")(
            device.index,
            os.path.join(model_name_or_path, om_model_file)
        )

    @classmethod
    @check_import(exception)
    def forward(
            cls,
            inputs: BatchEncoding,
            model: "InferSession",
            device: torch.device,
            architecture: str
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        feeds = [input_ids, attention_mask]

        if "token_type_ids" in inputs:
            feeds.append(inputs.get("token_type_ids"))
        if "position_ids" in inputs:
            feeds.append(inputs.get("position_ids"))

        dtype_size = 4
        hidden_size = 1024
        size = dtype_size * hidden_size * input_ids.numel()

        model_output = model.infer(feeds=feeds, mode="dymshape", custom_sizes=size)
        model_output = torch.from_numpy(model_output[0])

        return cls.architectures_map.get(architecture).get("output")(model_output)


class ModelFactory:

    model_factory_map = {
        "float": FloatModel,
        "onnx": ONNXModel,
        "om": OMModel
    }

    @classmethod
    def get_model_ins(
            cls,
            model_name_or_path: Union[str, os.PathLike],
            trust_remote_code: bool,
            torch_dtype: torch.dtype,
            device: torch.device,
            architecture: str,
            model_type: str,
    ) -> Any:
        if model_type in cls.model_factory_map:
            return cls.model_factory_map.get(model_type).get_model_ins(
                model_name_or_path,
                trust_remote_code,
                torch_dtype,
                device,
                architecture
            )
        raise ValueError(f"unsupported model type: {model_type}.")

    @classmethod
    def forward(
            cls,
            inputs: BatchEncoding,
            model: Any,
            device: torch.device,
            architecture: str,
            model_type: str
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        if model_type in cls.model_factory_map:
            return cls.model_factory_map.get(model_type).forward(
                inputs,
                model,
                device,
                architecture
            )
        raise ValueError(f"unsupported model type: {model_type}.")


@dataclass
class ModelCls:
    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer
    model: Union[PreTrainedModel, "ORTModel", "InferSession"]


class ModelRunner:
    def __init__(
            self,
            model_name_or_path: Union[str, os.PathLike],
            trust_remote_code: bool,
            torch_dtype: torch.dtype,
            device: torch.device,
            model_type: str,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self.device = device
        self.model_type = model_type

        self.init_device()

        self.model_cls = self.get_model_cls()
        self.config = self.model_cls.config
        self.tokenizer = self.model_cls.tokenizer
        self.model = self.model_cls.model

        self.architecture = get_auto_model_cls(self.config.architectures)

    @staticmethod
    def generate_inputs(
            model_input_names: List[str],
            vocab_size: int,
            batch_size: int,
            seq_len: int,
            device: torch.device
    ) -> Dict:
        input_shape = (batch_size, seq_len)
        inputs = {}
        if "input_ids" in model_input_names:
            inputs["input_ids"] = torch.randint(0, vocab_size, input_shape, dtype=torch.int64, device=device)
        if "token_type_ids" in model_input_names:
            inputs["token_type_ids"] = torch.zeros(input_shape, dtype=torch.int64, device=device)
        if "attention_mask" in model_input_names:
            inputs["attention_mask"] = torch.randint(0, 2, input_shape, dtype=torch.int64, device=device)
        if "position_ids" in model_input_names:
            inputs["position_ids"] = torch.arange(0, seq_len, dtype=torch.int64, device=device)
        return inputs

    def init_device(self):
        if self.model_type == "om":
            self.device = torch.device("cpu:0")
            return
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)
        elif self.device.type == "npu":
            torch_npu.npu.set_device(self.device.index)
            torch_npu.npu.set_compile_mode(jit_compile=False)

    def get_model_cls(self) -> ModelCls:
        config = safe_from_pretrained(
            AutoConfig,
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code
        )
        modify_config(config, self.model_name_or_path)
        tokenizer = safe_from_pretrained(
            AutoTokenizer,
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code
        )
        model = ModelFactory.get_model_ins(
            self.model_name_or_path,
            self.trust_remote_code,
            self.torch_dtype,
            self.device,
            get_auto_model_cls(config.architectures),
            self.model_type
        )

        return ModelCls(config, tokenizer, model)

    def tokenize(
            self,
            sentences: Union[str, List[str], List[List[str]]],
            padding: Union[str, bool] = True,
            truncation: Union[str, bool] = True,
            return_tensors: str = "pt",
            max_length: Union[str, int] = 512,
    ) -> BatchEncoding:
        if isinstance(sentences, str):
            sentences = [sentences]

        return self.tokenizer(
            sentences,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            max_length=max_length,
        ).to(self.device)

    def forward(
            self,
            inputs: BatchEncoding
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput]:
        return ModelFactory.forward(
            inputs,
            self.model,
            self.device,
            self.architecture,
            self.model_type
        )

    def embed(self, inputs: BatchEncoding) -> torch.Tensor:
        results = ModelFactory.forward(
            inputs,
            self.model,
            self.device,
            self.architecture,
            self.model_type
        )[0][:, 0]
        results = torch.nn.functional.normalize(results, p=2, dim=1).cpu()
        return results

    def rerank(self, inputs: BatchEncoding) -> torch.Tensor:
        results = ModelFactory.forward(
            inputs,
            self.model,
            self.device,
            self.architecture,
            self.model_type
        ).logits.view(-1, ).float()
        results = results.cpu()
        return results


class Arguments:

    parser = argparse.ArgumentParser()

    def __init__(self):
        self.set_common_args()

    @classmethod
    def set_runner_args(cls):
        parser = cls().parser
        parser.add_argument(
            "request",
            type=str,
            choices=["embed", "rerank"]
        )
        parser.add_argument(
            "--texts",
            type=str,
            nargs='+',
            default=["样例数据-1", "样例数据-2"]
        )
        parser.add_argument(
            "--max_batch_size",
            type=int,
            default=1
        )
        return parser.parse_args()

    @classmethod
    def set_tester_args(cls):
        parser = cls().parser
        parser.add_argument(
            "task",
            type=str,
            choices=["performance", "retrieval", "reranking"]
        )
        parser.add_argument(
            "--dataset_path",
            help="dataset path"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=20
        )
        parser.add_argument(
            "--loop",
            type=int,
            default=100
        )
        parser.add_argument(
            "--outputs",
            type=str,
            default="outputs/results",
        )
        return parser.parse_args()

    def set_common_args(self):
        self.parser.add_argument(
            "--model_name_or_path",
            help="model and tokenizer path"
        )
        self.parser.add_argument(
            "--trust_remote_code",
            action="store_true"
        )
        self.parser.add_argument(
            "--model_type",
            type=str,
            choices=["float", "onnx", "om"],
            default="float"
        )
        self.parser.add_argument(
            "--torch_dtype",
            type=str,
            default="float16"
        )
        self.parser.add_argument(
            "--device_type",
            type=str,
            choices=["cpu", "cuda", "npu"],
            default="cpu"
        )
        self.parser.add_argument(
            "--device_id",
            type=int,
            default=0
        )
        self.parser.add_argument(
            "--padding",
            type=lambda value: value.lower() == TRUE if value.lower() in [TRUE, FALSE] else value,
            default="max_length"
        )
        self.parser.add_argument(
            "--truncation",
            type=lambda value: value.lower() == TRUE if value.lower() in [TRUE, FALSE] else value,
            default=True
        )
        self.parser.add_argument(
            "--return_tensors",
            type=str,
            choices=["pt", "np"],
            default="pt"
        )
        self.parser.add_argument(
            "--max_seq_len",
            type=int,
        )
