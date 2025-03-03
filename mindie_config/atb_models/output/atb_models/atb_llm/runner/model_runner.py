# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import Dict, Iterable, List, Optional, Union
import os
import json

import torch
from torch import nn
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from ..models import get_model
from ..utils import file_utils
from ..utils.dist import initialize_distributed
from ..utils.weights import Weights
from ..utils.env import ENV
from ..utils.cpu_binding import bind_cpus
from ..utils.initial import NPUSocInfo, load_atb_speed
from ..utils.log import logger, print_log
from ..utils.log.error_code import ErrorCode
from ..utils.adapter_manager import AdapterManager


class ModelRunner:
    """
    Class for running model.

    Class attributes:
        model (nn.Module, optional): Model instance, defaults to None.
        soc_info (NPUSocInfo, optional): SOC info instance, defaults to None.
        head_size (int, optional): Head size of multi-head attention, defaults to None.
        num_heads (int, optional): Number of head of multi-head attention, defaults to None.
        num_kv_heads (int, optional): Number of key-value heads, defaults to None.
        num_layers (int, optional): Number of layers, defaults to None.
        device (torch.device, optional): Device to run the model, defaults to None.
        dtype (torch.dtype, optional): Dtype of data, defaults to None.
        k_head_size (int, optional): Head size of key head in multi-head attention, defaults to None.
        v_head_size (int, optional): Head size of value head in multi-head attention, defaults to None.
    
    Args:
        model_name_or_path (str): Model name or path.
        rank (int): Rank of current process.
        world_size (int): World size of multi process.
        npu_id (int, optional): NPU id of current process, defaults to None.
        local_rank (int, optional): Local rank of current process, defaults to None.
        is_flash_causal_lm (bool, optional): Whether to use flash causal lm, defaults to True.
        load_tokenizer (bool, optional): Whether to load tokenizer, defaults to True.
        max_position_embeddings (int, optional): Max positionembeddings, defaults to None.
        tokenizer_path (str, optional): Tokenizer path, defaults to None.
        **kwargs (dict, optional): Additional keyword arguments.
    """
    model: Optional[nn.Module] = None
    soc_info: Optional[NPUSocInfo] = None
    head_size: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    num_layers: Optional[int] = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    k_head_size: Optional[int] = None
    v_head_size: Optional[int] = None

    def __init__(self,
                 model_name_or_path: str,
                 rank: int,
                 world_size: int,
                 npu_id: Optional[int] = None,
                 local_rank: Optional[int] = None,
                 is_flash_causal_lm: bool = True,
                 load_tokenizer: bool = True,
                 max_position_embeddings: Optional[int] = None,
                 tokenizer_path: Optional[str] = None,
                 **kwargs):
        self.model_name_or_path = model_name_or_path
        self.rank = rank
        self.local_rank = local_rank if local_rank is not None else rank
        self.npu_id = npu_id if npu_id is not None else self.local_rank
        self.world_size = world_size
        self.inference_mode = kwargs.get("inference_mode", "")
        self.enable_atb_torch = kwargs.get("enable_atb_torch", False)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        load_atb_speed()
        
        if ENV.bind_cpu:
            try:
                bind_cpus(world_size, self.npu_id, ratio=1.0)
            except RuntimeError as e:
                print_log(rank, logger.info, e)
            except ValueError as e:
                print_log(rank, logger.info, e)
            except Exception as _:
                print_log(rank, logger.info, 'Skip binding cpu.')
        router_ins = get_model(model_name_or_path, is_flash_causal_lm=is_flash_causal_lm, load_tokenizer=load_tokenizer,
                               max_position_embeddings=max_position_embeddings, revision=None,
                               tokenizer_path=tokenizer_path, trust_remote_code=self.trust_remote_code,
                               enable_atb_torch=self.enable_atb_torch)
        self.model_cls = router_ins.model_cls
        self.config = router_ins.config
        self.tokenizer = router_ins.tokenizer
        self.input_builder = router_ins.input_builder
        self.postprocessor = router_ins.postprocessor
        self.config_dict = router_ins.config_dict
        self.enable_atb_torch = router_ins.enable_atb_torch
        self.dtype = self.config.torch_dtype
        self.quantize = self.config.quantize
        self.kv_quant_type = self.config.quantization_config.kv_quant_type
        self.fa_quant_type = self.config.quantization_config.fa_quant_type
        self.kv_cache_dtype = torch.int8 if self.kv_quant_type is not None or \
            self.fa_quant_type is not None else self.dtype
        
        if self.dtype not in [torch.float16, torch.bfloat16]:
            error_msg = "`torch_dtype` is only supported for type `float16` and" \
                " `bfloat16`, loaded from config.json -> torch_dtype. " \
                "The specific types supported by each model are different, please refer to the model README file."
            logger.error(error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise NotImplementedError(error_msg)

        print_log(rank, logger.info, f'model_runner.quantize: {self.quantize}, '
                                     f'model_runner.kv_quant_type: {self.kv_quant_type}, '
                                     f'model_runner.fa_quant_type: {self.fa_quant_type}, '
                                     f'model_runner.dtype: {self.dtype}', need_filter=True)
    
        self.adapter_manager = None
        self.lora_adapter = None
        lora_adapter_json_path = os.path.join(model_name_or_path, "lora_adapter.json")
        if os.path.exists(lora_adapter_json_path):
            lora_adapter_json_path = file_utils.standardize_path(lora_adapter_json_path, check_link=False)
            file_utils.check_file_safety(lora_adapter_json_path)
            with file_utils.safe_open(lora_adapter_json_path, mode="r", encoding="utf-8") as f:
                self.lora_adapter = json.load(f)

        self.process_group, self.device = initialize_distributed(self.rank, self.npu_id, world_size)
        torch.npu.set_compile_mode(jit_compile=False)

        print_log(rank, logger.info, f'init tokenizer done: {self.tokenizer}')

    def load_weights(self, **kwargs):
        """Load weights from file."""
        weights = Weights(
            self.model_name_or_path, self.device, self.dtype,
            process_group=self.process_group,
            quantize=self.quantize,
            extension=".safetensors",
            **kwargs
        )
        if "OMP_NUM_THREADS" not in os.environ and self.world_size > 1:
            os.environ["OMP_NUM_THREADS"] = "1"
        try:
            self.model = self.model_cls(self.config, weights, inference_mode=self.inference_mode,
                                        trust_remote_code=self.trust_remote_code)
        except TypeError as e:
            print_log(self.rank, logger.warning, f'init model: {e}')
            self.model = self.model_cls(self.config, weights)
        if self.lora_adapter is not None:
            self.model.adapter_manager = AdapterManager(weights)
            self.model.update_adapter_manager()
            self.model.adapter_manager.preload_adapter(self.lora_adapter)
            self.adapter_manager = self.model.adapter_manager

        self.model.to(weights.device)
        weights.release_file_handler()

        if self.lora_adapter is not None:
            if self.model.adapter_manager.lora_weights_loader is not None:
                self.model.adapter_manager.lora_weights_loader.release_file_handler()
            self.model.adapter_manager.prepare_adapter_weights()
        if self.enable_atb_torch:
            self.model.init_graph()

        self.soc_info = self.model.soc_info
        self.head_size = self.model.head_size
        self.num_heads = self.model.num_attention_heads
        self.num_kv_heads = self.model.num_key_value_heads
        self.num_layers = self.model.num_layers

        # not equal k v length for mla
        if hasattr(self.model, 'kv_lora_rank') and hasattr(self.model, 'qk_rope_head_dim'):
            self.num_kv_heads = 1
            self.k_head_size = self.model.kv_lora_rank + self.model.qk_rope_head_dim
            self.v_head_size = 0
        else:
            self.k_head_size = self.model.head_size
            self.v_head_size = self.model.head_size

        print_log(self.rank, logger.info, f'model:\n {self.model}')

    def build_inputs(self, conversations: List[List[Dict[str, str]]], **kwargs) -> list:
        """Build inputs for model."""
        return [self.input_builder.make_context(self.rank, conversation, **kwargs) for conversation in conversations]

    def forward(self, **kwargs) -> Union[CausalLMOutputWithPast, tuple]:
        """Call model's forward pass."""
        return self.model.forward(**kwargs)

    def generate(self, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        """Generate output text via calling model's generate method."""
        return self.model.generate(**kwargs)

    def generate_position_ids(self, input_ids: np.ndarray) -> Iterable:
        """Generate position ids."""
        return self.input_builder.generate_position_ids(input_ids)

    def save_pretrained(self, **kwargs):
        """Save pretrained model."""
        save_directory_key = 'save_directory'
        if save_directory_key not in kwargs:
            raise ValueError(f'{save_directory_key} is required')
        kwargs[save_directory_key] = os.path.join(kwargs[save_directory_key], f'part{self.rank}-of-{self.world_size}')
        self.model.save_pretrained(**kwargs)
