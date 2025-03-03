# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import torch

from atb_llm.utils.quantize.quant_type import QuantType
from atb_llm.utils.quantize.pack_type import LinearType
from atb_llm.utils.data.weight_wrapper import NormWrapper, get_module
from atb_llm.utils.data.moe_weight_wrapper import MoeWeightWrapper
from atb_llm.utils.log import logger
from atb_llm.utils.log.error_code import ErrorCode


class MlaWrapper(NormWrapper):
    def __init__(self,
                 norm_name,
                 wrapper_name,
                 num_attention_heads,
                 num_key_value_heads,
                 qk_nope_head_dim,
                 qk_rope_head_dim,
                 q_lora_rank,
                 kv_lora_rank,
                 v_head_dim):
        super().__init__(norm_name)
        self.wrapper_name = wrapper_name
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.q_head_dim_before = qk_nope_head_dim + qk_rope_head_dim
        self.k_head_dim = kv_lora_rank + qk_rope_head_dim
        self.v_head_dim = v_head_dim


class Deepseekv2WeightWrapper(MoeWeightWrapper):
    def __init__(self,
                 soc_info,
                 tp_rank,
                 mla_wrapper,
                 moe_mlp_wrapper,
                 num_experts):
        super().__init__(soc_info, tp_rank, None, moe_mlp_wrapper, num_experts)
        self.attn_wrapper = mla_wrapper
        self.supported_quantize_type = [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]

    @staticmethod
    def preprocess_kv_weights(kv_b_proj_weight, mla_wrapper):
        kv_b_proj_weight = kv_b_proj_weight.reshape(mla_wrapper.num_key_value_heads,
                                                    mla_wrapper.qk_nope_head_dim + mla_wrapper.v_head_dim,
                                                    mla_wrapper.kv_lora_rank)
        k_b_proj_preprocessed = kv_b_proj_weight[:, :mla_wrapper.qk_nope_head_dim, :].contiguous()
        v_b_proj_preprocessed = kv_b_proj_weight[:, mla_wrapper.qk_nope_head_dim:, :].transpose(1, 2).contiguous()

        return k_b_proj_preprocessed, v_b_proj_preprocessed

    @staticmethod
    def trans_rope_weight(weight, rope_dim):
        weight_1 = weight[..., -rope_dim:: 2, :].contiguous()
        weight_2 = weight[..., -rope_dim + 1:: 2, :].contiguous()
        weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)

        return weight.contiguous()

    @staticmethod
    def view_tenor(weight, mla_wrapper, proj_name, pre_view=True):
        if proj_name == "projq":
            if pre_view:
                return weight.view(mla_wrapper.num_attention_heads, mla_wrapper.q_head_dim_before, -1).contiguous()
            else:
                return weight.view(mla_wrapper.num_attention_heads * mla_wrapper.q_head_dim_before, -1).contiguous()
        elif proj_name == "projk":
            return weight.view((mla_wrapper.k_head_dim), -1).contiguous()
        else:
            msg = f"`proj_name`'s type field must be one of ['projq', 'projk'], " \
                  f"got {proj_name}"
            logger.error(msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise ValueError(msg)

    def preprocess_linear_for_rope(self, linear, mla_wrapper, quantize_type, proj_name):

        weight = linear.weight.data
        weight = self.view_tenor(weight, mla_wrapper, proj_name=proj_name, pre_view=True)
        weight = self.trans_rope_weight(weight, mla_wrapper.qk_rope_head_dim)
        linear.weight.data = self.view_tenor(weight, mla_wrapper, proj_name=proj_name, pre_view=False)

        if weight.dtype not in [torch.float16, torch.bfloat16] and quantize_type in self.supported_quantize_type:
            scale = linear.weight_scale.data
            scale = self.view_tenor(scale, mla_wrapper, proj_name=proj_name, pre_view=True)
            scale = self.trans_rope_weight(scale, mla_wrapper.qk_rope_head_dim)
            linear.weight_scale.data = self.view_tenor(scale, mla_wrapper, proj_name=proj_name, pre_view=False)
            if quantize_type in [QuantType.W8A8_DYNAMIC]:
                linear.weight_scale.data = linear.weight_scale.data.flatten()

            offset = linear.weight_offset.data
            offset = self.view_tenor(offset, mla_wrapper, proj_name=proj_name, pre_view=True)
            offset = self.trans_rope_weight(offset, mla_wrapper.qk_rope_head_dim)
            linear.weight_offset.data = self.view_tenor(offset, mla_wrapper, proj_name=proj_name, pre_view=False)
            if quantize_type in [QuantType.W8A8_DYNAMIC]:
                linear.weight_offset.data = linear.weight_offset.data.flatten()
        
        elif weight.dtype not in [torch.float16, torch.bfloat16] and quantize_type in [QuantType.W8A8]:
            deq_scale = linear.deq_scale.data
            deq_scale = self.view_tenor(deq_scale, mla_wrapper, proj_name=proj_name, pre_view=True)
            deq_scale = self.trans_rope_weight(deq_scale, mla_wrapper.qk_rope_head_dim)
            linear.deq_scale.data = self.view_tenor(deq_scale, mla_wrapper,
                                                    proj_name=proj_name, pre_view=False).flatten()

            quant_bias = linear.quant_bias.data
            quant_bias = self.view_tenor(quant_bias, mla_wrapper, proj_name=proj_name, pre_view=True)
            quant_bias = self.trans_rope_weight(quant_bias, mla_wrapper.qk_rope_head_dim)
            linear.quant_bias.data = self.view_tenor(quant_bias, mla_wrapper,
                                                     proj_name=proj_name, pre_view=False).flatten()

    def register_absorbed_linear_wrapper(self, linear, mla_wrapper, quantize_type):
        k_b_proj, v_b_proj = self.preprocess_kv_weights(linear.weight.data, mla_wrapper)
        self.register_absorbed_linear(linear, k_b_proj, quantize_type)
        self.register_absorbed_linear(linear, v_b_proj, quantize_type)

    # quant for per channel/ no bias
    def register_absorbed_linear(self, linear, weight, quantize_type):
        if linear.weight.dtype in [torch.float16, torch.bfloat16]:
            self.weights.append(weight)
            self.weights.append(self.placeholder)
            self.weights.extend([self.placeholder] * 4)
            self.layer_linear_type.append(LinearType.FP)
            self.layer_linear_transpose_types.append(LinearType.INVALID)
        # keep quant param as original linear
        elif quantize_type in self.supported_quantize_type:
            self.weights.append(weight)
            self.weights.append(self.placeholder)
            self.weights.append(self.placeholder)
            self.weights.append(linear.weight_offset.data)
            self.weights.append(linear.weight_scale.data)
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
            self.layer_linear_transpose_types.append(LinearType.INVALID)
        else:
            weight = self.weight_format_cast(weight)
            self.weights.append(weight)
            self.weights.append(linear.quant_bias.data)
            self.weights.append(linear.deq_scale.data)
            self.weights.append(linear.input_offset.data)
            self.weights.append(linear.input_scale.data)
            if quantize_type == QuantType.W8A8SC:
                self.weights.append(linear.index.data)
            else:
                self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
            # batch matmul no transpose
            self.layer_linear_transpose_types.append(LinearType.INVALID)

    def register_layer_attn(self, layer, wrapper, quantize_type):
        wrapper_module = get_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)

        if wrapper.q_lora_rank is not None:
            self.register_linear_wrapper(wrapper_module.q_a_proj.linear, quantize_type)
            self.register_norm(wrapper_module.q_a_layernorm)
            self.preprocess_linear_for_rope(wrapper_module.q_b_proj.linear, wrapper,
                                            quantize_type, proj_name="projq")
            # deepseekv2 no bias
            self.register_linear_wrapper(wrapper_module.q_b_proj.linear, quantize_type)
        else:
            self.preprocess_linear_for_rope(wrapper_module.q_proj.linear, wrapper,
                                            quantize_type, proj_name="projq")
            # deepseekv2 no bias
            self.register_linear_wrapper(wrapper_module.q_proj.linear, quantize_type)
            # if not qlora，add 8 placeholders.
            self.weights.extend([self.placeholder] * 8)
            self.layer_linear_type.append(LinearType.FP)
            self.layer_linear_transpose_types.append(LinearType.INVALID)

        self.preprocess_linear_for_rope(wrapper_module.kv_a_proj_with_mqa.linear, wrapper,
                                            quantize_type, proj_name="projk")
        self.register_linear_wrapper(wrapper_module.kv_a_proj_with_mqa.linear, quantize_type)
        self.register_norm(wrapper_module.kv_a_layernorm)

        self.register_absorbed_linear_wrapper(wrapper_module.kv_b_proj.linear, wrapper, quantize_type)
        self.register_linear_wrapper(wrapper_module.o_proj.linear, quantize_type)