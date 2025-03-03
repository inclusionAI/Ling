# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import ast
import torch

from ..quantize.pack_type import LinearType, ALL_PACK_LIST
from ..quantize.quant_type import QuantType
from .weight_wrapper import NormWrapper, WeightWrapper

SHARED = "shared"


def get_moe_module(obj, name, delimiter="."):
    names = name.split(delimiter)
    for name in names:
        if name.isdigit():
            obj_idx = ast.literal_eval(name)
            obj = obj[obj_idx]
        else:
            obj = getattr(obj, name)
    return obj


class MoeMlpWrapper(NormWrapper):
    def __init__(self, norm_name, wrapper_name, router_name=None, pack_name=None,
                 sep_names=None, down_name=None, shared_experts=False):
        super().__init__(norm_name)
        self.router_name = router_name
        self.wrapper_name = wrapper_name
        self.pack_name = pack_name if pack_name else None
        self.gate_name = sep_names[0] if sep_names and len(sep_names) == 3 else None
        self.up_name = sep_names[1] if sep_names and len(sep_names) == 3 else None
        self.down_name = down_name
        self.shared_experts = shared_experts
        self.soc_info = None
        self.attn_linear_transpose_types = []
        self.attn_linear_types = []
        self.moe_linear_transpose_types = []


class MoeWeightWrapper(WeightWrapper):
    def __init__(self, soc_info, tp_rank, attn_wrapper, moe_mlp_wrapper, num_experts):
        super().__init__(soc_info, tp_rank, attn_wrapper, None)
        self.moe_mlp_wrapper = moe_mlp_wrapper
        self.moe_mlp_wrapper.soc_info = soc_info
        self.moe_mlp_wrapper.soc_info.matmul_nd_nz = False
        self.num_experts = num_experts
        self.shared_experts = self.moe_mlp_wrapper.shared_experts
        self.attn_linear_types = []
        self.mlp_linear_types = []
        self.moe_linear_types = []
        self.attn_linear_transpose_types = []
        self.mlp_linear_transpose_types = []
        self.moe_linear_transpose_types = []
        self.gmm_quant_nd_nz = False

    def register_linear_list_without_bias(self, linear_list, hidden_dim, is_down_weight=False):
        tensor_stacked = None
        if not is_down_weight and linear_list[0].weight.data.shape[0] == hidden_dim:
            tensor_stacked = torch.stack([linear.weight.data for linear in linear_list], dim=0)
        elif is_down_weight and linear_list[0].weight.data.shape[1] == hidden_dim:
            tensor_stacked = torch.stack([linear.weight.data for linear in linear_list], dim=0)
        else:
            tensor_stacked = torch.stack([linear.weight.data.transpose(0, 1) for linear in linear_list], dim=0)
        if self.gmm_quant_nd_nz:
            self.weights.append(self.weight_format_cast(tensor_stacked, enable_nz=True))
        else:
            self.weights.append(self.weight_format_cast(tensor_stacked))

    def register_linear_list_bias(self, linear_list, hidden_dim, is_down_weight=False):
        self.register_linear_list_without_bias(linear_list, hidden_dim, is_down_weight=is_down_weight)
        if hasattr(linear_list[0], 'bias') and (getattr(linear_list[0], 'bias') is not None):
            bias_stacked = torch.stack([linear.bias.data for linear in linear_list], dim=0)
            if self.gmm_quant_nd_nz:
                self.weights.append(self.weight_format_cast(bias_stacked, enable_nz=True))
            else:
                self.weights.append(self.weight_format_cast(bias_stacked))
        else:
            self.weights.append(self.placeholder)

    def besides_float_and_antiquant(self, linear_list, quantize_type, hidden_dim, is_down_weight):
        self.register_linear_list_without_bias(linear_list, hidden_dim, is_down_weight=is_down_weight)
        quant_bias_list = []
        deq_scale_list = []
        input_offset_list = []
        input_scale_list = []
        for linear in linear_list:
            quant_bias_list.append(super().weight_format_cast(linear.quant_bias.data))
            deq_scale_list.append(super().weight_format_cast(linear.deq_scale.data))
            input_offset_list.append(super().weight_format_cast(linear.input_offset.data))
            input_scale_list.append(super().weight_format_cast(linear.input_scale.data))
        self.weights.append(torch.stack(quant_bias_list, dim=0))
        self.weights.append(torch.stack(deq_scale_list), dim=0)
        self.weights.append(torch.stack(input_offset_list), dim=0)
        self.weights.append(torch.stack(input_scale_list), dim=0)
        del quant_bias_list
        del deq_scale_list
        del input_offset_list
        del input_scale_list

        if quantize_type == QuantType.W8A8SC:
            for linear in linear_list:
                self.weights.append(super().weight_format_cast(linear.index.data))
        else:
            self.weights.append(self.placeholder)
        self.layer_linear_type.append(LinearType.INT)

    def register_linear_list_wrapper(self, linear_list, quantize_type, hidden_dim, is_down_weight=False):
        trans_flag = LinearType.INVALID
        if linear_list[0].weight.dtype in [torch.float16, torch.bfloat16]:
            self.register_linear_list_bias(linear_list, hidden_dim, is_down_weight=is_down_weight)
            self.weights.extend([self.placeholder] * 4)
            self.layer_linear_type.append(LinearType.FP)
        elif quantize_type in [QuantType.W4A16, QuantType.W8A16, QuantType.W8A8_DYNAMIC]:
            self.register_linear_list_bias(linear_list, hidden_dim, is_down_weight=is_down_weight)
            self.weights.append(self.placeholder)
            offset_list, scale_list = [], []
            for linear in linear_list:
                offset_list.append(linear.weight_offset.data)
                if quantize_type == QuantType.W8A8_DYNAMIC:
                    scale_dtype = \
                        linear.weight_scale.dtype if linear.weight_scale.dtype == torch.bfloat16 else torch.float32
                    scale_list.append(linear.weight_scale.data.type(scale_dtype))
                else:
                    scale_list.append(linear.weight_scale.data)
            self.weights.append(torch.stack(offset_list, dim=0))
            self.weights.append(torch.stack(scale_list, dim=0))
            del offset_list, scale_list
            self.weights.append(self.placeholder)
            self.layer_linear_type.append(LinearType.INT)
        else:
            self.besides_float_and_antiquant(linear_list, quantize_type, hidden_dim, is_down_weight=is_down_weight)
        self.layer_linear_transpose_types.append(trans_flag)

    def register_layer_linear_pack(self, layer, wrapper, quantize_type, linear_type='attn'):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if linear_type == "shared_mlp":
            self.register_layer_norm(layer, wrapper, pack_type)
            linear = get_moe_module(wrapper_module, f"shared_experts.{wrapper.pack_name}").linear
            self.register_linear_wrapper(linear, quantize_type)
        elif linear_type == "dense_mlp":
            self.register_layer_norm(layer, wrapper, pack_type)
            linear = get_moe_module(wrapper_module, wrapper.pack_name).linear
            self.register_linear_wrapper(linear, quantize_type)
        else:
            super().register_layer_norm(layer, wrapper, pack_type)
            linear = get_moe_module(wrapper_module, wrapper.pack_name).linear
            super().register_linear_wrapper(linear, quantize_type)
        if linear_type == 'attn':
            self.weights.extend([self.placeholder] * 12)
            self.layer_linear_type.extend([LinearType.INVALID, LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID, LinearType.INVALID])
        elif linear_type == 'mlp':
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        elif linear_type == "shared_mlp" or linear_type == "dense_mlp":
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_linear_sep(self, layer, wrapper, quantize_type, linear_type='attn'):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        super().register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'attn':
            q_linear = get_moe_module(wrapper_module, wrapper.q_name).linear
            k_linear = get_moe_module(wrapper_module, wrapper.k_name).linear
            v_linear = get_moe_module(wrapper_module, wrapper.v_name).linear
            super().register_linear_wrapper(q_linear, quantize_type)
            super().register_linear_wrapper(k_linear, quantize_type)
            super().register_linear_wrapper(v_linear, quantize_type)
        elif linear_type == 'mlp':
            gate_linear = get_moe_module(wrapper_module, wrapper.gate_name).linear
            up_linear = get_moe_module(wrapper_module, wrapper.up_name).linear
            super().register_linear_wrapper(gate_linear, quantize_type)
            super().register_linear_wrapper(up_linear, quantize_type)
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_sep')

    def register_layer_moe_linear_pack(self, layer, wrapper, quantize_type, linear_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if not self.shared_experts:
            self.register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'moe_mlp':
            router = get_moe_module(wrapper_module, wrapper.router_name)
            self.register_linear_wrapper(router, quantize_type)
            linear_list = []
            for i in expert_roster:
                pack_name = f"{wrapper.pack_name}.{i}"
                linear_list.append(get_moe_module(wrapper_module, pack_name).linear)
            self.register_linear_list_wrapper(linear_list, quantize_type, hidden_dim=wrapper_module.hidden_dim)
            self.layer_linear_type.extend([LinearType.INVALID])
            self.layer_linear_transpose_types.extend([LinearType.INVALID])
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_moe_linear_sep(self, layer, wrapper, quantize_type, linear_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        self.register_layer_norm(layer, wrapper, pack_type)
        if linear_type == 'moe_mlp':
            router = get_moe_module(wrapper_module, wrapper.router_name)
            self.register_linear_wrapper(router, quantize_type)
            gate_linear_list = []
            up_linear_list = []
            for i in expert_roster:
                gate_name = f"{wrapper.gate_name}.{i}"
                up_name = f"{wrapper.up_name}.{i}"
                gate_linear_list.append(get_moe_module(wrapper_module, gate_name).linear)
                up_linear_list.append(get_moe_module(wrapper_module, up_name).linear)
            self.register_linear_list_wrapper(gate_linear_list, quantize_type, hidden_dim=wrapper_module.hidden_dim)
            self.register_linear_list_wrapper(up_linear_list, quantize_type, hidden_dim=wrapper_module.hidden_dim)
        else:
            raise AssertionError(f'{linear_type} not yet implemented in register_layer_linear_pack')

    def register_layer_moe_mlp(self, layer, wrapper, quantize_type, linear_type='mlp'):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            self.register_layer_linear_pack(layer, wrapper, quantize_type, linear_type)
        else:
            self.register_layer_linear_sep(layer, wrapper, quantize_type, linear_type)
        if linear_type == "shared_mlp":
            down_linear = get_moe_module(wrapper_module, f"shared_experts.{wrapper.down_name}").linear
        else:
            down_linear = get_moe_module(wrapper_module, wrapper.down_name).linear
        self.register_linear_wrapper(down_linear, quantize_type)

    def register_layer_moe_mlp_experts(self, layer, wrapper, quantize_type, expert_roster):
        wrapper_module = get_moe_module(layer, wrapper.wrapper_name)
        pack_type = wrapper_module.pack_type
        if pack_type in ALL_PACK_LIST:
            self.register_layer_moe_linear_pack(layer, wrapper, quantize_type, 'moe_mlp', expert_roster)
        else:
            self.register_layer_moe_linear_sep(layer, wrapper, quantize_type, 'moe_mlp', expert_roster)
        down_linear_list = []
        for i in expert_roster:
            down_name = f"{wrapper.down_name}.{i}"
            down_linear_list.append(get_moe_module(wrapper_module, down_name).linear)
        self.register_linear_list_wrapper(down_linear_list, quantize_type,
                                          hidden_dim=wrapper_module.hidden_dim, is_down_weight=True)

    def pad_linear_types(self, list_type, target_length):
        if list_type == "attention":
            self.attn_linear_types.append(self.layer_linear_type.copy())
            self.attn_linear_transpose_types.append(self.layer_linear_transpose_types.copy())
            for _ in range(target_length - len(self.attn_linear_types[-1])):
                self.attn_linear_types[-1].append(LinearType.INVALID)
                self.attn_linear_transpose_types[-1].append(-1)
        elif list_type == "shared":
            self.mlp_linear_types.append(self.layer_linear_type.copy())
            self.mlp_linear_transpose_types.append(self.layer_linear_transpose_types.copy())
            for _ in range(target_length - len(self.mlp_linear_types[-1])):
                self.mlp_linear_types[-1].append(LinearType.INVALID)
                self.mlp_linear_transpose_types[-1].append(-1)
        elif list_type == "moe":
            self.moe_linear_types.append(self.layer_linear_type.copy())
            self.moe_linear_transpose_types.append(self.layer_linear_transpose_types.copy())
            for _ in range(target_length - len(self.moe_linear_types[-1])):
                self.moe_linear_types[-1].append(LinearType.INVALID)
                self.moe_linear_transpose_types[-1].append(-1)
        self.layer_linear_type.clear()
        self.layer_linear_transpose_types.clear()

    def register_moe_layer(self, layer, quantize_type, dense_layer=False, expert_roster=None, attn_quantize_type=None):
        if quantize_type == QuantType.W8A8_DYNAMIC:
            self.gmm_quant_nd_nz = True
        if attn_quantize_type is None:
            attn_quantize_type = quantize_type
        if_shared_expert = False
        if not expert_roster:
            expert_roster = [i for i in range(self.num_experts)]
        self.register_layer_attn(layer, self.attn_wrapper, attn_quantize_type)
        self.pad_linear_types(list_type="attention", target_length=6)
        if dense_layer:
            self.register_layer_moe_mlp(layer, self.moe_mlp_wrapper, quantize_type, "dense_mlp")
            self.weights.append(self.placeholder)
            self.weights.append(self.placeholder)
            self.weights.extend([self.placeholder] * 20) # place holder for gmm quant & router quant
            self.weights.extend([self.placeholder] * 2)
            self.pad_linear_types(list_type=SHARED, target_length=4)
            self.pad_linear_types(list_type="moe", target_length=4)
        else:
            if self.shared_experts:
                if_shared_expert = True
                self.register_layer_moe_mlp(layer, self.moe_mlp_wrapper, quantize_type, "shared_mlp")
                self.weights.append(self.placeholder)
                self.weights.extend([self.placeholder] * 5)
                self.pad_linear_types(list_type=SHARED, target_length=4)
            self.register_layer_moe_mlp_experts(layer, self.moe_mlp_wrapper, quantize_type, expert_roster)
            self.pad_linear_types(list_type="moe", target_length=4)
            if not if_shared_expert:
                self.pad_linear_types(list_type=SHARED, target_length=4)

        attn_pack_type = get_moe_module(layer, self.attn_wrapper.wrapper_name).pack_type
        moe_mlp_pack_type = get_moe_module(layer, self.moe_mlp_wrapper.wrapper_name).pack_type
        self.pack_quant_type.append([attn_pack_type, moe_mlp_pack_type])
