/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/deepseekv2/operation/latent_attention.h"
#include "models/deepseekv2/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseekV2 {

std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2LayerInTensorCandidates = {
        {"default", {
            "in_hidden_states", "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
            "in_final_state",
            "in_cos_table", "in_sin_table", "in_attention_mask", "in_k_cache", "in_v_cache", "in_seq_len",
            "in_place_holder", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"}},
        {"default_weight", {
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
            "in_q_proj_a_compress_idx", "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
            "in_q_proj_b_compress_idx", "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias",
            "in_kv_proj_with_mqa_descale", "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale",
            "in_kv_proj_with_mqa_compress_idx", "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
            "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
            "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
            "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
            "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
            "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale", "in_attention_out_offset",
            "in_attention_out_scale", "in_attention_out_compress_idx", "in_selfattention_out_norm_weight",
            "in_selfattention_out_norm_bias", "in_selfattention_out_new_norm_weight",
            "in_selfattention_out_new_norm_bias", "in_mlp_gateup_weight_shared_expert",
            "in_mlp_gateup_bias_shared_expert", "in_mlp_gateup_descale_shared_expert",
            "in_mlp_gateup_offset_shared_expert", "in_mlp_gateup_scale_shared_expert",
            "in_mlp_gateup_compress_idx_shared_expert", "in_mlp_down_weight_shared_expert",
            "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
            "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert",
            "in_mlp_down_compress_idx_shared_expert", "in_shared_expert_gate_weight", "in_shared_expert_gate_bias",
            "in_shared_expert_gate_descale", "in_shared_expert_gate_offset", "in_shared_expert_gate_scale",
            "in_shared_expert_gate_compress_idx", "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias",
            "in_block_sparse_moe_gate_descale", "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale",
            "in_block_sparse_moe_gate_compress_idx", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert", "in_mlp_down_bias_expert",
            "in_mlp_down_descale_expert", "in_mlp_down_offset_expert", "in_mlp_down_scale_expert",
            "in_mlp_down_compress_idx_expert"}},
        {"dp", {"in_dp_input_indices", "in_dp_gather_indices0", "in_dp_gather_indices1"}},
    };
    return deepseekV2LayerInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetDeepseekV2LayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> deepseekV2LayerIntermediateTensorCandidates = {
        {"dp", {"intermediate_dp_0", "intermediate_dp_1", "intermediate_dp_2", "intermediate_dp_3"}},
    };
    return deepseekV2LayerIntermediateTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const DecoderLayerParam &param, uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto deepseekV2InTensorCandidates = GetDeepseekV2LayerInTensorCandidates();
    auto deepseekV2IntermediateCandidates = GetDeepseekV2LayerIntermediateTensorCandidates();
    if (param.layerId < param.firstKDenseReplace) {
        if (param.tensorParallelInfo.worldSize <= 1) {
            deepseekV2IntermediateCandidates["default"] = {
                "intermediate_attention_out", "intermediate_mlp_out", "intermediate_selfattention_norm_out"};
        } else {
            deepseekV2IntermediateCandidates["default"] = {
                "intermediate_attention_out", "intermediate_mlp_out", "intermediate_selfattention_norm_out",
                "intermediate_moe_out_with_shared"};
        }
    } else {
        deepseekV2IntermediateCandidates["default"] = {
            "intermediate_attention_out", "intermediate_mlp_out", "intermediate_selfattention_norm_out",
            "intermediate_moe_out_with_shared"};
        deepseekV2IntermediateCandidates["shared_expert"] = {"intermediate_shared_expert_out"};
    }

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out_decoder_layer"};

    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default_weight", inTensorList);
    atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "default", inTensorList);
    atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "default", intermediateTensorList);

    if (param.hasSharedExpert && param.layerId >= param.firstKDenseReplace) {
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "shared_expert", intermediateTensorList);
    }

    if (param.hasDP) {
        atb_speed::common::AddTensorToList(deepseekV2InTensorCandidates, "dp", inTensorList);
        atb_speed::common::AddTensorToList(deepseekV2IntermediateCandidates, "dp", intermediateTensorList);
    }

    inTensorNum = inTensorList.size();
    internalTensorNum = intermediateTensorList.size();
    outTensorNum = outTensorList.size();

    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

atb::Status SetLatentAttentionParam(
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> &latentAttentionParam,
    const DecoderLayerParam &param)
{
    latentAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    latentAttentionParam.isBF16 = param.isBF16;
    latentAttentionParam.attnLinearQuantType = param.attnLinearQuantType;
    latentAttentionParam.packQuantType = param.packQuantType.at(0);
    latentAttentionParam.attnLinearTransposeType = param.attnLinearTransposeType;
    latentAttentionParam.enableLcoc = param.enableLcoc;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.normEps;
    latentAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    latentAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    latentAttentionParam.qLoraRank = param.qLoraRank;
    latentAttentionParam.headNum = param.headNum;
    latentAttentionParam.qkNopeHeadDim = param.qkNopeHeadDim;
    latentAttentionParam.qkRopeHeadDim = param.qkRopeHeadDim;
    latentAttentionParam.kvLoraRank = param.kvLoraRank;
    latentAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    latentAttentionParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
    latentAttentionParam.isFA = param.isFA;
    latentAttentionParam.isPrefill = param.isPrefill;
    latentAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    latentAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.selfAttentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    latentAttentionParam.selfAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        latentAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        latentAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    latentAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    latentAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    latentAttentionParam.pageAttentionParam.kvHeadNum = 1;
    latentAttentionParam.pageAttentionParam.mlaVHeadSize = param.kvLoraRank;
    latentAttentionParam.pageAttentionParam.qkScale = param.softmaxScale;
    latentAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    latentAttentionParam.selfOutLinearTensorParallelInfo = {
        param.tpRank, param.tpSize, param.backend, param.tpRankTableFile, param.tpDomain};
    latentAttentionParam.reshapeCacheParm.kvCacheCfg = atb::infer::ReshapeAndCacheParam::KvCacheCfg::K_CACHE_V_BYPASS;
    return atb::NO_ERROR;
}

int64_t SetAttentionNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                         std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node attentionNode;
    atb_speed::common::LatentAttentionParam<atb::infer::RmsNormParam> latentAttentionParam;
    SetLatentAttentionParam(latentAttentionParam, param);
    CHECK_OPERATION_STATUS_RETURN(Attention(latentAttentionParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
        "in_hidden_states",
        "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
        "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale",
        "in_q_proj_a_offset", "in_q_proj_a_scale", "in_q_proj_a_compress_idx",
        "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
        "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale",
        "in_q_proj_b_offset", "in_q_proj_b_scale", "in_q_proj_b_compress_idx",
        "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
        "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
        "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
        "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
        "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
        "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
        "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
        "in_attention_out_weight", "in_attention_out_bias", "in_attention_out_descale",
        "in_attention_out_offset", "in_attention_out_scale", "in_attention_out_compress_idx",
        "in_cos_table", "in_sin_table", "in_seq_len", "in_k_cache",
        "in_attention_mask", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"
    };
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
    opGraph.nodes.push_back(attentionNode);
    ATB_SPEED_LOG_DEBUG("Attention calculation success");
    return atb::NO_ERROR;
}

atb::Status SetSelfResidualAddNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attention_out"});
    selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attention_out"});
    opGraph.nodes.push_back(selfResidualAddNode);
    ATB_SPEED_LOG_DEBUG("SelfResidualAdd calculation success");
    return atb::NO_ERROR;
}

int64_t SetSelfNormNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                        std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfNormNode;
    atb::infer::RmsNormParam mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    std::vector<std::string> selfNormInTensorNames;
    std::vector<std::string> selfNormOutTensorNames;
    selfNormOutTensorNames.push_back("intermediate_selfattention_norm_out");
    if (param.enableAddNorm) {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormParam.preNormParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormQuantParam.preNormParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
        selfNormInTensorNames.push_back("in_hidden_states");
        selfNormOutTensorNames.push_back("intermediate_attention_out");
    } else {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormParam.normParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormQuantParam.normParam.epsilon = param.normEps;
        mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    }
    selfNormInTensorNames.push_back("intermediate_attention_out");
    if (param.mlpNormQuantType == atb::infer::QUANT_INT8) { // w8a8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormQuantParam, &selfNormNode.operation));
        if (param.isAntiOutlier) {
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
        } else {
            selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
            selfNormInTensorNames.push_back("in_selfattention_out_norm_bias");
        }
    } else if (param.normHasBias) { // FP
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
        selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
        selfNormInTensorNames.push_back("in_selfattention_out_new_norm_bias");
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mlpRmsNormParam, &selfNormNode.operation));
        if (param.isAntiOutlier) {
            selfNormInTensorNames.push_back("in_selfattention_out_new_norm_weight");
        } else {
            selfNormInTensorNames.push_back("in_selfattention_out_norm_weight");
        }
    }
    selfNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormInTensorNames);
    selfNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, selfNormOutTensorNames);
    opGraph.nodes.push_back(selfNormNode);
    ATB_SPEED_LOG_DEBUG("SelfNorm calculation success");
    return atb::NO_ERROR;
}

int64_t SetAllGatherNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                         std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allGatherBeforeNode;
    atb::infer::GatherParam gatherParam0;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherParam0, &allGatherBeforeNode.operation));
    allGatherBeforeNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_selfattention_norm_out", "in_dp_gather_indices0"});
    allGatherBeforeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_0"});
    opGraph.nodes.push_back(allGatherBeforeNode);
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.dpRank;
    allGatherParam.rankSize = param.dpSize;
    allGatherParam.backend = param.backend;
    allGatherParam.rankTableFile = param.dpRankTableFile;
    allGatherParam.commDomain = param.dpDomain;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_0"});
    allGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_1"});
    opGraph.nodes.push_back(allGatherNode);
    atb::Node allGatherAfterNode;
    atb::infer::GatherParam gatherParam1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherParam1, &allGatherAfterNode.operation));
    allGatherAfterNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_1", "in_dp_gather_indices1"});
    allGatherAfterNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_2"});
    allGatherAfterNode.inTensorReshapeFuncs.reserve(allGatherAfterNode.inTensorIds.size());
    allGatherAfterNode.inTensorReshapeFuncs.resize(allGatherAfterNode.inTensorIds.size());
    allGatherAfterNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2：新shape维度为2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0, 0, 1： 新shape前两维合轴
        newShape.dims[1] = oldShape.dims[2]; // 1, 2: 新shape最后一维不变
    };
    opGraph.nodes.push_back(allGatherAfterNode);
    ATB_SPEED_LOG_DEBUG("AllGather calculation success");
    return atb::NO_ERROR;
}

int64_t SetmlpExpertNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                         std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mlpExpertNode;
    atb_speed::common::SharedExpertParam mlpExpertParam;
    mlpExpertParam.isBF16 = param.isBF16;
    mlpExpertParam.transposeGateup = param.transpose;
    mlpExpertParam.transposeDown = param.transpose;
    mlpExpertParam.hasSharedExpertGate = false;
    mlpExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    mlpExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    mlpExpertParam.packQuantType = param.packQuantType.at(1);
    atb_speed::common::CreateSharedExpertOperation(mlpExpertParam, &mlpExpertNode.operation);
    std::vector<std::string> mlpExpertInTensorNames = {
        param.hasDP ? "intermediate_dp_2" : "intermediate_selfattention_norm_out",
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    mlpExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpExpertInTensorNames);
    if (param.layerId < param.firstKDenseReplace && param.tensorParallelInfo.worldSize <= 1) {
        mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    } else {
        mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(
            tensorMap, {"intermediate_moe_out_with_shared"});
    }
    opGraph.nodes.push_back(mlpExpertNode);
    ATB_SPEED_LOG_DEBUG("mlp expert calculation success");
    return atb::NO_ERROR;
}

int64_t SetMoe(atb::GraphParam &opGraph, const DecoderLayerParam &param, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node moeNode;
    atb_speed::common::SparseMoeParam sparseMoeParam;
    sparseMoeParam.isBF16 = param.isBF16;
    sparseMoeParam.transpose = param.transpose;
    sparseMoeParam.numOfExperts = param.numOfExperts;
    sparseMoeParam.num = param.numOfSelectedExperts;
    sparseMoeParam.routingMethod = param.routingMethod;
    sparseMoeParam.numOfGroups = param.numOfGroups;
    sparseMoeParam.topkGroups = param.topkGroups;
    sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
    sparseMoeParam.routedScalingFactor = param.routedScalingFactor;
    sparseMoeParam.processLogits = param.processLogits;
    sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
    sparseMoeParam.packQuantType = param.packQuantType.at(1);
    sparseMoeParam.enableFusedRouting = param.enableFusedRouting;
    atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("SparseMoe op is nullptr: ");
    }
    std::vector<std::string> moeInTensorNames = {
        param.hasDP ? "intermediate_dp_2" : "intermediate_selfattention_norm_out",
        "in_block_sparse_moe_gate_weight", "in_block_sparse_moe_gate_bias", "in_block_sparse_moe_gate_descale",
        "in_block_sparse_moe_gate_offset", "in_block_sparse_moe_gate_scale", "in_block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert", "in_mlp_gateup_descale_expert",
        "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert", "in_mlp_gateup_compress_idx_expert",
        "in_mlp_down_weight_expert", "in_mlp_down_bias_expert", "in_mlp_down_descale_expert",
        "in_mlp_down_offset_expert", "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert",
        "in_expert_array", "in_expert_group", "in_one_hot", "in_zero_hot",
    };
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, moeInTensorNames);
    if (!param.hasSharedExpert && param.tensorParallelInfo.worldSize <= 1) {
        moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    } else {
        moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(
            tensorMap, {"intermediate_moe_out_with_shared"});
    }
    opGraph.nodes.push_back(moeNode);
    ATB_SPEED_LOG_DEBUG("Moe sparse calculation success");
    return atb::NO_ERROR;
}

int64_t SetSharedExpert(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                        std::map<std::string, uint32_t> tensorMap)
{
    atb::Node sharedExpertNode;
    atb_speed::common::SharedExpertParam sharedExpertParam;
    sharedExpertParam.isBF16 = param.isBF16;
    sharedExpertParam.transposeGateup = param.transpose;
    sharedExpertParam.transposeDown = param.transpose;
    sharedExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
    sharedExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    sharedExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    sharedExpertParam.packQuantType = param.packQuantType.at(1);
    atb_speed::common::CreateSharedExpertOperation(sharedExpertParam, &sharedExpertNode.operation);
    std::vector<std::string> sharedExpertInTensorNames = {
        param.hasDP ? "intermediate_dp_2" : "intermediate_selfattention_norm_out",
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert",
        "in_mlp_down_descale_shared_expert", "in_mlp_down_offset_shared_expert",
        "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    sharedExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, sharedExpertInTensorNames);
    sharedExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_shared_expert_out"});
    opGraph.nodes.push_back(sharedExpertNode);
    ATB_SPEED_LOG_DEBUG("Shared expert calculation success");
    return atb::NO_ERROR;
}

int64_t AddExpertAdd(
    atb::GraphParam &opGraph,
    const DecoderLayerParam &param,
    std::map<std::string, uint32_t> tensorMap)
{
    atb::Node expertAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
    expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_moe_out_with_shared",
                                                                                "intermediate_shared_expert_out"});
    if (param.tensorParallelInfo.worldSize <= 1) {
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    } else {
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(
            tensorMap, {"intermediate_moe_out_with_shared"});
    }
    opGraph.nodes.push_back(expertAddNode);
    ATB_SPEED_LOG_DEBUG("create add operation");
    return atb::NO_ERROR;
}

int64_t SetAllReduce(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                     std::map<std::string, uint32_t> tensorMap)
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.backend = param.backend;
    if (param.hasDP) {
        allReduceParam.rank = param.esRank;
        allReduceParam.rankSize = param.esSize;
        allReduceParam.rankTableFile = param.esRankTableFile;
        allReduceParam.commDomain = param.esDomain;
    } else {
        allReduceParam.rank = param.tpRank;
        allReduceParam.rankSize = param.tpSize;
        allReduceParam.rankTableFile = param.tpRankTableFile;
        allReduceParam.commDomain = param.tpDomain;
    }
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap,
        {"intermediate_moe_out_with_shared"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out"});
    opGraph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

int64_t SetInvertAllGather(atb::GraphParam &opGraph, std::map<std::string, uint32_t> tensorMap)
{
    atb::Node invertAllGatherNode;
    atb::infer::GatherParam gatherParam;
    atb::CreateOperation(gatherParam, &invertAllGatherNode.operation);
    invertAllGatherNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_mlp_out",
                                                                                      "in_dp_input_indices"});
    invertAllGatherNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_dp_3"});
    opGraph.nodes.push_back(invertAllGatherNode);
    ATB_SPEED_LOG_DEBUG("create invertAllGatherNode");
    return atb::NO_ERROR;
}

atb::Status SetMlpResidualAddNode(atb::GraphParam &opGraph, const DecoderLayerParam &param,
                                  std::map<std::string, uint32_t> tensorMap)
{
    atb::Node mlpResidualAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    std::vector<std::string> mlpResidualAddInTensorNames = {
        "intermediate_attention_out",
        param.hasDP ? "intermediate_dp_3" : "intermediate_mlp_out"
    };
    mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpResidualAddInTensorNames);
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out_decoder_layer"});
    opGraph.nodes.push_back(mlpResidualAddNode);
    ATB_SPEED_LOG_DEBUG("create mlpResidualAdd");
    return atb::NO_ERROR;
}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph inTensorNum: " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph outTensorNum: " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("layer graph internalTensorNum: " << opGraph.internalTensorNum);

    CHECK_OPERATION_STATUS_RETURN(SetAttentionNode(opGraph, param, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(SetSelfResidualAddNode(opGraph, tensorMap));
    if (param.hasDP) {
        CHECK_OPERATION_STATUS_RETURN(SetAllGatherNode(opGraph, param, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(SetSelfNormNode(opGraph, param, tensorMap));
    if (param.layerId < param.firstKDenseReplace) {
        CHECK_OPERATION_STATUS_RETURN(SetmlpExpertNode(opGraph, param, tensorMap));
    } else {
        if (param.hasSharedExpert) {
            CHECK_OPERATION_STATUS_RETURN(SetSharedExpert(opGraph, param, tensorMap));
        }
        CHECK_OPERATION_STATUS_RETURN(SetMoe(opGraph, param, tensorMap));
        if (param.hasSharedExpert) {
            CHECK_OPERATION_STATUS_RETURN(AddExpertAdd(opGraph, param, tensorMap));
        }
    };
    if (param.tensorParallelInfo.worldSize > 1) {
        CHECK_OPERATION_STATUS_RETURN(SetAllReduce(opGraph, param, tensorMap));
    }
    if (param.hasDP) {
        CHECK_OPERATION_STATUS_RETURN(SetInvertAllGather(opGraph, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNode(opGraph, param, tensorMap));
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

DecoderLayer::DecoderLayer() {}

DecoderLayer::~DecoderLayer() {}

} // namespace deepseekV2
} // namespace atb_speed
