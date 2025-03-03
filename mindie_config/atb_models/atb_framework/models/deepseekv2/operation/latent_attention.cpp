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

#include "operations/fusion/utils.h"
#include "operations/aclnn/ops/repeat_operation.h"
#include "models/deepseekv2/operation/latent_attention.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetLatentAttnInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>>  latentAttnInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_weight_bias",
            "in_q_proj_a_weight", "in_q_proj_a_bias", "in_q_proj_a_descale", "in_q_proj_a_offset", "in_q_proj_a_scale",
            "in_q_proj_a_compress_idx",
            "in_q_proj_a_layernorm_weight", "in_q_proj_a_layernorm_bias",
            "in_q_proj_b_weight", "in_q_proj_b_bias", "in_q_proj_b_descale", "in_q_proj_b_offset", "in_q_proj_b_scale",
            "in_q_proj_b_compress_idx",
            "in_kv_proj_with_mqa_weight", "in_kv_proj_with_mqa_bias", "in_kv_proj_with_mqa_descale",
            "in_kv_proj_with_mqa_offset", "in_kv_proj_with_mqa_scale", "in_kv_proj_with_mqa_compress_idx",
            "in_kv_proj_a_layernorm_weight", "in_kv_proj_a_layernorm_bias",
            "in_k_proj_b_for_q_weight", "in_k_proj_b_for_q_bias", "in_k_proj_b_for_q_descale",
            "in_k_proj_b_for_q_offset", "in_k_proj_b_for_q_scale", "in_k_proj_b_for_q_compress_idx",
            "in_v_proj_b_for_o_weight", "in_v_proj_b_for_o_bias", "in_v_proj_b_for_o_descale",
            "in_v_proj_b_for_o_offset", "in_v_proj_b_for_o_scale", "in_v_proj_b_for_o_compress_idx",
            "in_attn_out_weight", "in_attn_out_bias", "in_attn_out_descale", "in_attn_out_offset",  "in_attn_out_scale",
            "in_attn_out_compress_idx",
            "in_cos_embed", "in_sin_embed", "in_seq_len", "in_k_cache", "in_attention_mask",
            "in_token_offset", "in_layer_id", "in_block_tables",
            "in_slots_in_pa_or_logn_in_fa"}
        },
    };
    return latentAttnInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLatentAttnIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> latentAttnIntermediateTensorCandidates = {
        {"default",
            {
                "in_input_norm", "latent_q", "latent_kv",
                "nope_q", "rope_q", "rope_k", "rope_q_o", "rope_k_o",
                "intermediate_q", "intermediate_k", "intermediate_v", "intermediate_self_attention"
            }
        },
        {"prefill",
            {
                "rope_k_o_repeat", "intermediate_k_nope", "intermediate_k_mha", "intermediate_v_mha",
                "temp_v_proj_b"
            }
        },
        {"decode",
            {
                "nope_q_transpose", "reproj_nope_q_transpose", "reproj_nope_q", "reproj_o_transpose", "reproj_o",
                "intermediate_self_attention_transpose",
            }
        },
        {"q_lora",
            {
                "latent_q_norm", "q_lora_out",
            }
        }
    };
    return latentAttnIntermediateTensorCandidates;
}

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(const LatentAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> outTensorList = {"out"};
    std::vector<std::string> intermediateTensorList = {};
    auto latentAttnInTensorCandidates = GetLatentAttnInTensorCandidates();
    auto latentAttnIntermediateTensorCandidates = GetLatentAttnIntermediateTensorCandidates();

    // 添加默认的Tensor
    AddTensorToList(latentAttnInTensorCandidates, "default", inTensorList);
    AddTensorToList(latentAttnIntermediateTensorCandidates, "default", intermediateTensorList);
    if (param.qLoraRank != 0) {
        AddTensorToList(latentAttnIntermediateTensorCandidates, "q_lora", intermediateTensorList);
    }
    if (param.isPrefill) {
        AddTensorToList(latentAttnIntermediateTensorCandidates, "prefill", intermediateTensorList);
    } else {
        AddTensorToList(latentAttnIntermediateTensorCandidates, "decode", intermediateTensorList);
    }
    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

void SqueezeHeadNumHeadDim(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dimNum == 4) {  // 4: FA
        newShape.dimNum = 3;  // 3: 新的shape维度为3
        newShape.dims[0] = oldShape.dims[0];  // 0, 0: 新shape的第0维不变
        newShape.dims[1] = oldShape.dims[1];  // 1, 1: 新shape的第1维不变
        newShape.dims[2] =  oldShape.dims[2] * oldShape.dims[3];  // 2, 2, 3: 后两维合轴
    } else {
        newShape.dimNum = 2;  // 2: 新的shape维度为2
        newShape.dims[0] = oldShape.dims[0];  // 0, 0: 新shape的第0维不变
        newShape.dims[1] =  oldShape.dims[1] * oldShape.dims[2];  // 1, 1, 2: 后两维合轴
    }
}


template <typename NormParamType>
bool UseNormQuant(const LatentAttentionParam<NormParamType> &param, uint64_t linearIndex)
{
    LinearQuantType quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[linearIndex], true);
    if (quantType == LinearQuantType::LINEAR_W8A8_DEQUANT || \
        quantType == LinearQuantType::LINEAR_W8A8_SC_DEQUANT) {
        return true;
    } else {
        return false;
    }
}

template <typename NormParamType>
atb::Status AddLAttnPreNormNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node normNode;

    if (UseNormQuant(param, Q_PROJ_A_LINEAR_INDEX)) {  // W8A8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &normNode.operation));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_input"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_weight"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_norm_bias"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_scale"));
        normNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_offset"));
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &normNode.operation));
        normNode.inTensorIds = {GetTensorIdx(tensorMap, "in_input"), GetTensorIdx(tensorMap, "in_norm_weight")};
    }
    normNode.outTensorIds = {GetTensorIdx(tensorMap, "in_input_norm")};
    opGraph.nodes.push_back(normNode);
    ATB_SPEED_LOG_DEBUG("Attention PreNorm calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnQProjANode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qAProjNode;
    atb_speed::common::FusionLinearParam qAProjNodeParam;
    qAProjNodeParam.isBF16 = param.isBF16;
    qAProjNodeParam.hasBias = param.selfAttnHasBias;
    qAProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[Q_PROJ_A_LINEAR_INDEX], true);
    qAProjNodeParam.quantGroupSize = param.quantGroupSize;
    qAProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_a_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_a_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_a_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_a_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_a_compress_idx"),
    };
    qAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_q")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qAProjNodeParam, &qAProjNode.operation));
    opGraph.nodes.push_back(qAProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_q_a calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnQProjBNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qNormNode;
    if (UseNormQuant(param, Q_PROJ_B_LINEAR_INDEX)) {  // W8A8
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normQuantParamType, &qNormNode.operation));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "latent_q"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_bias"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_scale"));
        qNormNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_proj_b_offset"));
    } else {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &qNormNode.operation));
        qNormNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q"),
                                GetTensorIdx(tensorMap, "in_q_proj_a_layernorm_weight")};
    }
    qNormNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_q_norm")};
    opGraph.nodes.push_back(qNormNode);

    atb::Node qBProjNode;
    atb_speed::common::FusionLinearParam qBProjNodeParam;
    qBProjNodeParam.isBF16 = param.isBF16;
    qBProjNodeParam.hasBias = param.selfAttnHasBias;
    qBProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[Q_PROJ_B_LINEAR_INDEX], true);
    qBProjNodeParam.quantGroupSize = param.quantGroupSize;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qBProjNodeParam, &qBProjNode.operation));
    qBProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "latent_q_norm"),
        GetTensorIdx(tensorMap, "in_q_proj_b_weight"),
        GetTensorIdx(tensorMap, "in_q_proj_b_scale"),
        GetTensorIdx(tensorMap, "in_q_proj_b_offset"),
        GetTensorIdx(tensorMap, "in_q_proj_b_descale"),
        GetTensorIdx(tensorMap, "in_q_proj_b_bias"),
        GetTensorIdx(tensorMap, "in_q_proj_b_compress_idx"),
    };
    qBProjNode.outTensorIds = {GetTensorIdx(tensorMap, "q_lora_out")};
    opGraph.nodes.push_back(qBProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_q_b calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddSplitQNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitQNode;
    atb::infer::SplitParam splitQParam = {(param.isFA ? 3 : 2), 2, {param.qkNopeHeadDim, param.qkRopeHeadDim}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitQNode.operation));
    if (param.qLoraRank == 0) {    // 如果是lite
        splitQNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_q")};
    } else {
        splitQNode.inTensorIds = {GetTensorIdx(tensorMap, "q_lora_out")};
    }
    splitQNode.inTensorReshapeFuncs.resize(splitQNode.inTensorIds.size());
    splitQNode.outTensorIds = {GetTensorIdxList(tensorMap, {"nope_q", "rope_q"})};
    if (param.isFA) {
        splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
                newShape.dimNum = 4; // 4: dimNum
                newShape.dims[0] = oldShape.dims[0];
                newShape.dims[1] = oldShape.dims[1];
                newShape.dims[2] = param.selfAttentionParam.headNum; // 2: dim id
                newShape.dims[3] = param.qkNopeHeadDim + param.qkRopeHeadDim; // 3: dim id
            };
    } else {
        splitQNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = param.selfAttentionParam.headNum;
            newShape.dims[2] = param.qkNopeHeadDim + param.qkRopeHeadDim; // 2: dim id
        };
    }
    opGraph.nodes.push_back(splitQNode);
    ATB_SPEED_LOG_DEBUG("MLA split q calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddReprojQNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node transposeQnopeInNode;
    atb::infer::TransposeParam transposeQnopeInParam;
    transposeQnopeInNode.inTensorIds = {GetTensorIdx(tensorMap, "nope_q")};
    transposeQnopeInNode.outTensorIds = {GetTensorIdx(tensorMap, "nope_q_transpose")};
    if (param.isFA) {
        transposeQnopeInParam.perm = {0, 2, 1, 3};
    } else {
        transposeQnopeInParam.perm = {1, 0, 2};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeQnopeInParam, &transposeQnopeInNode.operation));
    opGraph.nodes.push_back(transposeQnopeInNode);
    atb::Node qReprojNode;
    atb_speed::common::FusionLinearParam qReprojNodeParam;
    qReprojNodeParam.isBF16 = param.isBF16;
    qReprojNodeParam.hasBias = param.selfAttnHasBias;
    qReprojNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_Q_LINEAR_INDEX], false);
    qReprojNodeParam.quantGroupSize = param.quantGroupSize;
    qReprojNodeParam.transposeType = false;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(qReprojNodeParam, &qReprojNode.operation));
    qReprojNode.inTensorIds = {
        GetTensorIdx(tensorMap, "nope_q_transpose"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
    };
    qReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_nope_q_transpose")};
    opGraph.nodes.push_back(qReprojNode);
    atb::Node transposeQnopeOutNode;
    atb::infer::TransposeParam transposeQnopeOutParam;
    transposeQnopeOutNode.inTensorIds = {GetTensorIdx(tensorMap, "reproj_nope_q_transpose")};
    transposeQnopeOutNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_nope_q")};
    if (param.isFA) {
        transposeQnopeOutParam.perm = {0, 2, 1, 3};
    } else {
        transposeQnopeOutParam.perm = {1, 0, 2};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeQnopeOutParam, &transposeQnopeOutNode.operation));
    opGraph.nodes.push_back(transposeQnopeOutNode);
    ATB_SPEED_LOG_DEBUG("MLA reproj q calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnKVAProjNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kvAProjNode;
    atb_speed::common::FusionLinearParam kvAProjNodeParam;
    kvAProjNodeParam.isBF16 = param.isBF16;
    kvAProjNodeParam.hasBias = param.selfAttnHasBias;
    kvAProjNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_A_LINEAR_INDEX], true);
    kvAProjNodeParam.quantGroupSize = param.quantGroupSize;
    kvAProjNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_input_norm"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_weight"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_scale"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_offset"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_descale"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_bias"),
        GetTensorIdx(tensorMap, "in_kv_proj_with_mqa_compress_idx"),
    };
    kvAProjNode.outTensorIds = {GetTensorIdx(tensorMap, "latent_kv")};
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kvAProjNodeParam, &kvAProjNode.operation));
    opGraph.nodes.push_back(kvAProjNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_kv_a calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddSplitKNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam = {(param.isFA ? 2 : 1), 2, {param.kvLoraRank, param.qkRopeHeadDim}};
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {GetTensorIdx(tensorMap, "latent_kv")};
    splitKNode.outTensorIds = {GetTensorIdxList(tensorMap, {"intermediate_v", "rope_k"})};
    opGraph.nodes.push_back(splitKNode);
    ATB_SPEED_LOG_DEBUG("MLA spilt_k calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddLAttnKVNormNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kvNormNode;
    kvNormNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_v"), GetTensorIdx(tensorMap, "in_kv_proj_a_layernorm_weight")
    };
    kvNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v")};
    if (!param.isFA) {
        kvNormNode.inTensorReshapeFuncs.resize(kvNormNode.inTensorIds.size());
        kvNormNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = 1;
            newShape.dims[2] = param.kvLoraRank; // 2: dim id
        };
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.normParamType, &kvNormNode.operation));
    opGraph.nodes.push_back(kvNormNode);
    ATB_SPEED_LOG_DEBUG("MLA kv norm calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnRopeNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node ropeNode;
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.ropeParam.rotaryCoeff;
    CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {
        GetTensorIdx(tensorMap, "rope_q"), GetTensorIdx(tensorMap, "rope_k"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_seq_len")
    };
    ropeNode.outTensorIds = {
        GetTensorIdx(tensorMap, "rope_q_o"), GetTensorIdx(tensorMap, "rope_k_o"),
    };
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        SqueezeHeadNumHeadDim(oldShape, newShape);
    };
    opGraph.nodes.push_back(ropeNode);
    ATB_SPEED_LOG_DEBUG("MLA rope calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnQCatNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qCatNode;
    atb::infer::ConcatParam qCatParam;
    qCatParam.concatDim = -1;
    if (param.isPrefill) {
        qCatNode.inTensorIds = {
            GetTensorIdx(tensorMap, "nope_q"), GetTensorIdx(tensorMap, "rope_q_o")};
    } else {
        qCatNode.inTensorIds = {
            GetTensorIdx(tensorMap, "reproj_nope_q"), GetTensorIdx(tensorMap, "rope_q_o")
        };
    }
    qCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    qCatNode.inTensorReshapeFuncs.resize(qCatNode.inTensorIds.size());
    if (param.isFA) {
        qCatNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 4; // 4: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[1];
            newShape.dims[2] = param.selfAttentionParam.headNum; // 2: dim id
            newShape.dims[3] = param.qkRopeHeadDim; // 3: dim id
        };
    } else {
        qCatNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = param.selfAttentionParam.headNum;
            newShape.dims[2] = param.qkRopeHeadDim; // 2: dim id
        };
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(qCatParam, &qCatNode.operation));
    opGraph.nodes.push_back(qCatNode);
    ATB_SPEED_LOG_DEBUG("MLA qCatNode calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnKCatPrefillNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node keyRepeatNode;
    atb_speed::common::AclNNRepeatParam kvRepeatParam;
    kvRepeatParam.repeatsArray = {1, param.selfAttentionParam.headNum, 1};
    keyRepeatNode.inTensorIds = {GetTensorIdx(tensorMap, "rope_k_o")};
    keyRepeatNode.outTensorIds = {GetTensorIdx(tensorMap, "rope_k_o_repeat")};
    keyRepeatNode.inTensorReshapeFuncs.resize(keyRepeatNode.inTensorIds.size());
    keyRepeatNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim id
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
        newShape.dims[2] = oldShape.dims[1]; // 2:dim id
    };
    keyRepeatNode.operation = new atb_speed::common::RepeatOperation("RepeatNode", kvRepeatParam);
    opGraph.nodes.push_back(keyRepeatNode);

    atb::Node kCatNode;
    atb::infer::ConcatParam kCatParam;
    kCatParam.concatDim = 2; // 2: dim id
    kCatNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_k_nope"), GetTensorIdx(tensorMap, "rope_k_o_repeat")};
    kCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_mha")};
    kCatNode.inTensorReshapeFuncs.resize(kCatNode.inTensorIds.size());
    kCatNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim id
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.selfAttentionParam.headNum;
        newShape.dims[2] = param.qkNopeHeadDim; // 2:dim id
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kCatParam, &kCatNode.operation));
    opGraph.nodes.push_back(kCatNode);
    ATB_SPEED_LOG_DEBUG("MLA kCatNode prefill calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnKCatNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kCatNode;
    atb::infer::ConcatParam kCatParam;
    kCatParam.concatDim = 2; // 2: dim id
    kCatNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_v"), GetTensorIdx(tensorMap, "rope_k_o")
    };
    kCatNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k")};
    if (!param.isFA) {
        kCatNode.inTensorReshapeFuncs.resize(kCatNode.inTensorIds.size());
        kCatNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3; // 3: dimNum
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = 1;
            newShape.dims[2] = param.qkRopeHeadDim; // 2: dim id
        };
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kCatParam, &kCatNode.operation));
    opGraph.nodes.push_back(kCatNode);
    ATB_SPEED_LOG_DEBUG("MLA kCatNode calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddReprojVTransInNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node transposeVInNode;
    atb::infer::TransposeParam transposeVInParam;
    transposeVInNode.inTensorIds = {GetTensorIdx(tensorMap, "reproj_o")};
    transposeVInNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_o_transpose")};
    if (param.isFA) {
        transposeVInParam.perm = {0, 2, 1, 3};
    } else {
        transposeVInParam.perm = {1, 0, 2};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeVInParam, &transposeVInNode.operation));
    opGraph.nodes.push_back(transposeVInNode);
    ATB_SPEED_LOG_DEBUG("MLA reproj v transpose calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddReprojVNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node vReprojNode;
    atb_speed::common::FusionLinearParam vReprojNodeParam;
    vReprojNodeParam.isBF16 = param.isBF16;
    vReprojNodeParam.hasBias = param.selfAttnHasBias;
    vReprojNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_V_LINEAR_INDEX], false);
    vReprojNodeParam.quantGroupSize = param.quantGroupSize;
    vReprojNodeParam.transposeType = false;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(vReprojNodeParam, &vReprojNode.operation));
    vReprojNode.inTensorIds = {
        GetTensorIdx(tensorMap, "reproj_o_transpose"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_weight"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_scale"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_offset"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_descale"), GetTensorIdx(tensorMap, "in_v_proj_b_for_o_bias"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_compress_idx"),
    };
    vReprojNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_transpose")};
    opGraph.nodes.push_back(vReprojNode);
    atb::Node transposeVOutNode;
    atb::infer::TransposeParam transposeVOutParam;
    transposeVOutNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention_transpose")};
    transposeVOutNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
    if (param.isFA) {
        transposeVOutParam.perm = {0, 2, 1, 3}; // 2, 3: dim id
    } else {
        transposeVOutParam.perm = {1, 0, 2}; // 2: dim id
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeVOutParam, &transposeVOutNode.operation));
    opGraph.nodes.push_back(transposeVOutNode);
    ATB_SPEED_LOG_DEBUG("MLA reproj v calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnKProjBNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node kProjBNode;
    atb_speed::common::FusionLinearParam kProjBNodeParam;
    kProjBNodeParam.isBF16 = param.isBF16;
    kProjBNodeParam.hasBias = param.selfAttnHasBias;
    kProjBNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType
            : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_Q_LINEAR_INDEX], false);
    kProjBNodeParam.quantGroupSize = param.quantGroupSize;
    kProjBNodeParam.transposeType = true;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(kProjBNodeParam, &kProjBNode.operation));
    kProjBNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_weight"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_scale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_offset"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_descale"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_bias"),
        GetTensorIdx(tensorMap, "in_k_proj_b_for_q_compress_idx"),
    };
    kProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_nope")};

    kProjBNode.inTensorReshapeFuncs.resize(kProjBNode.inTensorIds.size());
    kProjBNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim id
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
    };
    kProjBNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim id
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
    };

    opGraph.nodes.push_back(kProjBNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_k_b calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddLAttnVProjBNode(const LatentAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node transposeVProjBWeightNode;
    atb::infer::TransposeParam transposeVProjBWeightParam;
    transposeVProjBWeightNode.inTensorIds = {GetTensorIdx(tensorMap, "in_v_proj_b_for_o_weight")};
    transposeVProjBWeightNode.outTensorIds = {GetTensorIdx(tensorMap, "temp_v_proj_b")};
    if (param.isFA) {
        transposeVProjBWeightParam.perm = {0, 1, 3, 2};
    } else {
        transposeVProjBWeightParam.perm = {0, 2, 1};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeVProjBWeightParam,
        &transposeVProjBWeightNode.operation));
    opGraph.nodes.push_back(transposeVProjBWeightNode);

    atb::Node vProjBNode;
    atb_speed::common::FusionLinearParam vProjBNodeParam;
    vProjBNodeParam.isBF16 = param.isBF16;
    vProjBNodeParam.hasBias = param.selfAttnHasBias;
    vProjBNodeParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED
            ? param.packQuantType
            : param.denseQuantType,
        param.attnLinearQuantType[KV_PROJ_B_FOR_V_LINEAR_INDEX], false);
    vProjBNodeParam.quantGroupSize = param.quantGroupSize;
    vProjBNodeParam.transposeType = true;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(vProjBNodeParam, &vProjBNode.operation));
    vProjBNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "temp_v_proj_b"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_scale"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_offset"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_descale"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_bias"),
        GetTensorIdx(tensorMap, "in_v_proj_b_for_o_compress_idx"),
    };
    vProjBNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_mha")};
    vProjBNode.inTensorReshapeFuncs.resize(vProjBNode.inTensorIds.size());
    vProjBNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim num
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
    };
    vProjBNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2: dim num
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2]; // 2: dim id
    };

    opGraph.nodes.push_back(vProjBNode);
    ATB_SPEED_LOG_DEBUG("MLA proj_v_b calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
atb::Status AddSelfOutLinearParallelNode(const LatentAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfOutLinearParallelNode;
    atb_speed::common::LinearParallelParam selfOutLinearParam;
    selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
    selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
    selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.attnLinearQuantType[O_LINEAR_INDEX], false);
    selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
    selfOutLinearParam.supportLcoc = param.enableLcoc;
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
    selfOutLinearParallelNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_self_attention"),
        GetTensorIdx(tensorMap, "in_attn_out_weight"),
        GetTensorIdx(tensorMap, "in_attn_out_scale"),
        GetTensorIdx(tensorMap, "in_attn_out_offset"),
        GetTensorIdx(tensorMap, "in_attn_out_descale"),
        GetTensorIdx(tensorMap, "in_attn_out_bias"),
        GetTensorIdx(tensorMap, "in_attn_out_compress_idx"),
    };
    selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
    if (!param.isFA) {
        selfOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            SqueezeHeadNumHeadDim(oldShape, newShape);
        };
    }
    selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(selfOutLinearParallelNode);
    ATB_SPEED_LOG_DEBUG("MLA o_proj calculation success");
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status Attention(const LatentAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "Attention";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(param,
        opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum " << opGraph.internalTensorNum);
    // PreNorm Node
    CHECK_OPERATION_STATUS_RETURN(AddLAttnPreNormNode(param, opGraph, tensorMap));
    // Q_proj Node
    CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjANode(param, opGraph, tensorMap));
    if (param.qLoraRank > 0) {
        CHECK_OPERATION_STATUS_RETURN(AddLAttnQProjBNode(param, opGraph, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(AddSplitQNode(param, opGraph, tensorMap));
    // KV_proj Node
    CHECK_OPERATION_STATUS_RETURN(AddLAttnKVAProjNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddSplitKNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddLAttnKVNormNode(param, opGraph, tensorMap));
    // Rope Node
    if (param.rotaryType != RotaryType::NO_ROTARY) {
        CHECK_OPERATION_STATUS_RETURN(AddLAttnRopeNode(param, opGraph, tensorMap));
    }
    if (param.isPrefill) {
        CHECK_OPERATION_STATUS_RETURN(AddLAttnKProjBNode(param, opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(AddLAttnKCatPrefillNode(param, opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(AddLAttnVProjBNode(param, opGraph, tensorMap));
    } else {
         // Reproj Q
        CHECK_OPERATION_STATUS_RETURN(AddReprojQNode(param, opGraph, tensorMap));
    }
    // Cat Node
    CHECK_OPERATION_STATUS_RETURN(AddLAttnQCatNode(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddLAttnKCatNode(param, opGraph, tensorMap));
    // SelfAttention Node
    CHECK_OPERATION_STATUS_RETURN(AddSelfAttention(opGraph, param, tensorMap));
    if (!param.isPrefill) {
        // Reproj V
        CHECK_OPERATION_STATUS_RETURN(AddReprojVTransInNode(param, opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(AddReprojVNode(param, opGraph, tensorMap));
    }
    // Dense Node
    CHECK_OPERATION_STATUS_RETURN(AddSelfOutLinearParallelNode(param, opGraph, tensorMap));

    opGraph.inferShapeFunc = [=]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}


template <typename NormParamType>
int64_t AddSelfAttention(
    atb::GraphParam &opGraph, const LatentAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    if (!param.isFA) {  // PA
        // ReshapeAndCache Node
        atb::Node reshapeAndCacheNode;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.reshapeCacheParm, &reshapeAndCacheNode.operation));
        reshapeAndCacheNode.inTensorIds = {
            param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION ? \
                GetTensorIdx(tensorMap, "intermediate_k_int8") : GetTensorIdx(tensorMap, "intermediate_k"),
            GetTensorIdx(tensorMap, "in_k_cache"),
            GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
        };
        reshapeAndCacheNode.outTensorIds = {
            GetTensorIdx(tensorMap, "in_k_cache"),
        };
        opGraph.nodes.push_back(reshapeAndCacheNode);
    }

    // SelfAttentionNode
    atb::Node selfAttentionNode;
    if (param.isFA) { // FA
        CHECK_OPERATION_STATUS_RETURN(ConstructFaNode(selfAttentionNode, param, tensorMap));
        selfAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_o")};
    } else {
        if (param.isPrefill) {  // PA Prefill
            CHECK_OPERATION_STATUS_RETURN(ConstructPaEncoderNode(selfAttentionNode, param, tensorMap));
            selfAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_self_attention")};
        } else {  // PA Decode
            CHECK_OPERATION_STATUS_RETURN(ConstructPaDecoderNode(selfAttentionNode, param, tensorMap));
            selfAttentionNode.outTensorIds = {GetTensorIdx(tensorMap, "reproj_o")};
        }
    }
    opGraph.nodes.push_back(selfAttentionNode);
    ATB_SPEED_LOG_DEBUG("MLA self-attention calculation success");
    return atb::NO_ERROR;
}

void UnSqueezeLayerAxis(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum + 1;
    newShape.dims[0] = 1;  // Layer Axis
    for (uint64_t i = 0; i < oldShape.dimNum; i++) {
        newShape.dims[i + 1] = oldShape.dims[i];
    }
}

template <typename NormParamType>
int64_t ConstructFaNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "in_k_cache"),
    };
    if (param.selfAttentionParam.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_token_offset"));
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_layer_id"));
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(3) =  // 3: 第三个K_CACHE使用UnSqueezeLayerAxis方法进行reshape
        &UnSqueezeLayerAxis;
    selfAttentionNode.inTensorReshapeFuncs.at(4) =  // 4: 第四个V_CACHE使用UnSqueezeLayerAxis方法进行reshape
        &UnSqueezeLayerAxis;
    ATB_SPEED_LOG_DEBUG("MLA FA calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
int64_t ConstructPaEncoderNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k_mha"),
        GetTensorIdx(tensorMap, "intermediate_v_mha"),
    };
    if (param.selfAttentionParam.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3: dim num
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.selfAttentionParam.headNum;
        newShape.dims[2] = param.qkNopeHeadDim; // 2: dim id
    };
    ATB_SPEED_LOG_DEBUG("MLA PA encoder calculation success");
    return atb::NO_ERROR;
}


template <typename NormParamType>
int64_t ConstructPaDecoderNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<NormParamType> &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.pageAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_block_tables"),
        GetTensorIdx(tensorMap, "in_seq_len")
    };
    if (param.pageAttentionParam.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    ATB_SPEED_LOG_DEBUG("MLA PA decoder calculation success");
    return atb::NO_ERROR;
}

template int64_t AddSelfAttention(
    atb::GraphParam &opGraph, const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructFaNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaEncoderNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaDecoderNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);

template int64_t AddSelfAttention(
    atb::GraphParam &opGraph, const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructFaNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaEncoderNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);
template int64_t ConstructPaDecoderNode(
    atb::Node &selfAttentionNode, const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, uint32_t> &tensorMap);


template std::map<std::string, uint32_t> ConstructTensorMap(
    const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status AddLAttnPreNormNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnQProjANode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnQProjBNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSplitQNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddReprojQNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnQCatNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKVAProjNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSplitKNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKVNormNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnRopeNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKCatNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKCatPrefillNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKProjBNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnVProjBNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddReprojVTransInNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddReprojVNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSelfOutLinearParallelNode(const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status Attention(
    const LatentAttentionParam<atb::infer::RmsNormParam> &param,
    atb::Operation **operation);


template std::map<std::string, uint32_t> ConstructTensorMap(
    const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status AddLAttnPreNormNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnQProjANode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnQProjBNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSplitQNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddReprojQNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnQCatNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKVAProjNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSplitKNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKVNormNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnRopeNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKCatNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKCatPrefillNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnKProjBNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddLAttnVProjBNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddReprojVTransInNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddReprojVNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSelfOutLinearParallelNode(const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status Attention(
    const LatentAttentionParam<atb::infer::LayerNormParam> &param,
    atb::Operation **operation);

} // namespace common
} // namespace atb_speed