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

#include "operations/fusion/attention/qkv_linear_split.h"
#include "operations/fusion/attention/self_attention.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/fusion/attention/fusion_attention.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetAttnInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> attnInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
            "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
            "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
            "in_weight_2", "in_scale_2", "in_offset_2", "in_descale_2", "in_bias_2", "in_compress_idx_2",
            "in_cos_embed", "in_sin_embed", "in_seq_len", "in_k_cache", "in_v_cache", "in_attention_mask",
            "in_token_offset", "in_layer_id", "in_block_tables",
            "in_slots_in_pa_or_logn_in_fa",
            "in_weight_dense", "in_scale_dense", "in_offset_dense", "in_descale_dense", "in_bias_dense",
            "in_compress_idx_dense"}
        },
        {"alibi_mask_compress", {"in_slopes"}},
        {"compress_head_alibi", {"in_batch_wins", "in_ra_seq_len"}},
        {"compress_head_rope",
            {"in_batch_wins", "in_ra_seq_len", "in_pffset_index", "in_ra_offset", "in_reshape_seq_len"}},
        {"speculate", {"in_q_len"}},
        {"kv_quant_scale",
            {"in_k_quant_scale", "in_k_dequant_scale", "in_v_quant_scale", "in_v_dequant_scale"}
        },
        {"kv_quant_offset",
            {"in_k_quant_offset", "in_k_dequant_offset", "in_v_quant_offset", "in_v_dequant_offset"}
        },
        {"fa3_quant",
            {"in_q_quant_scale", "in_k_quant_scale", "in_v_quant_scale", "in_qk_descale",
            "q_offset", "kv_offset", "fa3_v_quant_scale", "fa3_offset"}
        },
        {"reduce_quant",
            {"in_reduce_quant_scale", "in_reduce_quant_offset", "in_gather_quant_scale", "in_gather_quant_offset"}
        },
        {"lora", {
            "in_seq_len_cum_sum", "in_lora_a_0", "in_lora_b_0", "in_lora_a_1", "in_lora_b_1",
            "in_lora_a_2", "in_lora_b_2", "in_dense_lora_a", "in_dense_lora_b"}
        },
        {"lora_with_mask", {"in_im_mask"}},
        {"log_n_scale", {"in_log_n_scale"}},
    };
    return attnInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetAttnIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> attnIntermediateTensorCandidates = {
        {"default",
            {"intermediate_q", "intermediate_k", "intermediate_v", "intermediate_self_attention"}
        },
        {"kv_quant_scale",
            {"intermediate_k_int8", "intermediate_v_int8"}
        },
        {"q_quant_scale",
            {"intermediate_q_int8"}
        },
    };
    return attnIntermediateTensorCandidates;
}

template <typename NormParamType>
atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<NormParamType> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>> &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList, std::vector<std::string> &intermediateTensorList)
{
    // 添加KV cache int8特性的Tensor
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        AddTensorToList(attnInTensorCandidates, "kv_quant_scale", inTensorList);
        AddTensorToList(attnIntermediateTensorCandidates, "kv_quant_scale", intermediateTensorList);
        if (param.pageAttentionParam.hasQuantOffset) {
            AddTensorToList(attnInTensorCandidates, "kv_quant_offset", inTensorList);
        }
    }

    // 添加FA3特性的Tensor
    if (!param.isPrefill && param.pageAttentionParam.quantType == \
        atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        AddTensorToList(attnIntermediateTensorCandidates, "q_quant_scale", intermediateTensorList);
    }
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        AddTensorToList(attnInTensorCandidates, "fa3_quant", inTensorList);
        AddTensorToList(attnIntermediateTensorCandidates, "kv_quant_scale", intermediateTensorList);
    }
    return atb::NO_ERROR;
}


template <typename NormParamType>
std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto attnInTensorCandidates = GetAttnInTensorCandidates();
    auto attnIntermediateTensorCandidates = GetAttnIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(attnInTensorCandidates, "default", inTensorList);
    AddTensorToList(attnIntermediateTensorCandidates, "default", intermediateTensorList);

    // 添加Mask Alibi特性的Tensor
    if (
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN
    ) {
        AddTensorToList(attnInTensorCandidates, "alibi_mask_compress", inTensorList);
    }

    // 添加多头压缩特性的Tensor
    if (param.pageAttentionParam.compressType == atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD) {
        AddTensorToList(attnInTensorCandidates, "compress_head_alibi", inTensorList);
    } else if (param.pageAttentionParam.compressType ==
                    atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE) {
        AddTensorToList(attnInTensorCandidates, "compress_head_rope", inTensorList);
    }
    // 添加并行解码特性的Tensor
    if (param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
        AddTensorToList(attnInTensorCandidates, "speculate", inTensorList);
    }

    ConstructAttentionQuantTensorMap(param, attnInTensorCandidates, attnIntermediateTensorCandidates,
        inTensorList, intermediateTensorList);

    // 添加lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(attnInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(attnInTensorCandidates, "lora", inTensorList);
    }
    // 添加lccl all reduce int8特性的Tensor
    if (param.selfOutLinearTensorParallelInfo.quantType != \
        atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        AddTensorToList(attnInTensorCandidates, "reduce_quant", inTensorList);
    }

    // 添加logN attention特性
    if (param.pageAttentionParam.scaleType == atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN) {
        AddTensorToList(attnInTensorCandidates, "log_n_scale", inTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
atb::Status AddFAttnQKVLinearSplitNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qkvLinearSplitNode;
    CHECK_OPERATION_STATUS_RETURN(QKVLinearSplit(param, &qkvLinearSplitNode.operation));
    std::vector<std::string> qkvInTensorNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
        "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
        "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
        "in_weight_2", "in_scale_2", "in_offset_2", "in_descale_2", "in_bias_2", "in_compress_idx_2",
    };
    if (param.supportLora) {
        if (param.useImMask) {
            qkvInTensorNames.push_back("in_im_mask");
        }
        qkvInTensorNames.push_back("in_seq_len_cum_sum");
        qkvInTensorNames.push_back("in_lora_a_0");
        qkvInTensorNames.push_back("in_lora_b_0");
        qkvInTensorNames.push_back("in_lora_a_1");
        qkvInTensorNames.push_back("in_lora_b_1");
        qkvInTensorNames.push_back("in_lora_a_2");
        qkvInTensorNames.push_back("in_lora_b_2");
    }
    qkvLinearSplitNode.inTensorIds = GetTensorIdxList(tensorMap, qkvInTensorNames);
    qkvLinearSplitNode.outTensorIds = \
        GetTensorIdxList(tensorMap, {"intermediate_q", "intermediate_k", "intermediate_v"});
    opGraph.nodes.push_back(qkvLinearSplitNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddFAttnRopeNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node ropeNode;
    atb_speed::common::RotaryPositionEmbeddingParam ropeParam;
    ropeParam.rotaryType = param.rotaryType;
    ropeParam.isFA = param.isFA;
    ropeParam.headDim = param.headDim;
    ropeParam.headNum = param.selfAttentionParam.headNum;
    ropeParam.kvHeadNum = param.selfAttentionParam.kvHeadNum;
    ropeParam.ropeParam = param.ropeParam;

    RotaryPositionEmbedding(ropeParam, &ropeNode.operation);

    ropeNode.inTensorIds = {  // [B,S,N,D] PA [BS,ND]
        GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "in_cos_embed"), GetTensorIdx(tensorMap, "in_sin_embed"),
        GetTensorIdx(tensorMap, "in_seq_len")
    };
    if (!param.isFA) {
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        ropeNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
        ropeNode.inTensorReshapeFuncs.at(1) = &SqueezeHeadNumHeadDim;
    }
    ropeNode.outTensorIds = {  // FA [B,S,N,D] PA [BS,N,D]
        GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "intermediate_k"),
    };
    opGraph.nodes.push_back(ropeNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddKVValueQuantNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isK)
{
    atb::Node kvValueQuantNode;
    atb::infer::ElewiseParam kvValueQuantParam;
    kvValueQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CREATE_OPERATION(kvValueQuantParam, &kvValueQuantNode.operation);
    if (isK) {
        kvValueQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_k"), GetTensorIdx(tensorMap, "in_k_quant_scale"),
            GetTensorIdx(tensorMap, "in_k_quant_offset"),
        };
        kvValueQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_int8")};
    } else {
        kvValueQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_v"), GetTensorIdx(tensorMap, "in_v_quant_scale"),
            GetTensorIdx(tensorMap, "in_v_quant_offset"),
        };
        kvValueQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_int8")};
    }
    kvValueQuantNode.inTensorReshapeFuncs.resize(kvValueQuantNode.inTensorIds.size());
    kvValueQuantNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
    };
    kvValueQuantNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        UnsqueezeHeadNumHeadDim(oldShape, newShape, param.selfAttentionParam.kvHeadNum, param.headDim);
    };
    opGraph.nodes.push_back(kvValueQuantNode);
    return atb::NO_ERROR;
}

atb::Status AddQKVQuantNode(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, std::string nodeType)
{
    atb::Node qkvQuantNode;
    atb::infer::ElewiseParam qkvQuantParam;
    qkvQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CREATE_OPERATION(qkvQuantParam, &qkvQuantNode.operation);
    if (nodeType == "Q") {
        qkvQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "in_q_quant_scale"),
            GetTensorIdx(tensorMap, "q_offset")
        };
        qkvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q_int8")};
    } else if (nodeType == "K") {
        qkvQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_k"), GetTensorIdx(tensorMap, "in_k_quant_scale"),
            GetTensorIdx(tensorMap, "kv_offset")
        };
        qkvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_k_int8")};
    } else if (nodeType == "V") {
        qkvQuantNode.inTensorIds = {
            GetTensorIdx(tensorMap, "intermediate_v"), GetTensorIdx(tensorMap, "in_v_quant_scale"),
            GetTensorIdx(tensorMap, "kv_offset")
        };
        qkvQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_v_int8")};
    }
    qkvQuantNode.inTensorReshapeFuncs.resize(qkvQuantNode.inTensorIds.size());
    opGraph.nodes.push_back(qkvQuantNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSelfOutLinearParallelNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node selfOutLinearParallelNode;
    atb_speed::common::LinearParallelParam selfOutLinearParam;
    selfOutLinearParam.parallelType = atb_speed::common::ROW_PARALLEL;
    selfOutLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.layerLinearQuantType[DENSE_LINEAR_INDEX], false);
    selfOutLinearParam.biasAfterSync = param.selfOutLinearTensorParallelInfo.worldSize > 1 && \
        selfOutLinearParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT && \
        param.selfAttnHasBias;
    selfOutLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    selfOutLinearParam.fusionLinearParam.hasBias = param.selfAttnHasBias && !selfOutLinearParam.biasAfterSync;
    selfOutLinearParam.fusionLinearParam.supportLora = param.supportLora;
    selfOutLinearParam.fusionLinearParam.useImMask = param.useImMask;
    selfOutLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    selfOutLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[DENSE_LINEAR_INDEX];
    selfOutLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    selfOutLinearParam.tensorParallelInfo = param.selfOutLinearTensorParallelInfo;
    selfOutLinearParam.supportLcoc = param.supportLcoc;
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(selfOutLinearParam, &selfOutLinearParallelNode.operation));
    std::vector<std::string> denseInTensorNames = {
        "intermediate_self_attention", "in_weight_dense", "in_scale_dense", "in_offset_dense", "in_descale_dense",
        "in_bias_dense", "in_compress_idx_dense"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            denseInTensorNames.push_back("in_im_mask");
        }
        denseInTensorNames.push_back("in_seq_len_cum_sum");
        denseInTensorNames.push_back("in_dense_lora_a");
        denseInTensorNames.push_back("in_dense_lora_b");
    }
    if (param.selfOutLinearTensorParallelInfo.quantType != \
        atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        denseInTensorNames.push_back("in_reduce_quant_scale");
        denseInTensorNames.push_back("in_reduce_quant_offset");
        denseInTensorNames.push_back("in_gather_quant_scale");
        denseInTensorNames.push_back("in_gather_quant_offset");
    }
    selfOutLinearParallelNode.inTensorIds = GetTensorIdxList(tensorMap, denseInTensorNames);
    if (!param.isFA) {
        selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
        selfOutLinearParallelNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
    }
    selfOutLinearParallelNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(selfOutLinearParallelNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddQScaleNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node qScaleNode;
    atb::infer::ElewiseParam qScaleParam;
    qScaleParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    qScaleParam.mulsParam.varAttr = 1.0 / sqrt(param.headDim);
    CREATE_OPERATION(qScaleParam, &qScaleNode.operation);
    qScaleNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    qScaleNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_q")};
    opGraph.nodes.push_back(qScaleNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status Attention(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchSizePtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "Attention";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum " << opGraph.internalTensorNum);

    if (CheckParamVectorSize(param.layerLinearQuantType, DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearQuantType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (CheckParamVectorSize(param.layerLinearTransposeType, DENSE_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearTransposeType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    // QKV Node
    CHECK_OPERATION_STATUS_RETURN(AddFAttnQKVLinearSplitNode(param, opGraph, tensorMap));
    
    // Rope Node
    if (param.rotaryType != RotaryType::NO_ROTARY) {
        CHECK_OPERATION_STATUS_RETURN(AddFAttnRopeNode(param, opGraph, tensorMap));
    }

    // QScale Node
    if (param.enableQScale) {
        CHECK_OPERATION_STATUS_RETURN(AddQScaleNode(param, opGraph, tensorMap));
    }

    bool atbAttentionDequant = param.attnBackend == atb_speed::common::OpBackend::ATB && \
        param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION;
    bool aclnnAttentionDequant = param.attnBackend == atb_speed::common::OpBackend::ACLNN && \
        param.aclnnIncreAttentionParam.hasKVQuant;
    if (atbAttentionDequant || aclnnAttentionDequant) {
        // K Quant
        CHECK_OPERATION_STATUS_RETURN(AddKVValueQuantNode(param, opGraph, tensorMap, true));
        // V Quant
        CHECK_OPERATION_STATUS_RETURN(AddKVValueQuantNode(param, opGraph, tensorMap, false));
    }

    // FA3 QKV Quant Node
    if (!param.isPrefill && param.pageAttentionParam.quantType == \
        atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        CHECK_OPERATION_STATUS_RETURN(AddQKVQuantNode(opGraph, tensorMap, "Q"));
    }
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
            CHECK_OPERATION_STATUS_RETURN(AddQKVQuantNode(opGraph, tensorMap, "K"));
            CHECK_OPERATION_STATUS_RETURN(AddQKVQuantNode(opGraph, tensorMap, "V"));
    }

    // SelfAttention Node
    CHECK_OPERATION_STATUS_RETURN(AddSelfAttention(opGraph, param, tensorMap));

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

template atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>> &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList, std::vector<std::string> &intermediateTensorList);
template std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status AddFAttnRopeNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddKVValueQuantNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isK);
template atb::Status AddSelfOutLinearParallelNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddQScaleNode(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status Attention(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::Operation **operation);

template atb::Status ConstructAttentionQuantTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    std::map<std::string, std::vector<std::string>> &attnInTensorCandidates,
    std::map<std::string, std::vector<std::string>> &attnIntermediateTensorCandidates,
    std::vector<std::string> &inTensorList, std::vector<std::string> &intermediateTensorList);
template std::map<std::string, uint32_t> ConstructTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template atb::Status AddFAttnRopeNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddKVValueQuantNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isK);
template atb::Status AddSelfOutLinearParallelNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddQScaleNode(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status Attention(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::Operation **operation);
} // namespace common
} // namespace atb_speed