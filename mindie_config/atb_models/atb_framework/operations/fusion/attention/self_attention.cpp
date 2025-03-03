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
#include <securec.h>
#include "operations/fusion/utils.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/aclnn/ops/attn_operation.h"
#include "operations/fusion/attention/self_attention.h"

namespace atb_speed {
namespace common {

template <typename NormParamType>
int64_t AddSelfAttention(
    atb::GraphParam& opGraph, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    if (!param.isFA) {  // PA
        CHECK_OPERATION_STATUS_RETURN(AddPaKVCacheOperation(opGraph, param, tensorMap));
    } else if (param.isFA && param.attnBackend == atb_speed::common::OpBackend::ACLNN) {  // ACLNN FA
        CHECK_OPERATION_STATUS_RETURN(AddFaKVCacheOperation(opGraph, param, tensorMap));
    }
    // SelfAttentionNode
    atb::Node selfAttentionNode;
    if (param.isFA) { // FA
        if (param.attnBackend == atb_speed::common::OpBackend::ACLNN && param.isPrefill) {  // ACLNN FA Encode
            CHECK_OPERATION_STATUS_RETURN(ConstructPaEncoderNode(selfAttentionNode, param, tensorMap));
        } else if (param.attnBackend == atb_speed::common::OpBackend::ACLNN && !param.isPrefill) {  // ACLNN FA Decode
            CHECK_OPERATION_STATUS_RETURN(ConstructAclNNDecoderNode(selfAttentionNode, param, tensorMap));
        } else {  // ATB FA
            CHECK_OPERATION_STATUS_RETURN(ConstructFaNode(selfAttentionNode, param, tensorMap));
        }
    } else {
        if (param.isPrefill && !param.enableSplitFuse) {  // PA Prefill
            CHECK_OPERATION_STATUS_RETURN(ConstructPaEncoderNode(selfAttentionNode, param, tensorMap));
        } else if (param.attnBackend == atb_speed::common::OpBackend::ATB) {  // ATB PA Decode
            CHECK_OPERATION_STATUS_RETURN(ConstructPaDecoderNode(selfAttentionNode, param, tensorMap));
        } else if (param.attnBackend == atb_speed::common::OpBackend::ACLNN) {  // ACLNN PA Decode
            CHECK_OPERATION_STATUS_RETURN(ConstructAclNNDecoderNode(selfAttentionNode, param, tensorMap));
        }
    }

    selfAttentionNode.outTensorIds = { GetTensorIdx(tensorMap, "intermediate_self_attention") };
    opGraph.nodes.push_back(selfAttentionNode);

    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddFaKVCacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    // moveKCache Node
    atb::infer::KvCacheParam kvCacheParam;
    atb::Node moveKCacheNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kvCacheParam, &moveKCacheNode.operation));
    moveKCacheNode.inTensorIds = {
        param.aclnnIncreAttentionParam.hasKVQuant ? \
            GetTensorIdx(tensorMap, "intermediate_k_int8") : GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "in_layer_id"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_token_offset"),
        GetTensorIdx(tensorMap, "in_seq_len"),
    };
    moveKCacheNode.inTensorReshapeFuncs.resize(moveKCacheNode.inTensorIds.size());
    moveKCacheNode.inTensorReshapeFuncs.at(0) = // 0: [B,S,N,D]=>[BS,ND]
        &SqueezeBatchAndHiddenSize;
    moveKCacheNode.inTensorReshapeFuncs.at(2) = [=](  // 2: [B,S,ND]=>[1,B,S,ND]
        const atb::Dims& oldShape, atb::Dims& newShape) {
        UnsqueezeAxis(oldShape, newShape, 0);
    };
    moveKCacheNode.outTensorIds = {};
    opGraph.nodes.push_back(moveKCacheNode);

    atb::Node moveVCacheNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(kvCacheParam, &moveVCacheNode.operation));
    moveVCacheNode.inTensorIds = {
        param.aclnnIncreAttentionParam.hasKVQuant ? \
            GetTensorIdx(tensorMap, "intermediate_v_int8") : GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "in_layer_id"),
        GetTensorIdx(tensorMap, "in_v_cache"),
        GetTensorIdx(tensorMap, "in_token_offset"),
        GetTensorIdx(tensorMap, "in_seq_len"),
    };
    moveVCacheNode.inTensorReshapeFuncs.resize(moveVCacheNode.inTensorIds.size());
    moveVCacheNode.inTensorReshapeFuncs.at(0) = // 0: [B,S,N,D]=>[BS,ND]
        &SqueezeBatchAndHiddenSize;
    moveVCacheNode.inTensorReshapeFuncs.at(2) = [=](  // 2: [B,S,ND]=>[1,B,S,ND]
        const atb::Dims& oldShape, atb::Dims& newShape) {
        UnsqueezeAxis(oldShape, newShape, 0);
    };
    moveVCacheNode.outTensorIds = {};
    opGraph.nodes.push_back(moveVCacheNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t AddPaKVCacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    // ReshapeAndCache Node
    atb::Node reshapeAndCacheNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.reshapeCacheParm, &reshapeAndCacheNode.operation));
    reshapeAndCacheNode.inTensorIds = {
        param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION || \
        param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE ? \
            GetTensorIdx(tensorMap, "intermediate_k_int8") : GetTensorIdx(tensorMap, "intermediate_k"),
        param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION || \
        param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE ? \
            GetTensorIdx(tensorMap, "intermediate_v_int8") : GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"),
        GetTensorIdx(tensorMap, "in_slots_in_pa_or_logn_in_fa"),
    };
    if (param.reshapeCacheParm.compressType == \
        atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD
        ) {
        reshapeAndCacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_batch_wins"));
        reshapeAndCacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    }  else if (param.reshapeCacheParm.compressType == \
                   atb::infer::ReshapeAndCacheParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE) {
        reshapeAndCacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_batch_wins"));
        reshapeAndCacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_reshape_seq_len"));
        reshapeAndCacheNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_pffset_index"));
    }
    reshapeAndCacheNode.outTensorIds = {
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"),
    };
    opGraph.nodes.push_back(reshapeAndCacheNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ConstructAclNNDecoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    // 输入FA QKV [B,S,H] PA Q [B,S,N,D] KV [num_blocks,block_size,ND]
    // 输出FA [B,S,H] PA [BS,N,D]
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"), GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"), GetTensorIdx(tensorMap, "in_attention_mask")
    };
    if (param.aclnnIncreAttentionParam.isFA) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_token_offset"));
    } else {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    }
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_block_tables"));
    if (param.aclnnIncreAttentionParam.hasKVQuant) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_dequant_scale"));
        if (param.aclnnIncreAttentionParam.hasQuantOffset) {
            selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_dequant_offset"));
        }
    }
    selfAttentionNode.operation = new atb_speed::common::AttnOperation(
        "AclNNAttentionNode", param.aclnnIncreAttentionParam);
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    if (param.isFA)  {
        selfAttentionNode.inTensorReshapeFuncs.at(0) = &SqueezeHeadNumHeadDim;
    } else {
        selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims& oldShape, atb::Dims& newShape) {
            UnsqueezeAxis(oldShape, newShape, 1);
        };
        selfAttentionNode.inTensorReshapeFuncs.at(1) = &SqueezeHeadNumHeadDim;  // 1: in_k_cache
        selfAttentionNode.inTensorReshapeFuncs.at(2) = &SqueezeHeadNumHeadDim;  // 2: in_v_cache
    }
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ConstructFaNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    // 输入[nTokens, vHiddenSize] 输出[nTokens, vHiddenSize]
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "intermediate_v"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"),
    };
    if (param.selfAttentionParam.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_token_offset"));
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_layer_id"));
    if (param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN
    ) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_slopes"));
    }
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(0) = &SqueezeBatchAndHiddenSize;  // 0: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(1) = &SqueezeBatchAndHiddenSize;  // 1: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(2) = &SqueezeBatchAndHiddenSize;  // 2: [B,S,N,D]=>[BS,ND]
    selfAttentionNode.inTensorReshapeFuncs.at(3) = [=](  // 3: [BS,N,D]=>[1,BS,N,D]
        const atb::Dims& oldShape, atb::Dims& newShape) {
        UnsqueezeAxis(oldShape, newShape, 0);
    };
    selfAttentionNode.inTensorReshapeFuncs.at(4) = [=](  // 4: [BS,N,D]=>[1,BS,N,D]
        const atb::Dims& oldShape, atb::Dims& newShape) {
        UnsqueezeAxis(oldShape, newShape, 0);
    };
    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ConstructPaEncoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    // 输入[BS, N, D] 输出[BS, N, D]
    ATB_SPEED_LOG_DEBUG("enter ConstructPaEncoderNode");
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.selfAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "intermediate_k"),
        GetTensorIdx(tensorMap, "intermediate_v"),
    };
    if (param.selfAttentionParam.maskType != atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    if (
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT || \
        param.selfAttentionParam.maskType == atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN
        ) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_slopes"));
    }
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    if (param.attnBackend == atb_speed::common::OpBackend::ACLNN) {
        selfAttentionNode.inTensorReshapeFuncs.at(0) = &SqueezeBatchAndHiddenSize;  // 0: [B,S,N,D]=>[BS,ND]
        selfAttentionNode.inTensorReshapeFuncs.at(1) = &SqueezeBatchAndHiddenSize;  // 1: [B,S,N,D]=>[BS,ND]
        selfAttentionNode.inTensorReshapeFuncs.at(2) = &SqueezeBatchAndHiddenSize;  // 2: [B,S,N,D]=>[BS,ND]
    }

    return atb::NO_ERROR;
}

template <typename NormParamType>
int64_t ConstructPaDecoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<NormParamType>& param,
    std::map<std::string, uint32_t>& tensorMap)
{
    // 输出[num_tokens, N, D] [num_block,block_size,N,D]
    // 输出[num_tokens, num_head, head_size]
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.pageAttentionParam, &selfAttentionNode.operation));
    selfAttentionNode.inTensorIds = {
        param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE ? \
        GetTensorIdx(tensorMap, "intermediate_q_int8") : GetTensorIdx(tensorMap, "intermediate_q"),
        GetTensorIdx(tensorMap, "in_k_cache"),
        GetTensorIdx(tensorMap, "in_v_cache"),
        GetTensorIdx(tensorMap, "in_block_tables"),
    };
    if (param.pageAttentionParam.compressType == atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD ||
        param.pageAttentionParam.compressType ==
        atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_ra_seq_len"));
    } else {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_seq_len"));
    }
    if (param.pageAttentionParam.maskType != atb::infer::PagedAttentionParam::MaskType::UNDEFINED) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_attention_mask"));
    }
    if (param.pageAttentionParam.calcType == atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_q_len"));
    }
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_DEQUANT_FUSION) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_dequant_scale"));
        if (param.pageAttentionParam.hasQuantOffset) {
            selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_k_dequant_offset"));
        }
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_v_dequant_scale"));
        if (param.pageAttentionParam.hasQuantOffset) {
            selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_v_dequant_offset"));
        }
    }
    if (param.pageAttentionParam.quantType == atb::infer::PagedAttentionParam::QuantType::TYPE_QUANT_QKV_ONLINE) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_qk_descale"));
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "fa3_v_quant_scale"));
    }
    if (param.pageAttentionParam.compressType ==
            atb::infer::PagedAttentionParam::CompressType::COMPRESS_TYPE_KVHEAD_ROPE) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_ra_offset"));
    }
    if (param.pageAttentionParam.scaleType == atb::infer::PagedAttentionParam::ScaleType::SCALE_TYPE_LOGN) {
        selfAttentionNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_log_n_scale"));
    }
    return atb::NO_ERROR;
}


template int64_t AddFaKVCacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<atb::infer::RmsNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t AddPaKVCacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<atb::infer::RmsNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t AddSelfAttention(
    atb::GraphParam& opGraph, const FusionAttentionParam<atb::infer::RmsNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t ConstructFaNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<atb::infer::RmsNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t ConstructPaEncoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<atb::infer::RmsNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t ConstructPaDecoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<atb::infer::RmsNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);

template int64_t AddFaKVCacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<atb::infer::LayerNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t AddPaKVCacheOperation(
    atb::GraphParam& opGraph, const FusionAttentionParam<atb::infer::LayerNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t AddSelfAttention(
    atb::GraphParam& opGraph, const FusionAttentionParam<atb::infer::LayerNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t ConstructFaNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<atb::infer::LayerNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t ConstructPaEncoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<atb::infer::LayerNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);
template int64_t ConstructPaDecoderNode(
    atb::Node& selfAttentionNode, const FusionAttentionParam<atb::infer::LayerNormParam>& param,
    std::map<std::string, uint32_t>& tensorMap);

} // namespace common
} // namespace atb_speed