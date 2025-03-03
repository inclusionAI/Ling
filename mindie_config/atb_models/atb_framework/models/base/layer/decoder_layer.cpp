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
#include "operations/fusion/norm/norm_linear.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace base {

static const std::string WEIGHT_QUANT_TPYE = "w8a8sc";
static const std::string WEIGHT_W4A16_QUANT_TPYE = "w4a16";

template <typename NormType>
DecoderLayer<NormType>::DecoderLayer(const LayerParam &param)
{
    this->param = param;
    this->param.CheckParam();
    this->inTensorCandidates = {
        {"input_norm_weight", {
            // shape: [hiddenSize]
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias"}},
        {"post_attn_norm_weight", {
            // shape: [hiddenSize]
            "in_post_attn_norm_weight", "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
            "in_post_attn_norm_new_bias"}},
        {"attn_weight", this->attnWeight},
        {"mlp_weight", this->mlpWeight},
        {"kv_quant_scale", {
            "in_k_quant_scale", "in_k_dequant_scale", "in_v_quant_scale", "in_v_dequant_scale"}},
        {"kv_quant_offset", {
            "in_k_quant_offset", "in_k_dequant_offset", "in_v_quant_offset", "in_v_dequant_offset"}},
        {"fa3_quant", {
            "in_q_quant_scale", "in_k_quant_scale", "in_v_quant_scale", "in_qk_descale",
            "q_offset", "kv_offset", "fa3_v_quant_scale", "fa3_offset"}},
        {"reduce_quant_attn", {
            "in_attn_reduce_quant_scale", "in_attn_reduce_quant_offset",
            "in_attn_gather_quant_scale", "in_attn_gather_quant_offset"}},
        {"reduce_quant_mlp", {
            "in_mlp_reduce_quant_scale", "in_mlp_reduce_quant_offset",
            "in_mlp_gather_quant_scale", "in_mlp_gather_quant_offset"}},
        {"default", {
            "in_hidden_states",  // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
            "in_cos_embedding", "in_sin_embedding", "in_attention_mask", "in_k_cache", "in_v_cache", "in_seq_len",
            "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"}},
        {"compress_head_alibi", {"wins_global", "in_ra_seqlens"}},
        {"compress_head_rope", {"wins_global", "in_ra_seqlens", "pffset_index", "razor_offset",
            "in_reshape_seqlen"}},  // [batchSize * Numhead]
        {"q_len", {"in_q_len"}},
        {"lora_common", {"in_seq_len_cum_sum"}},
        {"lora_attn", {
            "in_qkv_lora_a_0", "in_qkv_lora_b_0", "in_qkv_lora_a_1", "in_qkv_lora_b_1",
            "in_qkv_lora_a_2", "in_qkv_lora_b_2", "in_qkv_dense_lora_a", "in_qkv_dense_lora_b"}
        },
        {"lora_mlp", {
            "in_mlp_lora_a_0", "in_mlp_lora_b_0", "in_mlp_lora_a_1", "in_mlp_lora_b_1",
            "in_mlp_down_lora_a", "in_mlp_down_lora_b"}
        }
    };

    this->internalTensorCandidates = {{"default", {"intermediate_attn_out", "intermediate_mlp_out"}}};
}

template <typename NormType>
void DecoderLayer<NormType>::ConstructInTensorMap()
{
    this->inTensorList.clear();
    // 添加默认的Tensor
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "mlp_weight", this->inTensorList);

    // 添加KV cache int8特性的Tensor
    if (param.enableKvQuant) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "kv_quant_scale", this->inTensorList);
        if (param.kvQuantHasOffset) {
            atb_speed::common::AddTensorToList(this->inTensorCandidates, "kv_quant_offset", this->inTensorList);
        }
    }

    // 添加FA3特性的Tensor
    if (param.enableFA3) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "fa3_quant", this->inTensorList);
    }

    // 添加lccl reduce int8特性的Tensor
    if (param.enableReduceQuant) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "reduce_quant_attn", this->inTensorList);
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "reduce_quant_mlp", this->inTensorList);
    }

    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    atb_speed::common::AddTensorToList(
        this->internalTensorCandidates, "default", this->intermediateTensorList);

    // 添加头压缩特性的Tensor
    if (param.enableCompressHead) {
        atb_speed::common::AddTensorToList(
            this->inTensorCandidates,
            param.positionEmbeddingType == PositionEmbeddingType::ALIBI ? "compress_head_alibi" : "compress_head_rope",
            this->inTensorList);
    }

    // 添加并行解码特性或SplitFuse的Tensor
    if (param.enableSpeculate || param.enableSplitFuse) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
    }

    // 添加lora特性的Tensor
    if (param.enableLora) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "lora_common", this->inTensorList);
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "lora_attn", this->inTensorList);
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "lora_mlp", this->inTensorList);
    }
}

template <typename NormType>
void DecoderLayer<NormType>::ConstructInternalTensorMap()
{
    this->intermediateTensorList.clear();
    atb_speed::common::AddTensorToList(
        this->internalTensorCandidates, "default", this->intermediateTensorList);
}

template <typename NormType>
int64_t DecoderLayer<NormType>::BuildGraph(atb::Operation **operation)
{
    this->graph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    this->ConstructInTensorMap();
    this->ConstructInternalTensorMap();
    this->graph.inTensorNum = this->inTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.inTensorNum " << this->graph.inTensorNum);
    this->graph.internalTensorNum = this->intermediateTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.internalTensorNum " << this->graph.internalTensorNum);
    this->graph.outTensorNum = this->outTensorList.size();
    ATB_SPEED_LOG_DEBUG("this->graph.outTensorNum " << this->graph.outTensorNum);
    this->tensorMap = atb_speed::common::GetTensorMap(
        this->inTensorList, this->outTensorList, this->intermediateTensorList);
    std::stringstream ss;
    // 添加layer层 map打印
    for (auto tensor = this->tensorMap.cbegin(); tensor != this->tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("layer map tensor:\n" << ss.str());
    CHECK_OPERATION_STATUS_RETURN(this->AddOperationToGraph());

    uint32_t inHiddenStatesIdx = atb_speed::common::GetTensorIdx(tensorMap, "in_hidden_states");
    this->graph.inferShapeFunc = [inHiddenStatesIdx](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(inHiddenStatesIdx);
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(this->graph, operation));
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status DecoderLayer<NormType>::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttention());
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
    CHECK_OPERATION_STATUS_RETURN(this->AddMlp());
    CHECK_OPERATION_STATUS_RETURN(this->AddMlpResidualAdd());
    return atb::NO_ERROR;
}

template <typename NormType>
void DecoderLayer<NormType>::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    this->SetFusionAttentionNormParam(fusionAttentionParam);
    this->SetFusionAttentionLinearParam(fusionAttentionParam);

    // rope param
    if (param.positionEmbeddingType == ROPE) {
        fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
        fusionAttentionParam.ropeParam.rotaryCoeff = 2;  // 2: 旋转系数
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    } else if (param.positionEmbeddingType == ALIBI) {
        fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::NO_ROTARY;
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    } else if (param.positionEmbeddingType == ABSOLUTE) {
        fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::NO_ROTARY;
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }

    // attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.enableSplitFuse = param.enableSplitFuse;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.attnBackend = param.attnBackend;
    // self attention
    this->SetFusionAttentionATBSelfAttentionParam(fusionAttentionParam);
    // paged attention
    this->SetFusionAttentionATBPagedAttentionParam(fusionAttentionParam);
    // aclnnIncreAttention
    this->SetFusionAttentionAclNNIncreAttentionParam(fusionAttentionParam);
    // dense
    fusionAttentionParam.denseQuantType = atb_speed::common::ConvertQuantTypeToPackType(param.weightQuantType);
}

template<>
void DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionNormParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = this->param.normEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = this->param.normEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
}

template<>
void DecoderLayer<atb::infer::LayerNormParam>::SetFusionAttentionNormParam(
    atb_speed::common::FusionAttentionParam<atb::infer::LayerNormParam> &fusionAttentionParam)
{
    const int32_t beginParamsAxis = param.isFA ? 2 : 1;
    atb::infer::LayerNormParam attenLayerNormParam;
    attenLayerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    attenLayerNormParam.normParam.epsilon = this->param.normEps;
    attenLayerNormParam.normParam.beginNormAxis = beginParamsAxis;
    attenLayerNormParam.normParam.beginParamsAxis = 1;
    fusionAttentionParam.normParamType = attenLayerNormParam;
    atb::infer::LayerNormParam attenLayerNormQuantParam;
    attenLayerNormQuantParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    attenLayerNormQuantParam.normParam.epsilon = this->param.normEps;
    attenLayerNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    attenLayerNormQuantParam.normParam.beginNormAxis = beginParamsAxis;
    attenLayerNormQuantParam.normParam.beginParamsAxis = 1;
    fusionAttentionParam.normQuantParamType = attenLayerNormQuantParam;
    fusionAttentionParam.normHasBias = true;
}

template <typename NormType>
void DecoderLayer<NormType>::SetFusionAttentionLinearParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    // QKV param
    fusionAttentionParam.isGroupedQueryAttention = \
        this->param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = this->param.isBF16;
    fusionAttentionParam.layerLinearQuantType = this->param.linearQuantType;
    fusionAttentionParam.layerLinearTransposeType = this->param.linearTransposeType;
    fusionAttentionParam.packQuantType = this->param.packQuantType.at(0);
    fusionAttentionParam.quantGroupSize = this->param.quantGroupSize;
    fusionAttentionParam.supportLora = this->param.enableLora;
    fusionAttentionParam.loraEnableGMM = this->param.loraEnableGMM;
    fusionAttentionParam.qkvHasBias = this->param.linearHasBias.at(QKV_HASBIAS);
    // dense
    fusionAttentionParam.selfAttnHasBias = this->param.linearHasBias.at(SELFATTENTION_HASBIAS);
    fusionAttentionParam.supportLcoc = this->param.enableLcoc;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = this->param.tensorParallelInfo;
    if (this->param.enableReduceQuant) {
        fusionAttentionParam.selfOutLinearTensorParallelInfo.quantType = \
            atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL;
        fusionAttentionParam.selfOutLinearTensorParallelInfo.outDataType = ACL_FLOAT16;
    }
}

template <typename NormType>
void DecoderLayer<NormType>::SetFusionAttentionATBSelfAttentionParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    fusionAttentionParam.selfAttentionParam.headNum = this->param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = this->param.numKeyValueHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(this->param.hiddenSizePerAttentionHead);
    if (this->param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = this->param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.isTriuMask = this->param.isPrefill ? 1 : 0;
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    if (this->param.attnBackend == atb_speed::common::OpBackend::ACLNN && param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
}

template <typename NormType>
void DecoderLayer<NormType>::SetFusionAttentionATBPagedAttentionParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    fusionAttentionParam.pageAttentionParam.headNum = this->param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = this->param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(this->param.hiddenSizePerAttentionHead);
    if (this->param.enableCompressHead) {
        if (this->param.positionEmbeddingType == PositionEmbeddingType::ROPE) {
            fusionAttentionParam.pageAttentionParam.compressType = atb::infer::PagedAttentionParam::CompressType:: \
                COMPRESS_TYPE_KVHEAD_ROPE;
            fusionAttentionParam.reshapeCacheParm.compressType = atb::infer::ReshapeAndCacheParam::CompressType:: \
                COMPRESS_TYPE_KVHEAD_ROPE;
        } else {
            fusionAttentionParam.pageAttentionParam.compressType = atb::infer::PagedAttentionParam::CompressType:: \
                COMPRESS_TYPE_KVHEAD;
            fusionAttentionParam.reshapeCacheParm.compressType = atb::infer::ReshapeAndCacheParam::CompressType:: \
                COMPRESS_TYPE_KVHEAD;
        }
    }
    if (this->param.enableSpeculate) {
        fusionAttentionParam.pageAttentionParam.calcType = \
            atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = \
            atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
    }
    if (this->param.enableSplitFuse) {
        fusionAttentionParam.pageAttentionParam.calcType = \
            atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = \
            atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    }
    if (this->param.enableKvQuant) {
        fusionAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
        fusionAttentionParam.pageAttentionParam.hasQuantOffset  = this->param.kvQuantHasOffset;
    }
    if (this->param.enableFA3) {
        fusionAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_QKV_ONLINE;
        if (this->param.isBF16) {
            fusionAttentionParam.pageAttentionParam.outDataType = ACL_BF16;
        } else {
            fusionAttentionParam.pageAttentionParam.outDataType = ACL_FLOAT16;
        }
    }
}

template <typename NormType>
void DecoderLayer<NormType>::SetFusionAttentionAclNNIncreAttentionParam(
    atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam)
{
    fusionAttentionParam.aclnnIncreAttentionParam.headNum = this->param.numAttentionHeadsPerRank;
    fusionAttentionParam.aclnnIncreAttentionParam.kvHeadNum = this->param.numKeyValueHeadsPerRank;
    fusionAttentionParam.aclnnIncreAttentionParam.headDim = this->param.hiddenSizePerAttentionHead;
    fusionAttentionParam.aclnnIncreAttentionParam.hasMask = true;
    fusionAttentionParam.aclnnIncreAttentionParam.isFA = this->param.isFA;
    fusionAttentionParam.aclnnIncreAttentionParam.hasKVQuant = this->param.enableKvQuant;
    if (this->param.enableKvQuant) {
        fusionAttentionParam.aclnnIncreAttentionParam.hasQuantOffset = this->param.kvQuantHasOffset;
    }
}

template <>
atb::Status DecoderLayer<atb::infer::RmsNormParam>::CreateFusionAttentionOperation(atb::Operation **op)
{
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    this->SetFusionAttentionParam(fusionAttentionParam);
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, op));
    return atb::NO_ERROR;
}

template <>
atb::Status DecoderLayer<atb::infer::LayerNormParam>::CreateFusionAttentionOperation(atb::Operation **op)
{
    atb_speed::common::FusionAttentionParam<atb::infer::LayerNormParam> fusionAttentionParam;
    this->SetFusionAttentionParam(fusionAttentionParam);
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, op));
    return atb::NO_ERROR;
}

template <typename NormType>
std::map<unsigned int, std::vector<std::string>> DecoderLayer<NormType>::GetAttentionIntensor()
{
    std::map<unsigned int, std::vector<std::string>> attnInTensor = {};
    attnInTensor[common::AttnInTensorCategory::ATTN_DEFAULT] = {
        "in_hidden_states", "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight",
        "in_input_norm_new_bias",
        "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0", "in_qkv_descale_0", "in_qkv_bias_0",
        "in_qkv_compress_idx_0", "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1",
        "in_qkv_descale_1", "in_qkv_bias_1", "in_qkv_compress_idx_1",
        "in_qkv_weight_2", "in_qkv_scale_2", "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2",
        "in_qkv_compress_idx_2", "in_cos_embedding", "in_sin_embedding", "in_seq_len", "in_k_cache", "in_v_cache",
        "in_attention_mask", "in_token_offset", "in_layer_id", "in_block_tables", "in_slots",
        "in_qkv_dense_weight", "in_qkv_dense_scale", "in_qkv_dense_offset", "in_qkv_dense_descale",
        "in_qkv_dense_bias", "in_qkv_dense_compress_idx"};
    if (this->param.enableCompressHead) {
        if (this->param.positionEmbeddingType == PositionEmbeddingType::ALIBI) {
            attnInTensor[common::AttnInTensorCategory::ATTN_COMPRESS_HEAD_ALIBI] = \
                this->inTensorCandidates["compress_head_alibi"];
        } else {
            attnInTensor[common::AttnInTensorCategory::ATTN_COMPRESS_HEAD_ROPE] = \
                this->inTensorCandidates["compress_head_rope"];
                }
    }
    if (this->param.enableSpeculate || param.enableSplitFuse) {
        attnInTensor[common::AttnInTensorCategory::ATTN_SPECULATE] = this->inTensorCandidates["q_len"];
    }
    if (this->param.enableKvQuant) {
        attnInTensor[common::AttnInTensorCategory::ATTN_KV_QUANT_SCALE] = this->inTensorCandidates["kv_quant_scale"];
        if (this->param.kvQuantHasOffset) {
            attnInTensor[common::AttnInTensorCategory::ATTN_KV_QUANT_OFFSET] = \
                this->inTensorCandidates["kv_quant_offset"];
        }
    }
    if (this->param.enableFA3) {
        attnInTensor[common::AttnInTensorCategory::ATTN_FA3] = this->inTensorCandidates["fa3_quant"];
    }
    if (this->param.enableLora) {
        attnInTensor[common::AttnInTensorCategory::ATTN_LORA] = {"in_seq_len_cum_sum"};
        for (std::string tensor : this->inTensorCandidates.at("lora_attn")) {
            attnInTensor[common::AttnInTensorCategory::ATTN_LORA].push_back(tensor);
        }
    }
    if (this->param.enableReduceQuant) {
        attnInTensor[common::AttnInTensorCategory::ATTN_REDUCE_QUANT] = {};
        for (std::string tensor : this->inTensorCandidates.at("reduce_quant_attn")) {
            attnInTensor[common::AttnInTensorCategory::ATTN_REDUCE_QUANT].push_back(tensor);
        }
    }
    return attnInTensor;
}

template <typename NormType>
atb::Status DecoderLayer<NormType>::AddFusionAttention()
{
    atb::Node attentionNode;
    CHECK_OPERATION_STATUS_RETURN(this->CreateFusionAttentionOperation(&attentionNode.operation));

    // 按指定顺序对输入tensor进行排序
    std::map<unsigned int, std::vector<std::string>> attnInTensor = this->GetAttentionIntensor();
    std::vector<std::string> attnInTensorNames = {};
    for (unsigned int i = 0; i < common::AttnInTensorCategory::ATTN_END; i++) {
        attnInTensorNames.insert(attnInTensorNames.end(), attnInTensor[i].begin(), attnInTensor[i].end());
    }

    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_attn_out"});

    this->graph.nodes.push_back(attentionNode);
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status DecoderLayer<NormType>::AddFusionAttentionResidualAdd()
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Node selfResidualAddNode;
    if (!param.enableAddNorm) {
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
        selfResidualAddNode.inTensorIds = \
            atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states", "intermediate_attn_out"});
        selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states"});
        this->graph.nodes.push_back(selfResidualAddNode);
    }
    return atb::NO_ERROR;
}

template <typename NormType>
void DecoderLayer<NormType>::SetMlpParam(atb_speed::common::MlpParam<NormType> &mlpParam)
{
    mlpParam.isBF16 = this->param.isBF16;
    mlpParam.layerLinearQuantType = this->param.linearQuantType;
    mlpParam.layerLinearTransposeType = this->param.linearTransposeType;
    mlpParam.packQuantType = this->param.packQuantType.at(1);
    mlpParam.quantGroupSize = this->param.quantGroupSize;
    mlpParam.isLite = this->param.isLite;
    // norm
    this->SetMlpNormParam(mlpParam);
    // gate up
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(this->param.packQuantType.at(1), false);
    mlpParam.gateUpHasBias = this->param.linearHasBias.at(atb_speed::base::GATEUP_HASBIAS);
    mlpParam.enableAddNorm = this->param.enableAddNorm;
    mlpParam.supportLora = this->param.enableLora;
    mlpParam.loraEnableGMM = this->param.loraEnableGMM;
    // down
    mlpParam.downLinearTensorParallelInfo = this->param.tensorParallelInfo;
    if (this->param.enableReduceQuant) {
        mlpParam.downLinearTensorParallelInfo.quantType = \
            atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL;
        mlpParam.downLinearTensorParallelInfo.outDataType = ACL_FLOAT16;
    }
    mlpParam.downHasBias = this->param.linearHasBias.at(atb_speed::base::DOWN_HASBIAS);
    mlpParam.supportLcoc = this->param.enableLcoc;
    if (this->param.enableSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    }
    mlpParam.downQuantType = atb_speed::common::ConvertQuantTypeToPackType(param.weightQuantType);
}

template <>
void DecoderLayer<atb::infer::RmsNormParam>::SetMlpNormParam(
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam)
{
    atb::infer::RmsNormParam mlpRmsNormParam;
    if (this->param.enableAddNorm) {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormParam.preNormParam.epsilon = this->param.normEps;
    } else {
        mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormParam.normParam.epsilon = this->param.normEps;
    }
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    if (this->param.enableAddNorm) {
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_PRENORM;
        mlpRmsNormQuantParam.preNormParam.epsilon = this->param.normEps;
        mlpRmsNormQuantParam.preNormParam.quantType = atb::infer::QUANT_INT8;
    } else {
        mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
        mlpRmsNormQuantParam.normParam.epsilon = this->param.normEps;
        mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    }
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
}

template <>
void DecoderLayer<atb::infer::LayerNormParam>::SetMlpNormParam(
    atb_speed::common::MlpParam<atb::infer::LayerNormParam> &mlpParam)
{
    const int32_t beginParamsAxis = param.isFA ? 2 : 1;
    atb::infer::LayerNormParam mlpLayerNormParam;
    mlpLayerNormParam.layerType = param.enableAddNorm ? \
        atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_PRENORM : \
        atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    mlpLayerNormParam.normParam.epsilon = param.normEps;
    mlpLayerNormParam.normParam.beginNormAxis = beginParamsAxis;
    mlpLayerNormParam.normParam.beginParamsAxis = 1;
    mlpParam.normParamType = mlpLayerNormParam;
    atb::infer::LayerNormParam mlpLayerNormQuantParam;
    mlpLayerNormQuantParam.layerType = param.enableAddNorm ? \
        atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_PRENORM : \
        atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    mlpLayerNormQuantParam.normParam.epsilon = param.normEps;
    mlpLayerNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpLayerNormQuantParam.normParam.beginNormAxis = beginParamsAxis;
    mlpLayerNormQuantParam.normParam.beginParamsAxis = 1;
    mlpParam.normQuantParamType = mlpLayerNormQuantParam;
    mlpParam.normHasBias = true;
}

template <>
atb::Status DecoderLayer<atb::infer::RmsNormParam>::CreateMlpOperation(atb::Operation **op)
{
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    this->SetMlpParam(mlpParam);
    if (param.enableSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, op));
    } else {
        CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, op));
    }
    return atb::NO_ERROR;
}

template <>
atb::Status DecoderLayer<atb::infer::LayerNormParam>::CreateMlpOperation(atb::Operation **op)
{
    atb_speed::common::MlpParam<atb::infer::LayerNormParam> mlpParam;
    this->SetMlpParam(mlpParam);
    if (param.enableSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, op));
    } else {
        CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, op));
    }
    return atb::NO_ERROR;
}

template <typename NormType>
std::map<unsigned int, std::vector<std::string>> DecoderLayer<NormType>::GetMlpIntensor()
{
    std::map<unsigned int, std::vector<std::string>> mlpInTensor = {};
    mlpInTensor[common::MlpInTensorCategory::MLP_DEFAULT] = {
        "in_hidden_states", "in_post_attn_norm_weight", "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
        "in_post_attn_norm_new_bias", "in_mlp_weight_0", "in_mlp_scale_0", "in_mlp_offset_0", "in_mlp_descale_0",
        "in_mlp_bias_0", "in_mlp_compress_idx_0", "in_mlp_weight_1", "in_mlp_scale_1", "in_mlp_offset_1",
        "in_mlp_descale_1", "in_mlp_bias_1", "in_mlp_compress_idx_1", "in_mlp_down_weight", "in_mlp_down_scale",
        "in_mlp_down_offset", "in_mlp_down_descale", "in_mlp_down_bias", "in_mlp_down_compress_idx"
    };
    if (param.enableAddNorm) {
        mlpInTensor[common::MlpInTensorCategory::MLP_ADD_NORM] = {"intermediate_attn_out"};
    }
    if (param.enableLora) {
        mlpInTensor[common::MlpInTensorCategory::MLP_LORA] = {"in_seq_len_cum_sum"};
        for (std::string tensor : this->inTensorCandidates.at("lora_mlp")) {
            mlpInTensor[common::MlpInTensorCategory::MLP_LORA].push_back(tensor);
        }
    }
    if (param.enableReduceQuant) {
        mlpInTensor[common::MlpInTensorCategory::MLP_REDUCE_QUANT] = {};
        for (std::string tensor : this->inTensorCandidates.at("reduce_quant_mlp")) {
            mlpInTensor[common::MlpInTensorCategory::MLP_REDUCE_QUANT].push_back(tensor);
        }
    }
    return mlpInTensor;
}

template <typename NormType>
atb::Status DecoderLayer<NormType>::AddMlp()
{
    atb::Node mlpParallelNode;
    CHECK_OPERATION_STATUS_RETURN(this->CreateMlpOperation(&mlpParallelNode.operation));

    // 按指定顺序对输入tensor进行排序
    std::map<unsigned int, std::vector<std::string>> mlpInTensor = this->GetMlpIntensor();
    std::vector<std::string> mlpInTensorNames = {};
    for (unsigned int i = 0; i < common::MlpInTensorCategory::MLP_END; i++) {
        mlpInTensorNames.insert(mlpInTensorNames.end(), mlpInTensor[i].begin(), mlpInTensor[i].end());
    }

    std::vector<std::string> mlpOutTensorName = {"intermediate_mlp_out"};
    if (param.enableAddNorm) {
        mlpOutTensorName.push_back("in_hidden_states");
    }

    mlpParallelNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, mlpInTensorNames);
    mlpParallelNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, mlpOutTensorName);

    this->graph.nodes.push_back(mlpParallelNode);
    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status DecoderLayer<NormType>::AddMlpResidualAdd()
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::Node mlpResidualAddNode;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    mlpResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(this->tensorMap, {"in_hidden_states", "intermediate_mlp_out"});
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"out"});
    this->graph.nodes.push_back(mlpResidualAddNode);
    return atb::NO_ERROR;
}

template class DecoderLayer<atb::infer::RmsNormParam>;
template class DecoderLayer<atb::infer::LayerNormParam>;
} // namespace base
} // namespace atb_speed