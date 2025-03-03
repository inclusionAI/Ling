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

#include "nlohmann/json.hpp"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"
#include "models/internlm2/20b/layer/decoder_layer.h"

namespace atb_speed {
namespace internlm2_20b {

static const uint64_t IN_TENSOR_COUNT = 85;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;

void SetQKVLinearParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.linearTransposeType;
    fusionAttentionParam.packQuantType = param.packQuantType.at(0);
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    fusionAttentionParam.supportLora = param.supportLora;
    fusionAttentionParam.useImMask = param.useImMask;
    fusionAttentionParam.splitWithStride = param.splitWithStride;
    fusionAttentionParam.enableNormQuantOp = false;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
    attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
    attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
}

void SetRopeParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    // rope param
    if (param.positionEmbeddingType == ROPE) {
        fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
        fusionAttentionParam.ropeParam.rotaryCoeff = 2;  // 2: 旋转系数
        fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    }
}

void SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    SetQKVLinearParam(fusionAttentionParam, param);
    SetRopeParam(fusionAttentionParam, param);
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    if (param.hiddenSizePerAttentionHead == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param hiddenSizePerAttentionHead is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    // dense
    fusionAttentionParam.selfOutLinearTensorParallelInfo = param.tensorParallelInfo;
    if (param.kvQuant) {
        fusionAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
        fusionAttentionParam.pageAttentionParam.hasQuantOffset  = true;
    }
}

int64_t AddFusionAttention(atb::Node &attentionNode, const DecoderLayerParam &param)
{
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    SetFusionAttentionParam(fusionAttentionParam, param);
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, &attentionNode.operation));
    attentionNode.inTensorIds = {
        IN_HIDDEN_STATES, IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_BIAS, IN_INPUT_NORM_NEW_WEIGHT, IN_INPUT_NORM_NEW_BIAS,
        IN_QKV_WEIGHT_0, IN_QKV_SCALE_0, IN_QKV_OFFSET_0, IN_QKV_DESCALE_0, IN_QKV_BIAS_0, IN_QKV_COMPRESS_IDX_0,
        IN_QKV_WEIGHT_1, IN_QKV_SCALE_1, IN_QKV_OFFSET_1, IN_QKV_DESCALE_1, IN_QKV_BIAS_1, IN_QKV_COMPRESS_IDX_1,
        IN_QKV_WEIGHT_2, IN_QKV_SCALE_2, IN_QKV_OFFSET_2, IN_QKV_DESCALE_2, IN_QKV_BIAS_2, IN_QKV_COMPRESS_IDX_2,
        IN_COS_TABLE, IN_SIN_TABLE, IN_SEQ_LEN, IN_K_CACHE, IN_V_CACHE,
        IN_ATTENTION_MASK, IN_TOKEN_OFFSET, IN_LAYER_ID, IN_BLOCK_TABLES, IN_SLOTS,
        IN_ATTENTION_OUT_WEIGHT, IN_ATTENTION_OUT_SCALE, IN_ATTENTION_OUT_OFFSET,
        IN_ATTENTION_OUT_DESCALE, IN_ATTENTION_OUT_BIAS, IN_ATTENTION_OUT_COMPRESS_IDX
    };
    if (param.kvQuant) {
        std::vector<uint32_t> kvQuantInTensorIds = {
            IN_K_QUANT_SCALE, IN_K_DEQUANT_SCALE, IN_V_QUANT_SCALE, IN_V_DEQUANT_SCALE,
            IN_K_QUANT_OFFSET, IN_K_DEQUANT_OFFSET, IN_V_QUANT_OFFSET, IN_V_DEQUANT_OFFSET
        };
        for (auto tensorIds : kvQuantInTensorIds) {
            attentionNode.inTensorIds.push_back(tensorIds);
        }
    }
    if (param.supportLora) {
        if (param.useImMask) {
            attentionNode.inTensorIds.push_back(IN_LORA_IM_MASK);
        }
        std::vector<uint32_t> loraInTensorIds = {
            IN_PLACE_HOLDER, IN_QKV_LORA_A_0, IN_QKV_LORA_B_0, IN_QKV_LORA_A_1, IN_QKV_LORA_B_1,
            IN_QKV_LORA_A_2, IN_QKV_LORA_B_2, IN_DENSE_LORA_A, IN_DENSE_LORA_B
        };
        for (auto tensorIds : loraInTensorIds) {
            attentionNode.inTensorIds.push_back(tensorIds);
        }
    }
    attentionNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};

    return atb::NO_ERROR;
}

void SetMlpParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam, const DecoderLayerParam &param)
{
    mlpParam.isBF16 = param.isBF16;
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.layerLinearTransposeType = param.linearTransposeType;
    mlpParam.packQuantType = param.packQuantType.at(1);
    mlpParam.quantGroupSize = 128;  // 128: w4a16 pre group量化时的group size
    mlpParam.enableNormQuantOp = false;
    // gate up
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(param.packQuantType.at(1), false);
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormQuantParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
    mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    mlpParam.supportLora = param.supportLora;
    mlpParam.useImMask = param.useImMask;
    // down
    mlpParam.downLinearTensorParallelInfo = param.tensorParallelInfo;
    mlpParam.supportLcoc = param.supportLcoc;
    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    }
}

int64_t AddMlp(atb::Node &mlpParallelNode, const DecoderLayerParam &param)
{
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    SetMlpParam(mlpParam, param);
    if (param.supportSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, &mlpParallelNode.operation));
    } else {
        CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, &mlpParallelNode.operation));
    }
    mlpParallelNode.inTensorIds = {
        INTERMEDIATE_RESIDUAL_ADD_OUT, IN_ATTENTION_NORM_WEIGHT, IN_ATTENTION_NORM_BIAS,
        IN_ATTENTION_NORM_NEW_WEIGHT, IN_ATTENTION_NORM_NEW_BIAS,
        IN_MLP_WEIGHT_0, IN_MLP_SCALE_0, IN_MLP_OFFSET_0, IN_MLP_DESCALE_0, IN_MLP_BIAS_0, IN_MLP_COMPRESS_IDX_0,
        IN_MLP_WEIGHT_1, IN_MLP_SCALE_1, IN_MLP_OFFSET_1, IN_MLP_DESCALE_1, IN_MLP_BIAS_1, IN_MLP_COMPRESS_IDX_1,
        IN_MLP_DOWN_WEIGHT, IN_MLP_DOWN_SCALE, IN_MLP_DOWN_OFFSET, IN_MLP_DOWN_DESCALE,
        IN_MLP_DOWN_BIAS, IN_MLP_DOWN_COMPRESS_IDX
    };
    if (param.supportLora) {
        if (param.useImMask) {
            mlpParallelNode.inTensorIds.push_back(IN_LORA_IM_MASK);
        }
        std::vector<uint32_t> loraInTensorIds = {
            IN_PLACE_HOLDER, IN_MLP_LORA_A_0, IN_MLP_LORA_B_0, IN_MLP_LORA_A_1,
            IN_MLP_LORA_B_1, IN_MLP_DOWN_LORA_A, IN_MLP_DOWN_LORA_B
        };
        for (auto tensorIds : loraInTensorIds) {
            mlpParallelNode.inTensorIds.push_back(tensorIds);
        }
    }
    mlpParallelNode.outTensorIds = {INTERMEDIATE_MLP_OUT};
    return atb::NO_ERROR;
}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";

    atb::Node attentionNode;
    atb::Node selfResidualAddNode;
    atb::Node mlpParallelNode;
    atb::Node mlpResidualAddNode;

    CHECK_OPERATION_STATUS_RETURN(AddFusionAttention(attentionNode, param));
    opGraph.nodes.push_back(attentionNode);

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_ATTENTION_OUT
    };
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_RESIDUAL_ADD_OUT};
    opGraph.nodes.push_back(selfResidualAddNode);

    CHECK_OPERATION_STATUS_RETURN(AddMlp(mlpParallelNode, param));
    opGraph.nodes.push_back(mlpParallelNode);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    mlpResidualAddNode.inTensorIds = {
        INTERMEDIATE_RESIDUAL_ADD_OUT,
        INTERMEDIATE_MLP_OUT
    };
    mlpResidualAddNode.outTensorIds = {OUT_DECODER_LAYER};
    opGraph.nodes.push_back(mlpResidualAddNode);

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

} // namespace internlm2_20b
} // namespace atb_speed