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
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/qwen/layer/moe_decoder_layer.h"

namespace atb_speed {
namespace qwen {
static const uint64_t IN_TENSOR_COUNT = 84;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 8;

atb::Status SetopGraph(atb::GraphParam &opGraph, const MoeDecoderLayerParam &param)
{
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    return atb::NO_ERROR;
}

atb::Status SetFusionAttentionParam(
    const MoeDecoderLayerParam &param,
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam
)
{
    // QKV linear param
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.qkvHasBias = true;
    fusionAttentionParam.layerLinearQuantType = param.attnLinearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.attnLinearTransposeType;
    fusionAttentionParam.packQuantType = param.packQuantType.at(0);
    fusionAttentionParam.supportLcoc = param.supportLcoc;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;

    return atb::NO_ERROR;
}

atb::Status SetFusionAttentionNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    SetFusionAttentionParam(param, fusionAttentionParam);
    atb::Node &attentionNode = opGraph.nodes.at(nodeId++);
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2; // 2:设置新张量形状
    // self attention param
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    CHECK_PARAM_GT(param.hiddenSizePerAttentionHead, 0);
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ? \
            atb::infer::SelfAttentionParam::CalcType::ENCODER : atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    fusionAttentionParam.selfOutLinearTensorParallelInfo = {
        param.rank, param.worldSize, param.backend, param.rankTableFile
    };
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, &attentionNode.operation));
    attentionNode.inTensorIds = {
    IN_HIDDEN_STATES, IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_BIAS, IN_INPUT_NORM_NEW_WEIGHT, IN_INPUT_NORM_NEW_BIAS,
    IN_QKV_WEIGHT_0, IN_QKV_SCALE_0, IN_QKV_OFFSET_0, IN_QKV_DESCALE_0, IN_QKV_BIAS_0, IN_QKV_COMPRESS_IDX_0,
    IN_QKV_WEIGHT_1, IN_QKV_SCALE_1, IN_QKV_OFFSET_1, IN_QKV_DESCALE_1, IN_QKV_BIAS_1, IN_QKV_COMPRESS_IDX_1,
    IN_QKV_WEIGHT_2, IN_QKV_SCALE_2, IN_QKV_OFFSET_2, IN_QKV_DESCALE_2, IN_QKV_BIAS_2, IN_QKV_COMPRESS_IDX_2,
    IN_COS_TABLE, IN_SIN_TABLE, IN_SEQ_LEN, IN_K_CACHE, IN_V_CACHE, IN_ATTENTION_MASK, IN_TOKEN_OFFSET,
    IN_LAYER_ID, IN_BLOCK_TABLES, IN_SLOTS, IN_ATTENTION_OUT_WEIGHT, IN_ATTENTION_OUT_SCALE,
    IN_ATTENTION_OUT_OFFSET, IN_ATTENTION_OUT_DESCALE, IN_ATTENTION_OUT_BIAS, IN_ATTENTION_OUT_COMPRESS_IDX,
    };
    attentionNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};
    return atb::NO_ERROR;
}


atb::Status SetAttentionResidualAddNode(atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &selfResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {
        IN_HIDDEN_STATES,
        INTERMEDIATE_ATTENTION_OUT
    };
    selfResidualAddNode.outTensorIds = {INTERMEDIATE_ATTENTION_OUT};
    return atb::NO_ERROR;
}

atb::Status SetSelfNormNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &selfNormNode = opGraph.nodes.at(nodeId++);
    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOperation(selfNormParam, &selfNormNode.operation);
    if (selfNormNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("selfNormNode op is nullptr: ");
    }
    selfNormNode.inTensorIds = {INTERMEDIATE_ATTENTION_OUT, IN_SELFATTENTION_OUT_NORM_WEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFATTENTION_NORM_OUT};
    ATB_SPEED_LOG_DEBUG("create post rmsnorm");
    return atb::NO_ERROR;
}

atb::Status SetMoeNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &moeNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::SparseMoeParam sparseMoeParam;
    sparseMoeParam.transpose = param.transpose;
    sparseMoeParam.numOfExperts = param.numOfExperts;
    sparseMoeParam.num = param.numOfSelectedExperts;
    sparseMoeParam.isBF16 = param.isBF16;
    sparseMoeParam.expertParallelDegree = param.expertParallelDegree;
    sparseMoeParam.processLogits = "none";
    sparseMoeParam.supportSwiGLU = param.supportSwiGLU;
    sparseMoeParam.routingMethod = param.routingMethod;
    sparseMoeParam.moeLinearQuantType = param.moeLinearQuantType;
    sparseMoeParam.packQuantType = param.packQuantType.at(1);
    atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation);
    if (moeNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("SparseMoe op is nullptr: ");
    }
    moeNode.inTensorIds = {
        INTERMIDATE_SELFATTENTION_NORM_OUT,
        IN_BLOCK_SPARSE_MOE_GATE_WEIGHT,
        IN_BLOCK_SPARSE_MOE_GATE_BIAS,
        IN_BLOCK_SPARSE_MOE_GATE_DESCALE,
        IN_BLOCK_SPARSE_MOE_GATE_OFFSET,
        IN_BLOCK_SPARSE_MOE_GATE_SCALE,
        IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX,
        IN_MLP_GATEUP_WEIGHT_EXPERT,
        IN_MLP_GATEUP_BIAS_EXPERT,
        IN_MLP_GATEUP_DESCALE_EXPERT,
        IN_MLP_GATEUP_OFFSET_EXPERT,
        IN_MLP_GATEUP_SCALE_EXPERT,
        IN_MLP_GATEUP_COMPRESS_IDX_EXPERT,
        IN_MLP_DOWN_WEIGHT_EXPERT,
        IN_MLP_DOWN_BIAS_EXPERT,
        IN_MLP_DOWN_DESCALE_EXPERT,
        IN_MLP_DOWN_OFFSET_EXPERT,
        IN_MLP_DOWN_SCALE_EXPERT,
        IN_MLP_DOWN_COMPRESS_IDX_EXPERT,
        IN_EXPERT_ARRAY, IN_EXPERT_GROUP, IN_ONE_HOT, IN_ZERO_HOT};
    moeNode.outTensorIds = {INTERMIDATE_MOE_OUT};
    ATB_SPEED_LOG_DEBUG("Moe Dense calculation success");
    return atb::NO_ERROR;
}

atb::Status SetShareExpertNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &shareExpertNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::SharedExpertParam sharedMlpExpertParam;
    sharedMlpExpertParam.isBF16 = param.isBF16;
    sharedMlpExpertParam.transposeGateup = param.transpose;
    sharedMlpExpertParam.transposeDown = param.transpose;
    sharedMlpExpertParam.hasSharedExpertGate = param.hasSharedExpertGate;
    sharedMlpExpertParam.mlpLinearQuantType = param.mlpLinearQuantType;
    sharedMlpExpertParam.mlpLinearTransposeType = param.mlpLinearTransposeType;
    sharedMlpExpertParam.packQuantType = param.packQuantType.at(1);
    ATB_SPEED_LOG_DEBUG("sharedMlpExpertParam success");
    atb_speed::common::CreateSharedExpertOperation(
        sharedMlpExpertParam, &shareExpertNode.operation);
    shareExpertNode.inTensorIds = {
        INTERMIDATE_SELFATTENTION_NORM_OUT,
        IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT,
        IN_MLP_GATEUP_BIAS_SHARED_EXPERT,
        IN_MLP_GATEUP_DESCALE_SHARED_EXPERT,
        IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
        IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
        IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT,
        IN_MLP_DOWN_WEIGHT_SHARED_EXPERT,
        IN_MLP_DOWN_BIAS_SHARED_EXPERT,
        IN_MLP_DOWN_DESCALE_SHARED_EXPERT,
        IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
        IN_MLP_DOWN_SCALE_SHARED_EXPERT,
        IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT,
        IN_MLP_GATE_WEIGHT_SHARED_EXPERT,
        IN_MLP_GATE_BIAS_SHARED_EXPERT,
        IN_MLP_GATE_DESCALE_SHARED_EXPERT,
        IN_MLP_GATE_OFFSET_SHARED_EXPERT,
        IN_MLP_GATE_SCALE_SHARED_EXPERT,
        IN_MLP_GATE_COMPRESS_IDX_SHARED_EXPERT};
    shareExpertNode.outTensorIds = {INTERMEDIATE_SHARE_EXPERTS_OUT};
    ATB_SPEED_LOG_DEBUG("shared expert calculation success");
    return atb::NO_ERROR;
}

atb::Status SetShareAddSelectNode(atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &shareAddSelectNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOperation(addParam, &shareAddSelectNode.operation);
    shareAddSelectNode.inTensorIds = {INTERMEDIATE_SHARE_EXPERTS_OUT, INTERMIDATE_MOE_OUT};
    shareAddSelectNode.outTensorIds = {INTERMIDATE_MOE_OUT};
    ATB_SPEED_LOG_DEBUG("shared expert add success");
    
    return atb::NO_ERROR;
}

atb::Status SetAllReduceNode(const MoeDecoderLayerParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &moeAllReduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.worldSize;
    allReduceParam.backend = param.backend;
    allReduceParam.rankTableFile = param.rankTableFile;
    CreateOperation(allReduceParam, &moeAllReduceNode.operation);
    if (moeAllReduceNode.operation == nullptr) {
        ATB_SPEED_LOG_ERROR("moeAllReduceNode op is nullptr: ");
    }
    moeAllReduceNode.inTensorIds = {INTERMIDATE_MOE_OUT};
    moeAllReduceNode.outTensorIds = {INTERMEDIATE_MLP_OUT};
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

atb::Status SetMlpResidualAddNode(atb::GraphParam &opGraph, size_t &nodeId, atb::Operation **operation)
{
    atb::Node &mlpResidualAddNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMEDIATE_ATTENTION_OUT, INTERMEDIATE_MLP_OUT};
    mlpResidualAddNode.outTensorIds = {OUT_DECODER_LAYER};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    ATB_SPEED_LOG_DEBUG("decoder layer: residule create opgraph");

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status MoeDecoderLayer(const MoeDecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    size_t nodeId = 0;

    // set graph
    CHECK_OPERATION_STATUS_RETURN(SetopGraph(opGraph, param));
    // node0: attention
    CHECK_OPERATION_STATUS_RETURN(SetFusionAttentionNode(param, opGraph, nodeId));
    // node1: residual
    CHECK_OPERATION_STATUS_RETURN(SetAttentionResidualAddNode(opGraph, nodeId));
    // node2: norm
    CHECK_OPERATION_STATUS_RETURN(SetSelfNormNode(param, opGraph, nodeId));
    // node3: moe
    CHECK_OPERATION_STATUS_RETURN(SetMoeNode(param, opGraph, nodeId));
    // node4: shareExpert
    CHECK_OPERATION_STATUS_RETURN(SetShareExpertNode(param, opGraph, nodeId));
    // node5: shareExperts add moe
    CHECK_OPERATION_STATUS_RETURN(SetShareAddSelectNode(opGraph, nodeId));
    // node6: addreduce
    CHECK_OPERATION_STATUS_RETURN(SetAllReduceNode(param, opGraph, nodeId));
    // node7: residual
    CHECK_OPERATION_STATUS_RETURN(SetMlpResidualAddNode(opGraph, nodeId, operation));

    return atb::NO_ERROR;
}

MoeDecoderLayer::MoeDecoderLayer() {}
MoeDecoderLayer::~MoeDecoderLayer() {}
}  // namespace qwen
}  // namespace atb_speed