/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "decoder_layer.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace gptneox_20b {
static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 14;
static const uint64_t LAYER_NORM_AXIS_NUM = 1;

int64_t AddInputNormNode(atb::Node &inputLayerNormNode, const PALayerParam &param)
{
    // norm [n_tokens, hidden_size]
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_NUM;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_NUM;

    CREATE_OPERATION(layerNormParam, &inputLayerNormNode.operation);
    inputLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_INPUTLAYERNORMWEIGTH, IN_INPUTLAYERNORMBIAS};
    inputLayerNormNode.outTensorIds = {INTERMEDIATE_INPUTLAYERNORMOUT};
    return atb::NO_ERROR;
}

int64_t AddqkvLinearNodeNodePa(atb::Node &qkvLinearNode)
{
    // qkv [n_tokens, hidden_size] to [n_tokens, 3 * hidden_size]
    atb::infer::LinearParam linearParam;
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = {INTERMEDIATE_INPUTLAYERNORMOUT, IN_QKVWEIGHT, IN_QKVBIAS};
    qkvLinearNode.outTensorIds = {INTERMEDIATE_MIXEDQKVLINEAROUT};
    return atb::NO_ERROR;
}

int64_t AddPositionEmbeddingNode(atb::Node &positionEmbeddingNode, const PALayerParam &param)
{
    // rope [n_tokens, hidden_size] to 3 * [n_tokens, hidden_size]
    atb_speed::gptneox_20b::PositionEmbeddingPAParam positionEmbeddingPAParam;
    positionEmbeddingPAParam.headNum = param.headNum;
    positionEmbeddingPAParam.dk = param.dk;
    positionEmbeddingPAParam.rotaryPct = param.rotaryPct;
    CHECK_OPERATION_STATUS_RETURN(atb_speed::gptneox_20b::PositionEmbeddingPAOperation(
        positionEmbeddingPAParam, &positionEmbeddingNode.operation));
    positionEmbeddingNode.inTensorIds = {INTERMEDIATE_MIXEDQKVLINEAROUT, IN_COSEMBED, IN_SINEMBED, IN_INPUT_LENGTHS};
    positionEmbeddingNode.outTensorIds = {INTERMEDIATE_QUERYEMBED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE};
    return atb::NO_ERROR;
}

int64_t AddReshapeAndCacheNode(atb::Node &reshapeAndCacheNode)
{
    // self attention
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CREATE_OPERATION(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE, IN_CACHEK, IN_CACHEV, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {IN_CACHEK, IN_CACHEV};
    return atb::NO_ERROR;
}

int64_t AddMul0Node(atb::Node &mul0Node, const PALayerParam &param)
{
    atb::infer::ElewiseParam mul0Param;
    mul0Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    mul0Param.mulsParam.varAttr = param.qScale;
    CreateOperation(mul0Param, &mul0Node.operation);
    mul0Node.inTensorIds = {INTERMEDIATE_QUERYEMBED};
    mul0Node.outTensorIds = {INTERMEDIATE_QUERYEMBED_SCALED};
    mul0Node.inTensorReshapeFuncs.resize(mul0Node.inTensorIds.size());
    return atb::NO_ERROR;
}

int64_t AddAttentionNode(atb::Node &attentionNode, const PALayerParam &param)
{
    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = param.qkScale;
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
        faEnParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
        CREATE_OPERATION(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMEDIATE_QUERYEMBED_SCALED, INTERMEDIATE_KEYEMBED, INTERMEDIATE_VALUE,
                                     IN_ATTENTIONMASK, IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMEDIATE_SELFATTNOUT};
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = param.qkScale;
        paDeParam.kvHeadNum = param.headNum;
        CREATE_OPERATION(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMEDIATE_QUERYEMBED_SCALED, IN_CACHEK, IN_CACHEV, IN_BLOCK_TABLES,
                                     IN_INPUT_LENGTHS};
        attentionNode.outTensorIds = {INTERMEDIATE_SELFATTNOUT};
    }
    return atb::NO_ERROR;
}

int64_t AddSelfAttnLinearNode(atb::Node &selfAttnLinearNode)
{
    atb::infer::LinearParam linearParam;
    CREATE_OPERATION(linearParam, &selfAttnLinearNode.operation);
    selfAttnLinearNode.inTensorIds = {INTERMEDIATE_SELFATTNOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARBIAS};
    selfAttnLinearNode.outTensorIds = {INTERMEDIATE_SELFATTNLINEAROUT};
    selfAttnLinearNode.inTensorReshapeFuncs.resize(selfAttnLinearNode.inTensorIds.size());
    selfAttnLinearNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 2: dim num
        newShape.dims[0] = oldShape.dims[0];                    // 0: dim 0, n tokens
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 1 hidden size: old 1, head num , old 2 head size
    };
    return atb::NO_ERROR;
}

int64_t AddPostAttnLayerNormNode(atb::Node &postAttnLayerNormNode, const PALayerParam &param)
{
    // mlp
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = LAYER_NORM_AXIS_NUM;
    layerNormParam.normParam.beginParamsAxis = LAYER_NORM_AXIS_NUM;
    CREATE_OPERATION(layerNormParam, &postAttnLayerNormNode.operation);
    postAttnLayerNormNode.inTensorIds = {IN_HIDDENSTATES, IN_POSTATTNLAYERNORMWEIGHT, IN_POSTATTNLAYERNORMBIAS};
    postAttnLayerNormNode.outTensorIds = {INTERMEDIATE_POSTATTNLAYERNORMOUT};
    return atb::NO_ERROR;
}

int64_t AddFfnLinearNodePa(atb::Node &ffnLinearNode)
{
    atb::infer::LinearParam linearParam;
    CREATE_OPERATION(linearParam, &ffnLinearNode.operation);
    ffnLinearNode.inTensorIds = {INTERMEDIATE_POSTATTNLAYERNORMOUT, IN_FFNLINEARWEIGHT, IN_FFNLINEARBIAS};
    ffnLinearNode.outTensorIds = {INTERMEDIATE_FFNLINEAROUT};
    return atb::NO_ERROR;
}

int64_t AddFfnActNodePa(atb::Node &ffnActNode)
{
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    CREATE_OPERATION(activationParam, &ffnActNode.operation);
    ffnActNode.inTensorIds = {INTERMEDIATE_FFNLINEAROUT};
    ffnActNode.outTensorIds = {INTERMEDIATE_FFNACTOUT};
    return atb::NO_ERROR;
}

int64_t AddFfnOutLinearNode(atb::Node &ffnOutLinearNode)
{
    atb::infer::LinearParam linearParam;
    CREATE_OPERATION(linearParam, &ffnOutLinearNode.operation);
    ffnOutLinearNode.inTensorIds = {INTERMEDIATE_FFNACTOUT, IN_FFNOUTLINEARWEIGHT, IN_FFNOUTLINEARBIAS};
    ffnOutLinearNode.outTensorIds = {INTERMEDIATE_FFNOUTLINEAROUT};
    return atb::NO_ERROR;
}

int64_t AddFfnResidualAddNodePa(atb::Node &ffnResidualAddNode)
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &ffnResidualAddNode.operation);
    ffnResidualAddNode.inTensorIds = {INTERMEDIATE_SELFATTNLINEAROUT, INTERMEDIATE_FFNOUTLINEAROUT};
    ffnResidualAddNode.outTensorIds = {INTERMEDIATE_ATTNMLPADDOUT};
    return atb::NO_ERROR;
}

int64_t AddAllReduceNode(atb::Node &allReduceNode, const PALayerParam &param)
{
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.rank;
    allReduceParam.rankSize = param.rankSize;
    allReduceParam.backend = param.backend;
    CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = {INTERMEDIATE_ATTNMLPADDOUT};
    allReduceNode.outTensorIds = {INTERMEDIATE_ATTNMLP_ALLREDUCEOUT};
    return atb::NO_ERROR;
}

int64_t AddAttnResidualAddNodePa(atb::Node &attnResidualAddNode)
{
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addParam, &attnResidualAddNode.operation);
    attnResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMEDIATE_ATTNMLP_ALLREDUCEOUT};
    attnResidualAddNode.outTensorIds = {OUT_GPTNEOXLAYEROUT};
    return atb::NO_ERROR;
}

int64_t AddAttentionGroupNode(atb::GraphParam &opGraph, const PALayerParam &param)
{
    atb::Node attentionNode;
    atb::Node selfAttnLinearNode;

    CHECK_OPERATION_STATUS_RETURN(AddAttentionNode(attentionNode, param));
    opGraph.nodes.push_back(attentionNode);

    CHECK_OPERATION_STATUS_RETURN(AddSelfAttnLinearNode(selfAttnLinearNode));
    opGraph.nodes.push_back(selfAttnLinearNode);
    return atb::NO_ERROR;
}

int64_t AddMlpGroupNode(atb::GraphParam &opGraph, const PALayerParam &param)
{
    atb::Node postAttnLayerNormNode;
    atb::Node ffnLinearNode;
    atb::Node ffnActNode;
    atb::Node ffnOutLinearNode;

    atb::Node ffnResidualAddNode; // ffn add attention
    atb::Node allReduceNode;
    atb::Node attnResidualAddNode; // add hidden state

    CHECK_OPERATION_STATUS_RETURN(AddPostAttnLayerNormNode(postAttnLayerNormNode, param));
    opGraph.nodes.push_back(postAttnLayerNormNode);

    CHECK_OPERATION_STATUS_RETURN(AddFfnLinearNodePa(ffnLinearNode));
    opGraph.nodes.push_back(ffnLinearNode);

    CHECK_OPERATION_STATUS_RETURN(AddFfnActNodePa(ffnActNode));
    opGraph.nodes.push_back(ffnActNode);

    CHECK_OPERATION_STATUS_RETURN(AddFfnOutLinearNode(ffnOutLinearNode));
    opGraph.nodes.push_back(ffnOutLinearNode);

    CHECK_OPERATION_STATUS_RETURN(AddFfnResidualAddNodePa(ffnResidualAddNode));
    opGraph.nodes.push_back(ffnResidualAddNode);

    CHECK_OPERATION_STATUS_RETURN(AddAllReduceNode(allReduceNode, param));
    opGraph.nodes.push_back(allReduceNode);

    CHECK_OPERATION_STATUS_RETURN(AddAttnResidualAddNodePa(attnResidualAddNode));
    opGraph.nodes.push_back(attnResidualAddNode);

    return atb::NO_ERROR;
}

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG(__func__ << " called, headNum: " << param.headNum);
    atb::GraphParam opGraph;
    if (param.isPrefill) {
        opGraph.name = "Prefill_transformer_layer";
    } else {
        opGraph.name = "Decoder_transformer_layer";
    }
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;

    atb::Node inputLayerNormNode;
    atb::Node qkvLinearNode;
    atb::Node positionEmbeddingNode;
    atb::Node reshapeAndCacheNode;
    atb::Node mul0Node;
    atb::Node attentionNode;
    atb::Node selfAttnLinearNode;

    CHECK_OPERATION_STATUS_RETURN(AddInputNormNode(inputLayerNormNode, param));
    opGraph.nodes.push_back(inputLayerNormNode);

    CHECK_OPERATION_STATUS_RETURN(AddqkvLinearNodeNodePa(qkvLinearNode));
    opGraph.nodes.push_back(qkvLinearNode);

    CHECK_OPERATION_STATUS_RETURN(AddPositionEmbeddingNode(positionEmbeddingNode, param));
    opGraph.nodes.push_back(positionEmbeddingNode);

    CHECK_OPERATION_STATUS_RETURN(AddReshapeAndCacheNode(reshapeAndCacheNode));
    opGraph.nodes.push_back(reshapeAndCacheNode);

    CHECK_OPERATION_STATUS_RETURN(AddMul0Node(mul0Node, param));
    opGraph.nodes.push_back(mul0Node);

    CHECK_OPERATION_STATUS_RETURN(AddAttentionGroupNode(opGraph, param));
    CHECK_OPERATION_STATUS_RETURN(AddMlpGroupNode(opGraph, param));

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

enum PositionEmbeddingFusionTensorId : int {
    IN_MIXEDQKV = 0, // [seqLen, hiddenSize], half
    IN_COS_EMBED,    // [maxseqLen, rotaryNum], half
    IN_SIN_EMBED,    // [maxseqLen, rotaryNum], half
    IN_SEQLEN,
    OUT_Q_EMBED,
    OUT_K_EMBED,
    OUT_VALUE, // [seqlen, headNum, headDim], half

    INTERNAL_Q_SPLIT,
    INTERNAL_K_SPLIT,
    INTERNAL_Q_ROT,
    INTERNAL_Q_PASS,
    INTERNAL_K_ROT,
    INTERNAL_K_PASS,

    INTERNAL_Q_ROPE,
    INTERNAL_K_ROPE,
};
static const int64_t IN_TENSOR_COUNT_OP = 4;
static const int64_t OUT_TENSOR_COUNT_OP = 3;
static const int64_t INTERNAL_TENSOR_COUNT = 8;

static const int64_t OUT_TENSOR_DIM_NUM = 3;
static const int64_t QKV_SPLIT_DIM = 1;
static const int64_t QKV_SPLIT_NUM = 3;
static const int64_t Q_ROT_PASS_CAT_DIM = 2;

void MergeHeadFunc(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2]; // 2: hn * rotaryNum;
}

void SqueeezeDimOne(const atb::Dims &oldShape, atb::Dims &newShape)
{
    newShape.dimNum = oldShape.dimNum - 1;
    newShape.dims[0] = oldShape.dims[0];
    newShape.dims[1] = oldShape.dims[2]; // 2: second dim
    newShape.dims[2] = oldShape.dims[3]; // 2: second dim 3: third dim
}

int64_t AddSplitQkvNode(atb::Node &splitQkvNode, const PositionEmbeddingPAParam &param)
{
    // split mixedQKV to q k v
    // [sq, hn * 3 * hs] --> [sq, hn, 3*hs] --> 3 of [sq, hn, hs]
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = QKV_SPLIT_DIM;
    splitParam.splitNum = QKV_SPLIT_NUM;
    CREATE_OPERATION(splitParam, &splitQkvNode.operation);
    splitQkvNode.inTensorIds = { IN_MIXEDQKV };
    splitQkvNode.outTensorIds = { INTERNAL_Q_SPLIT, INTERNAL_K_SPLIT, OUT_VALUE };
    splitQkvNode.inTensorReshapeFuncs.resize(splitQkvNode.inTensorIds.size());
    splitQkvNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // 4: DimNum
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = QKV_SPLIT_NUM;
        newShape.dims[2] = param.headNum; // 2: headNum
        newShape.dims[3] = param.dk; // 3: dk
    };
    return atb::NO_ERROR;
}

int64_t AddQRotSliceNode(atb::Node &qRotSliceNode, const PositionEmbeddingPAParam &param)
{
    // [sq, hn, hs] --> [sq, hn, 0:rotaryNum]
    int64_t rotaryNum = static_cast<int64_t>(param.dk * param.rotaryPct);
    atb::SVector<int64_t> sliceOffsetRot = { 0, 0, 0 };
    atb::SVector<int64_t> sliceSizeRot = { -1, -1, rotaryNum };

    atb::infer::SliceParam sliceRotParam;
    sliceRotParam.offsets = sliceOffsetRot;
    sliceRotParam.size = sliceSizeRot;
    CREATE_OPERATION(sliceRotParam, &qRotSliceNode.operation);
    qRotSliceNode.inTensorIds = { INTERNAL_Q_SPLIT };
    qRotSliceNode.outTensorIds = { INTERNAL_Q_ROT };
    qRotSliceNode.inTensorReshapeFuncs.resize(qRotSliceNode.inTensorIds.size());
    qRotSliceNode.inTensorReshapeFuncs[0] = SqueeezeDimOne;
    return atb::NO_ERROR;
}

int64_t AddQPassSliceNode(atb::Node &qPassSliceNode, const PositionEmbeddingPAParam &param)
{
    // [sq, hn, hs] --> [sq, hn, rotaryNum:(rotaryNum + passNum)]
    int64_t rotaryNum = static_cast<int64_t>(param.dk * param.rotaryPct);
    int64_t passNum = param.dk - rotaryNum;
    atb::SVector<int64_t> sliceOffsetPass = { 0, 0, rotaryNum };
    atb::SVector<int64_t> sliceSizePass = { -1, -1, passNum };
    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetPass;
    slicePassParam.size = sliceSizePass;
    CREATE_OPERATION(slicePassParam, &qPassSliceNode.operation);
    qPassSliceNode.inTensorIds = { INTERNAL_Q_SPLIT };
    qPassSliceNode.outTensorIds = { INTERNAL_Q_PASS };
    qPassSliceNode.inTensorReshapeFuncs.resize(qPassSliceNode.inTensorIds.size());
    qPassSliceNode.inTensorReshapeFuncs[0] = SqueeezeDimOne;
    return atb::NO_ERROR;
}

int64_t AddKRotSliceNode(atb::Node &kRotSliceNode, const PositionEmbeddingPAParam &param)
{
    int64_t rotaryNum = static_cast<int64_t>(param.dk * param.rotaryPct);
    atb::SVector<int64_t> sliceOffsetRot = { 0, 0, 0 };
    atb::SVector<int64_t> sliceSizeRot = { -1, -1, rotaryNum };
    atb::infer::SliceParam sliceRotParam;
    sliceRotParam.offsets = sliceOffsetRot;
    sliceRotParam.size = sliceSizeRot;
    CREATE_OPERATION(sliceRotParam, &kRotSliceNode.operation);
    kRotSliceNode.inTensorIds = { INTERNAL_K_SPLIT };
    kRotSliceNode.outTensorIds = { INTERNAL_K_ROT };
    kRotSliceNode.inTensorReshapeFuncs.resize(kRotSliceNode.inTensorIds.size());
    kRotSliceNode.inTensorReshapeFuncs[0] = SqueeezeDimOne;
    return atb::NO_ERROR;
}

int64_t AddKPassSliceNode(atb::Node &kPassSliceNode, const PositionEmbeddingPAParam &param)
{
    int64_t rotaryNum = static_cast<int64_t>(param.dk * param.rotaryPct);
    int64_t passNum = param.dk - rotaryNum;
    atb::SVector<int64_t> sliceOffsetPass = { 0, 0, rotaryNum };
    atb::SVector<int64_t> sliceSizePass = { -1, -1, passNum };
    atb::infer::SliceParam slicePassParam;
    slicePassParam.offsets = sliceOffsetPass;
    slicePassParam.size = sliceSizePass;

    CREATE_OPERATION(slicePassParam, &kPassSliceNode.operation);
    kPassSliceNode.inTensorIds = { INTERNAL_K_SPLIT };
    kPassSliceNode.outTensorIds = { INTERNAL_K_PASS };
    kPassSliceNode.inTensorReshapeFuncs.resize(kPassSliceNode.inTensorIds.size());
    kPassSliceNode.inTensorReshapeFuncs[0] = SqueeezeDimOne;
    return atb::NO_ERROR;
}

int64_t AddRopeNodePa(atb::Node &ropeNode)
{
    // [sq, hn, rotaryNum] --> [sq, hn, rotaryNum]
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2; // 2 is rotary coeff
    CREATE_OPERATION(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = { INTERNAL_Q_ROT, INTERNAL_K_ROT, IN_COS_EMBED, IN_SIN_EMBED, IN_SEQLEN };
    ropeNode.outTensorIds = { INTERNAL_Q_ROPE, INTERNAL_K_ROPE };
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[0] = &MergeHeadFunc;
    ropeNode.inTensorReshapeFuncs[1] = &MergeHeadFunc;
    return atb::NO_ERROR;
}

int64_t AddQCatNode(atb::Node &qCatNode, const PositionEmbeddingPAParam &param)
{
    // [sq, hn * rotaryNum] -->  [sq, hn, (rotaryNum + passNum)]
    atb::ReshapeFunc splitHeadsFunc = [param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1] / param.headNum;
    };
    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = Q_ROT_PASS_CAT_DIM;
    CREATE_OPERATION(concatParam, &qCatNode.operation);
    qCatNode.inTensorIds = { INTERNAL_Q_ROPE, INTERNAL_Q_PASS };
    qCatNode.outTensorIds = { OUT_Q_EMBED };
    qCatNode.inTensorReshapeFuncs.resize(qCatNode.inTensorIds.size());
    qCatNode.inTensorReshapeFuncs[0] = splitHeadsFunc;
    return atb::NO_ERROR;
}

int64_t AddKCatNode(atb::Node &kCatNode, const PositionEmbeddingPAParam &param)
{
    atb::ReshapeFunc splitHeadsFunc = [param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = oldShape.dimNum + 1;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.headNum;
        newShape.dims[2] = oldShape.dims[1] / param.headNum;
    };
    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = Q_ROT_PASS_CAT_DIM;
    CREATE_OPERATION(concatParam, &kCatNode.operation);
    kCatNode.inTensorIds = { INTERNAL_K_ROPE, INTERNAL_K_PASS };
    kCatNode.outTensorIds = { OUT_K_EMBED };
    kCatNode.inTensorReshapeFuncs.resize(kCatNode.inTensorIds.size());
    kCatNode.inTensorReshapeFuncs[0] = splitHeadsFunc;
    return atb::NO_ERROR;
}

atb::Status PositionEmbeddingPAOperation(const PositionEmbeddingPAParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "positionEmbeddingPAOperation";
    opGraph.inTensorNum = IN_TENSOR_COUNT_OP;
    opGraph.outTensorNum = OUT_TENSOR_COUNT_OP;
    opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;

    atb::Node splitQkvNode;
    atb::Node qRotSliceNode;
    atb::Node qPassSliceNode;
    atb::Node kRotSliceNode;
    atb::Node kPassSliceNode;
    atb::Node ropeNode;
    atb::Node qCatNode;
    atb::Node kCatNode;

    if (param.headNum == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param headNum is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
    CHECK_OPERATION_STATUS_RETURN(AddSplitQkvNode(splitQkvNode, param));
    opGraph.nodes.push_back(splitQkvNode);
    CHECK_OPERATION_STATUS_RETURN(AddQRotSliceNode(qRotSliceNode, param));
    opGraph.nodes.push_back(qRotSliceNode);
    CHECK_OPERATION_STATUS_RETURN(AddQPassSliceNode(qPassSliceNode, param));
    opGraph.nodes.push_back(qPassSliceNode);
    CHECK_OPERATION_STATUS_RETURN(AddKRotSliceNode(kRotSliceNode, param));
    opGraph.nodes.push_back(kRotSliceNode);
    CHECK_OPERATION_STATUS_RETURN(AddKPassSliceNode(kPassSliceNode, param));
    opGraph.nodes.push_back(kPassSliceNode);
    CHECK_OPERATION_STATUS_RETURN(AddRopeNodePa(ropeNode));
    opGraph.nodes.push_back(ropeNode);
    CHECK_OPERATION_STATUS_RETURN(AddQCatNode(qCatNode, param));
    opGraph.nodes.push_back(qCatNode);
    CHECK_OPERATION_STATUS_RETURN(AddKCatNode(kCatNode, param));
    opGraph.nodes.push_back(kCatNode);

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = param.headNum;
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1] / param.headNum / 3; // 2:s 3:t
        outTensorDescs.at(1) = outTensorDescs.at(0);
        outTensorDescs.at(2) = outTensorDescs.at(0); // 2: second dim
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace gptneox_20b
} // namespace atb_speed
