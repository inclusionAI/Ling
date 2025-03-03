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
#include "encoder_vl_layer.h"

#include "operations/fusion/mlp_gate_v2.h"
#include "operations/fusion/parallel_layer_v2.h"

namespace atb_speed {
namespace vlmo {
enum EncoderLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_ATTENTIONMASK,
    IN_PASTKEY = 2,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_LAYERID,
    IN_GAMMA1 = 8,
    IN_GAMMA2,
    IN_NORMWEIGHT,
    IN_NORMBIASID,
    IN_QKVBAISID,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEBIASID,
    IN_NORM2VLWEIGHT,
    IN_NORM2VLBIAS,
    IN_MLPVLUPWEIGHT,
    IN_MLPVLDOWNWEIGHT,
    IN_MPLVLBIASUP,
    IN_MPLVLBIASDOWN,
    OUT_LAYEROUT = 22,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_QKVTRANSROUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_SELFLINEAROUT = 30,
    INTERMIDATE_GAMMA1_OUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_NORM2VL_OUT,
    INTERMIDATE_MLPVL_OUT,
    INTERMIDATE_GAMMA2_VL_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 22;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 13;

int64_t AddinputNormVlNode(atb::Node &inputNormNode, const EncoderVllayerParam &param)
{
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = 2; // 从第2维开始norm
    layerNormParam.normParam.beginParamsAxis = 0;
    ATB_SPEED_LOG_DEBUG("constructor inputNormVlNode param end");
    CREATE_OPERATION(layerNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIASID };
    inputNormNode.outTensorIds = { INTERMIDATE_INPUTNORMOUT };
    return atb::NO_ERROR;
}

int64_t AddqkvLinearVlNode(atb::Node &qkvLinearNode)
{
    atb::infer::LinearParam linearParam = { false, true, true };
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT, IN_QKVBAISID };
    qkvLinearNode.outTensorIds = { INTERMIDATE_QKVMIXEDLINEAROUT };
    return atb::NO_ERROR;
}

int64_t AddtransposeVlNode(atb::Node &transposeNode, const EncoderVllayerParam &param)
{
    if (param.headNum == 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    atb::infer::TransposeParam transParam;
    transParam.perm = { 2, 0, 1, 3, 4 };
    CREATE_OPERATION(transParam, &transposeNode.operation);
    transposeNode.inTensorIds = { INTERMIDATE_QKVMIXEDLINEAROUT };
    transposeNode.outTensorIds = { INTERMIDATE_QKVTRANSROUT };
    transposeNode.inTensorReshapeFuncs.resize(transposeNode.inTensorIds.size());
    ATB_SPEED_LOG_DEBUG("constructor transposeVl param end");
    transposeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 5;                 // 新形状有5个维度
        newShape.dims[0] = oldShape.dims[0]; // 第0维不变
        newShape.dims[1] = oldShape.dims[1]; // 第1维不变
        newShape.dims[2] = 3;             // 旧第2维分为三部分 [3, param.headNum, remian] 3是新的第2维值
        newShape.dims[3] = param.headNum; // 新的第3维 多头数
        newShape.dims[4] = oldShape.dims[2] / 3 / param.headNum; // 新的第4维 等于旧的第2维 / 3 / 多头数
    };
    ATB_SPEED_LOG_DEBUG("constructor transposeVlNode end");
    return atb::NO_ERROR;
}

int64_t AddsplitQKVVlNode(atb::Node &splitQKVNode)
{
    atb::infer::SplitParam splitParam = { 0, 3, {} }; // 3是等分次数
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = { INTERMIDATE_QKVTRANSROUT };
    splitQKVNode.outTensorIds = { INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV };
    return atb::NO_ERROR;
}

int64_t AddselfAttentionVlNode(atb::Node &selfAttentionNode, const EncoderVllayerParam &param)
{
    if (param.dk == 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    selfAttentionParam.qScale = 1.0f / std::sqrt(param.dk);
    selfAttentionParam.qkScale = 1.0f;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionNode.operation);
    selfAttentionNode.outTensorIds = { INTERMIDATE_SELFOUT };
    selfAttentionNode.inTensorIds = { INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV,
        IN_PASTKEY,         IN_PASTVALUE,       IN_ATTENTIONMASK,
        IN_TOKENOFFSET,     IN_SEQLEN,          IN_LAYERID };
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 张量维度数为2
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2]; // 0 1 2 为张量维度形状下标
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4]; // 1 3 4 为张量维度形状下标
    };
    selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 张量维度数为2
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2]; // 0 1 2 为张量维度形状下标
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4]; // 1 3 4 为张量维度形状下标
    };
    selfAttentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 张量维度数为2
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2]; // 0 1 2 为张量维度形状下标
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4]; // 1 3 4 为张量维度形状下标
    };
    return atb::NO_ERROR;
}

int64_t AddselfOutLinearVlNode(atb::Node &selfOutLinearNode, const EncoderVllayerParam &param)
{
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = true;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.outTensorIds = { INTERMIDATE_SELFLINEAROUT };
    selfOutLinearNode.inTensorIds = {
        INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEBIASID, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER
    };
    return atb::NO_ERROR;
}

int64_t Addgama1MultVlNode(atb::Node &gama1MultNode)
{
    atb::infer::ElewiseParam gamma1MutmalParam;
    gamma1MutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma1MutmalParam, &gama1MultNode.operation);
    gama1MultNode.inTensorIds = { IN_GAMMA1, INTERMIDATE_SELFLINEAROUT };
    gama1MultNode.outTensorIds = { INTERMIDATE_GAMMA1_OUT };
    return atb::NO_ERROR;
}

int64_t AddselfResidualAddVlNode(atb::Node &selfResidualAddNode)
{
    atb::infer::ElewiseParam addGamma1Param;
    addGamma1Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma1Param, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_GAMMA1_OUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };
    return atb::NO_ERROR;
}

int64_t AddnormalVlNode(atb::Node &normalVlNode, const EncoderVllayerParam &param)
{
    atb::infer::LayerNormParam layerTextNormParam;

    layerTextNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerTextNormParam.normParam.beginNormAxis = 2; // 从第2维开始norm
    layerTextNormParam.normParam.beginParamsAxis = 0;
    layerTextNormParam.normParam.epsilon = param.layerNormEps;
    CREATE_OPERATION(layerTextNormParam, &normalVlNode.operation);
    normalVlNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, IN_NORM2VLWEIGHT, IN_NORM2VLBIAS };
    normalVlNode.outTensorIds = { INTERMIDATE_NORM2VL_OUT };
    return atb::NO_ERROR;
}

int64_t AddmlpVlNode(atb::Node &mlpVlNode, const EncoderVllayerParam &param)
{
    atb_speed::common::MlpGateParamV2 mlpTextParam;
    mlpTextParam.commDownParam.rank = param.rank;
    mlpTextParam.commDownParam.rankSize = param.rankSize;
    mlpTextParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpTextParam.transposeB = true;
    mlpTextParam.isBias = true;
    mlpTextParam.noGate = true;
    mlpTextParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpTextParam, &mlpVlNode.operation);
    mlpVlNode.inTensorIds = { INTERMIDATE_NORM2VL_OUT,
        IN_MLPVLUPWEIGHT,
        IN_HOLDER,
        IN_MLPVLDOWNWEIGHT,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_MPLVLBIASUP,
        IN_HOLDER,
        IN_MPLVLBIASDOWN,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER };
    mlpVlNode.outTensorIds = { INTERMIDATE_MLPVL_OUT };
    return atb::NO_ERROR;
}

int64_t Addgama2MultVlNode(atb::Node &gama2MultVlNode)
{
    atb::infer::ElewiseParam gamma2TextMutmalParam;
    gamma2TextMutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma2TextMutmalParam, &gama2MultVlNode.operation);
    gama2MultVlNode.inTensorIds = { IN_GAMMA2, INTERMIDATE_MLPVL_OUT };
    gama2MultVlNode.outTensorIds = { INTERMIDATE_GAMMA2_VL_OUT };
    return atb::NO_ERROR;
}

int64_t AddselfResidualVlAddNode(atb::Node &selfResidualVlAddNode)
{
    atb::infer::ElewiseParam addGamma2TextParam;
    addGamma2TextParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma2TextParam, &selfResidualVlAddNode.operation);
    selfResidualVlAddNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_GAMMA2_VL_OUT };
    selfResidualVlAddNode.outTensorIds = { OUT_LAYEROUT };
    return atb::NO_ERROR;
}

int64_t AddVlGroup1(atb::GraphParam &opGraph, const EncoderVllayerParam &param)
{
    atb::Node inputNormNode;
    atb::Node qkvLinearNode;
    atb::Node transposeNode;
    atb::Node splitQKVNode;
    atb::Node selfAttentionNode;
    atb::Node selfOutLinearNode;
    CHECK_OPERATION_STATUS_RETURN(AddinputNormVlNode(inputNormNode, param));
    opGraph.nodes.push_back(inputNormNode);
    CHECK_OPERATION_STATUS_RETURN(AddqkvLinearVlNode(qkvLinearNode));
    opGraph.nodes.push_back(qkvLinearNode);
    CHECK_OPERATION_STATUS_RETURN(AddtransposeVlNode(transposeNode, param));
    opGraph.nodes.push_back(transposeNode);
    CHECK_OPERATION_STATUS_RETURN(AddsplitQKVVlNode(splitQKVNode));
    opGraph.nodes.push_back(splitQKVNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfAttentionVlNode(selfAttentionNode, param));
    opGraph.nodes.push_back(selfAttentionNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfOutLinearVlNode(selfOutLinearNode, param));
    opGraph.nodes.push_back(selfOutLinearNode);
    return atb::NO_ERROR;
}

int64_t AddVlGroup2(atb::GraphParam &opGraph, const EncoderVllayerParam &param)
{
    atb::Node gama1MultNode;
    atb::Node selfResidualAddNode;
    atb::Node normalVlNode;
    atb::Node mlpVlNode;
    atb::Node gama2MultVlNode;
    atb::Node selfResidualVlAddNode;
    CHECK_OPERATION_STATUS_RETURN(Addgama1MultVlNode(gama1MultNode));
    opGraph.nodes.push_back(gama1MultNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfResidualAddVlNode(selfResidualAddNode));
    opGraph.nodes.push_back(selfResidualAddNode);
    CHECK_OPERATION_STATUS_RETURN(AddnormalVlNode(normalVlNode, param));
    opGraph.nodes.push_back(normalVlNode);
    CHECK_OPERATION_STATUS_RETURN(AddmlpVlNode(mlpVlNode, param));
    opGraph.nodes.push_back(mlpVlNode);
    CHECK_OPERATION_STATUS_RETURN(Addgama2MultVlNode(gama2MultVlNode));
    opGraph.nodes.push_back(gama2MultVlNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfResidualVlAddNode(selfResidualVlAddNode));
    opGraph.nodes.push_back(selfResidualVlAddNode);
    return atb::NO_ERROR;
}

atb::Status EncoderVlLayer(const EncoderVllayerParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG(__func__ << " called, headNum: " << param.headNum);
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    CHECK_OPERATION_STATUS_RETURN(AddVlGroup1(opGraph, param));
    CHECK_OPERATION_STATUS_RETURN(AddVlGroup2(opGraph, param));

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

EncoderVlLayer::EncoderVlLayer() = default;

EncoderVlLayer::~EncoderVlLayer() = default;

void EncoderVlLayer::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}
} // namespace vlmo
} // namespace atb_speed
