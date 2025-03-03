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
#include "encoder_layer.h"

#include "operations/fusion/mlp_gate_v2.h"
#include "operations/fusion/parallel_layer_v2.h"

namespace atb_speed {
namespace vlmo {
enum EncoderLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_ATTENTIONMASK,
    IN_PASTKEY,
    IN_PASTVALUE,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_HOLDER,
    IN_LAYERID,
    IN_GAMMA1,
    IN_GAMMA2,
    IN_NORMWEIGHT,
    IN_NORMBIASID,
    IN_QKVBAISID,
    IN_QKVMIXEDLINEARWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEBIASID,
    IN_NORM2TEXTWEIGHT,
    IN_NORM2TEXTBIAS,
    IN_NORM2IMAGEWEIGHT,
    IN_NORM2IMAGEBIAS,
    IN_MLPTEXTUPWEIGHT,
    IN_MLPTEXTDOWNWEIGHT,
    IN_MPLTEXTBIASUP,
    IN_MPLTEXTBIASDOWN,
    IN_MLPIMAGEUPWIGHT,
    IN_MLPIMAGEDOWNWIGHT,
    IN_MPLIMAGEBIASUP,
    IN_MPLIMAGEBIASDOWN,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_QKVMIXEDLINEAROUT,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_QKVTRANSROUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_GAMMA1_OUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SLICE_TEXT_OUT,
    INTERMIDATE_NORM2TEXT_OUT,
    INTERMIDATE_MLPTEXT_OUT,
    INTERMIDATE_GAMMA2_TEXT_OUT,
    INTERMIDATE_SELFRESIDUALADDTEXTOUT,
    INTERMIDATE_SLICE_IMAGE_OUT,
    INTERMIDATE_NORM2IMAGE_OUT,
    INTERMIDATE_MLPIMAGE_OUT,
    INTERMIDATE_GAMMA2_IMAGE_OUT,
    INTERMIDATE_SELFRESIDUALADDIMAGEOUT,
};

static const uint64_t IN_TENSOR_COUNT = 28;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 20;


int64_t AddinputNormNodeFunc(atb::Node &inputNormNode, const EncoderLayerParam &param)
{
    atb::infer::LayerNormParam layerNormParam;
    layerNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerNormParam.normParam.epsilon = param.layerNormEps;
    layerNormParam.normParam.beginNormAxis = 2; // 从第2维开始norm
    layerNormParam.normParam.beginParamsAxis = 0;
    CREATE_OPERATION(layerNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = { IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORMBIASID };
    inputNormNode.outTensorIds = { INTERMIDATE_INPUTNORMOUT };
    return atb::NO_ERROR;
}

int64_t AddqkvLinearNodeFunc(atb::Node &qkvLinearNode)
{
    atb::infer::LinearParam linearParam = { false, true, true };
    CREATE_OPERATION(linearParam, &qkvLinearNode.operation);
    qkvLinearNode.inTensorIds = { INTERMIDATE_INPUTNORMOUT, IN_QKVMIXEDLINEARWEIGHT, IN_QKVBAISID };
    qkvLinearNode.outTensorIds = { INTERMIDATE_QKVMIXEDLINEAROUT };
    return atb::NO_ERROR;
}

int64_t AddtransposeNodeFunc(atb::Node &transposeNode, const EncoderLayerParam &param)
{
    atb::infer::TransposeParam transParam;
    transParam.perm = { 2, 0, 1, 3, 4 };
    CREATE_OPERATION(transParam, &transposeNode.operation);
    transposeNode.inTensorIds = { INTERMIDATE_QKVMIXEDLINEAROUT };
    transposeNode.outTensorIds = { INTERMIDATE_QKVTRANSROUT };
    transposeNode.inTensorReshapeFuncs.resize(transposeNode.inTensorIds.size());
    if (param.headNum == 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    transposeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 5;                 // 新形状有5个维度
        newShape.dims[0] = oldShape.dims[0]; // 第0维不变
        newShape.dims[1] = oldShape.dims[1]; // 第1维不变
        newShape.dims[2] = 3;             // 旧第2维分为三部分 [3, param.headNum, remian] 3是新的第2维值
        newShape.dims[3] = param.headNum; // 新的第3维
        newShape.dims[4] = oldShape.dims[2] / 3 / param.headNum; // 新的第4维 等于旧的第2维 / 3 / 多头数
    };
    return atb::NO_ERROR;
}

int64_t AddsplitQKVNodeFunc(atb::Node &splitQKVNode)
{
    atb::infer::SplitParam splitParam = { 0, 3, {} }; // 3是等分次数
    CREATE_OPERATION(splitParam, &splitQKVNode.operation);
    splitQKVNode.inTensorIds = { INTERMIDATE_QKVTRANSROUT };
    splitQKVNode.outTensorIds = { INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV };
    return atb::NO_ERROR;
}

int64_t AddselfAttentionNodeFunc(atb::Node &selfAttentionNode, const EncoderLayerParam &param)
{
    atb::infer::SelfAttentionParam selfAttentionParam;
    selfAttentionParam.headNum = param.headNum;
    if (param.dk == 0) {
        return atb::ERROR_INVALID_PARAM;
    }
    selfAttentionParam.qScale = 1.0f / std::sqrt(param.dk);
    selfAttentionParam.qkScale = 1.0f;
    selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_ALIBI;
    CREATE_OPERATION(selfAttentionParam, &selfAttentionNode.operation);
    selfAttentionNode.inTensorIds = { INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV,
        IN_PASTKEY,         IN_PASTVALUE,       IN_ATTENTIONMASK,
        IN_TOKENOFFSET,     IN_SEQLEN,          IN_LAYERID };
    selfAttentionNode.outTensorIds = { INTERMIDATE_SELFOUT };
    selfAttentionNode.inTensorReshapeFuncs.resize(selfAttentionNode.inTensorIds.size());
    selfAttentionNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 张量维度数为2
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2]; // 0 1 2 为张量维度形状下标
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4]; // 1 3 4 为张量维度形状下标
    };
    selfAttentionNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 张量维度数为2
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2]; // 0 1 2 为张量维度形状下标
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4]; // 1 3 4 为张量维度形状下标
    };
    selfAttentionNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;                                    // 张量维度数为2
        newShape.dims[0] = oldShape.dims[1] * oldShape.dims[2]; // 0 1 2 为张量维度形状下标
        newShape.dims[1] = oldShape.dims[3] * oldShape.dims[4]; // 1 3 4 为张量维度形状下标
    };
    return atb::NO_ERROR;
}

int64_t AddselfOutLinearNode(atb::Node &selfOutLinearNode, const EncoderLayerParam &param)
{
    atb_speed::common::ParallelParamV2 selfOutLinearParam;
    selfOutLinearParam.commParam.rank = param.rank;
    selfOutLinearParam.commParam.rankSize = param.rankSize;
    selfOutLinearParam.commParam.backend = param.backend;
    selfOutLinearParam.isBias = true;
    atb_speed::common::RowParallelLinearV2(selfOutLinearParam, &selfOutLinearNode.operation);
    selfOutLinearNode.inTensorIds = {
        INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEBIASID, IN_HOLDER, IN_HOLDER, IN_HOLDER, IN_HOLDER
    };
    selfOutLinearNode.outTensorIds = { INTERMIDATE_SELFLINEAROUT };
    return atb::NO_ERROR;
}

int64_t Addgama1MultNode(atb::Node &gama1MultNode)
{
    atb::infer::ElewiseParam gamma1MutmalParam;
    gamma1MutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma1MutmalParam, &gama1MultNode.operation);
    gama1MultNode.inTensorIds = { IN_GAMMA1, INTERMIDATE_SELFLINEAROUT };
    gama1MultNode.outTensorIds = { INTERMIDATE_GAMMA1_OUT };
    return atb::NO_ERROR;
}

int64_t AddselfResidualAddNode(atb::Node &selfResidualAddNode)
{
    atb::infer::ElewiseParam addGamma1Param;
    addGamma1Param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma1Param, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMIDATE_GAMMA1_OUT };
    selfResidualAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };
    return atb::NO_ERROR;
}

int64_t AddsliceTextNode(atb::Node &sliceTextNode, const EncoderLayerParam &param)
{
    atb::infer::SliceParam sliceTextParam;
    sliceTextParam.offsets = { 0, 0, 0 };
    sliceTextParam.size = { -1, param.maxTextLen, -1 };
    CREATE_OPERATION(sliceTextParam, &sliceTextNode.operation);
    sliceTextNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };
    sliceTextNode.outTensorIds = { INTERMIDATE_SLICE_TEXT_OUT };
    return atb::NO_ERROR;
}

int64_t AddnormalTextNode(atb::Node &normalTextNode, const EncoderLayerParam &param)
{
    atb::infer::LayerNormParam layerTextNormParam;

    layerTextNormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerTextNormParam.normParam.beginNormAxis = 2; // 从第2维开始norm
    layerTextNormParam.normParam.beginParamsAxis = 0;
    layerTextNormParam.normParam.epsilon = param.layerNormEps;
    CREATE_OPERATION(layerTextNormParam, &normalTextNode.operation);
    normalTextNode.inTensorIds = { INTERMIDATE_SLICE_TEXT_OUT, IN_NORM2TEXTWEIGHT, IN_NORM2TEXTBIAS };
    normalTextNode.outTensorIds = { INTERMIDATE_NORM2TEXT_OUT };
    return atb::NO_ERROR;
}

int64_t AddmlpTextNode(atb::Node &mlpTextNode, const EncoderLayerParam &param)
{
    atb_speed::common::MlpGateParamV2 mlpTextParam;
    mlpTextParam.commDownParam.rank = param.rank;
    mlpTextParam.commDownParam.rankSize = param.rankSize;
    mlpTextParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpTextParam.transposeB = true;
    mlpTextParam.isBias = true;
    mlpTextParam.noGate = true;
    mlpTextParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpTextParam, &mlpTextNode.operation);
    mlpTextNode.inTensorIds = { INTERMIDATE_NORM2TEXT_OUT,
        IN_MLPTEXTUPWEIGHT,
        IN_HOLDER,
        IN_MLPTEXTDOWNWEIGHT,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_MPLTEXTBIASUP,
        IN_HOLDER,
        IN_MPLTEXTBIASDOWN,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER };
    mlpTextNode.outTensorIds = { INTERMIDATE_MLPTEXT_OUT };
    return atb::NO_ERROR;
}

int64_t Addgama2MultTextNode(atb::Node &gama2MultTextNode)
{
    atb::infer::ElewiseParam gamma2TextMutmalParam;
    gamma2TextMutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma2TextMutmalParam, &gama2MultTextNode.operation);
    gama2MultTextNode.inTensorIds = { IN_GAMMA2, INTERMIDATE_MLPTEXT_OUT };
    gama2MultTextNode.outTensorIds = { INTERMIDATE_GAMMA2_TEXT_OUT };
    return atb::NO_ERROR;
}

int64_t AddselfResidualTextAddNode(atb::Node &selfResidualTextAddNode)
{
    atb::infer::ElewiseParam addGamma2TextParam;
    addGamma2TextParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma2TextParam, &selfResidualTextAddNode.operation);
    selfResidualTextAddNode.inTensorIds = { INTERMIDATE_SLICE_TEXT_OUT, INTERMIDATE_GAMMA2_TEXT_OUT };
    selfResidualTextAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDTEXTOUT };
    return atb::NO_ERROR;
}

int64_t AddsliceImageNode(atb::Node &sliceImageNode, const EncoderLayerParam &param)
{
    atb::infer::SliceParam sliceImageParam;
    sliceImageParam.offsets = { 0, param.maxTextLen, 0 };
    sliceImageParam.size = { -1, -1, -1 };
    CREATE_OPERATION(sliceImageParam, &sliceImageNode.operation);
    sliceImageNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDOUT };
    sliceImageNode.outTensorIds = { INTERMIDATE_SLICE_IMAGE_OUT };
    return atb::NO_ERROR;
}

int64_t AddnormalImageNode(atb::Node &normalImageNode, const EncoderLayerParam &param)
{
    atb::infer::LayerNormParam layerIMAGENormParam;
    layerIMAGENormParam.layerType = atb::infer::LayerNormParam::LayerNormType::LAYER_NORM_NORM;
    layerIMAGENormParam.normParam.beginNormAxis = 2; // 从第2维开始norm
    layerIMAGENormParam.normParam.beginParamsAxis = 0;
    layerIMAGENormParam.normParam.epsilon = param.layerNormEps;
    CREATE_OPERATION(layerIMAGENormParam, &normalImageNode.operation);
    normalImageNode.inTensorIds = { INTERMIDATE_SLICE_IMAGE_OUT, IN_NORM2IMAGEWEIGHT, IN_NORM2IMAGEBIAS };
    normalImageNode.outTensorIds = { INTERMIDATE_NORM2IMAGE_OUT };
    return atb::NO_ERROR;
}

int64_t AddmlpImageNode(atb::Node &mlpImageNode, const EncoderLayerParam &param)
{
    atb_speed::common::MlpGateParamV2 mlpIMAGEParam;
    mlpIMAGEParam.commDownParam.rank = param.rank;
    mlpIMAGEParam.commDownParam.rankSize = param.rankSize;
    mlpIMAGEParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    mlpIMAGEParam.transposeB = true;
    mlpIMAGEParam.isBias = true;
    mlpIMAGEParam.noGate = true;
    mlpIMAGEParam.isPack = false;
    atb_speed::common::MlpGateLayerV2(mlpIMAGEParam, &mlpImageNode.operation);
    mlpImageNode.inTensorIds = { INTERMIDATE_NORM2IMAGE_OUT,
        IN_MLPIMAGEUPWIGHT,
        IN_HOLDER,
        IN_MLPIMAGEDOWNWIGHT,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_MPLIMAGEBIASUP,
        IN_HOLDER,
        IN_MPLIMAGEBIASDOWN,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER,
        IN_HOLDER };
    mlpImageNode.outTensorIds = { INTERMIDATE_MLPIMAGE_OUT };
    return atb::NO_ERROR;
}

int64_t Addgama2MultImageNode(atb::Node &gama2MultImageNode)
{
    atb::infer::ElewiseParam gamma2IMAGEMutmalParam;
    gamma2IMAGEMutmalParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(gamma2IMAGEMutmalParam, &gama2MultImageNode.operation);
    gama2MultImageNode.inTensorIds = { IN_GAMMA2, INTERMIDATE_MLPIMAGE_OUT };
    gama2MultImageNode.outTensorIds = { INTERMIDATE_GAMMA2_IMAGE_OUT };
    return atb::NO_ERROR;
}

int64_t AddselfResidualImageAddNode(atb::Node &selfResidualImageAddNode)
{
    atb::infer::ElewiseParam addGamma2IMAGEParam;
    addGamma2IMAGEParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CREATE_OPERATION(addGamma2IMAGEParam, &selfResidualImageAddNode.operation);
    selfResidualImageAddNode.inTensorIds = { INTERMIDATE_SLICE_IMAGE_OUT, INTERMIDATE_GAMMA2_IMAGE_OUT };
    selfResidualImageAddNode.outTensorIds = { INTERMIDATE_SELFRESIDUALADDIMAGEOUT };
    return atb::NO_ERROR;
}

int64_t AddcatNode(atb::Node &catNode)
{
    atb::infer::ConcatParam catParam;
    catParam.concatDim = 1;
    CREATE_OPERATION(catParam, &catNode.operation);
    catNode.inTensorIds = { INTERMIDATE_SELFRESIDUALADDTEXTOUT, INTERMIDATE_SELFRESIDUALADDIMAGEOUT };
    catNode.outTensorIds = { OUT_LAYEROUT };
    return atb::NO_ERROR;
}

int64_t AddGroup1(atb::GraphParam &opGraph, const EncoderLayerParam &param)
{
    atb::Node inputNormNode;
    atb::Node qkvLinearNode;
    atb::Node transposeNode;
    atb::Node splitQKVNode;
    atb::Node selfAttentionNode;
    atb::Node selfOutLinearNode;
    atb::Node gama1MultNode;
    atb::Node selfResidualAddNode;
    atb::Node sliceTextNode;
    atb::Node normalTextNode;
    atb::Node mlpTextNode;
    CHECK_OPERATION_STATUS_RETURN(AddinputNormNodeFunc(inputNormNode, param));
    opGraph.nodes.push_back(inputNormNode);
    CHECK_OPERATION_STATUS_RETURN(AddqkvLinearNodeFunc(qkvLinearNode));
    opGraph.nodes.push_back(qkvLinearNode);
    CHECK_OPERATION_STATUS_RETURN(AddtransposeNodeFunc(transposeNode, param));
    opGraph.nodes.push_back(transposeNode);
    CHECK_OPERATION_STATUS_RETURN(AddsplitQKVNodeFunc(splitQKVNode));
    opGraph.nodes.push_back(splitQKVNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfAttentionNodeFunc(selfAttentionNode, param));
    opGraph.nodes.push_back(selfAttentionNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfOutLinearNode(selfOutLinearNode, param));
    opGraph.nodes.push_back(selfOutLinearNode);
    CHECK_OPERATION_STATUS_RETURN(Addgama1MultNode(gama1MultNode));
    opGraph.nodes.push_back(gama1MultNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfResidualAddNode(selfResidualAddNode));
    opGraph.nodes.push_back(selfResidualAddNode);
    CHECK_OPERATION_STATUS_RETURN(AddsliceTextNode(sliceTextNode, param));
    opGraph.nodes.push_back(sliceTextNode);
    CHECK_OPERATION_STATUS_RETURN(AddnormalTextNode(normalTextNode, param));
    opGraph.nodes.push_back(normalTextNode);
    CHECK_OPERATION_STATUS_RETURN(AddmlpTextNode(mlpTextNode, param));
    opGraph.nodes.push_back(mlpTextNode);
    return atb::NO_ERROR;
}

int64_t AddGroup2(atb::GraphParam &opGraph, const EncoderLayerParam &param)
{
    atb::Node gama2MultTextNode;
    atb::Node selfResidualTextAddNode;
    atb::Node sliceImageNode;
    atb::Node normalImageNode;
    atb::Node mlpImageNode;
    atb::Node gama2MultImageNode;
    atb::Node selfResidualImageAddNode;
    atb::Node catNode;

    CHECK_OPERATION_STATUS_RETURN(Addgama2MultTextNode(gama2MultTextNode));
    opGraph.nodes.push_back(gama2MultTextNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfResidualTextAddNode(selfResidualTextAddNode));
    opGraph.nodes.push_back(selfResidualTextAddNode);
    CHECK_OPERATION_STATUS_RETURN(AddsliceImageNode(sliceImageNode, param));
    opGraph.nodes.push_back(sliceImageNode);
    CHECK_OPERATION_STATUS_RETURN(AddnormalImageNode(normalImageNode, param));
    opGraph.nodes.push_back(normalImageNode);
    CHECK_OPERATION_STATUS_RETURN(AddmlpImageNode(mlpImageNode, param));
    opGraph.nodes.push_back(mlpImageNode);
    CHECK_OPERATION_STATUS_RETURN(Addgama2MultImageNode(gama2MultImageNode));
    opGraph.nodes.push_back(gama2MultImageNode);
    CHECK_OPERATION_STATUS_RETURN(AddselfResidualImageAddNode(selfResidualImageAddNode));
    opGraph.nodes.push_back(selfResidualImageAddNode);
    CHECK_OPERATION_STATUS_RETURN(AddcatNode(catNode));
    opGraph.nodes.push_back(catNode);
    return atb::NO_ERROR;
}

atb::Status EncoderLayer(const EncoderLayerParam &param, atb::Operation **operation)
{
    ATB_SPEED_LOG_DEBUG(__func__ << " called, headNum: " << param.headNum);
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = GetFuncNameAndNameSpace(__PRETTY_FUNCTION__);
    CHECK_OPERATION_STATUS_RETURN(AddGroup1(opGraph, param));
    CHECK_OPERATION_STATUS_RETURN(AddGroup2(opGraph, param));

    opGraph.inferShapeFunc = [](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

EncoderLayer::EncoderLayer() = default;

EncoderLayer::~EncoderLayer() = default;

void EncoderLayer::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TOKENOFFSET).hostData = tokenOffset_.data();
    variantPack.inTensors.at(IN_SEQLEN).hostData = seqLen_.data();
}
} // namespace vlmo
} // namespace atb_speed
