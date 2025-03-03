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

#include <limits>
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/infer_shape_functions.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/attention/qkv_linear_split.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetQKVInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qkvInTensorCandidates = {
        {"default", {
            "in_qkv_input", "in_qkv_norm_weight", "in_qkv_norm_bias", "in_qkv_norm_new_weight",
            "in_qkv_norm_new_bias",
            "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0", "in_qkv_descale_0", "in_qkv_bias_0",
            "in_qkv_compress_idx_0",
            "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1", "in_qkv_descale_1", "in_qkv_bias_1",
            "in_qkv_compress_idx_1",
            "in_qkv_weight_2", "in_qkv_scale_2", "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2",
            "in_qkv_compress_idx_2"}
        },
        {"lora", {
            "in_seq_len_cum_sum", "in_qkv_lora_a_0", "in_qkv_lora_b_0",
            "in_qkv_lora_a_1", "in_qkv_lora_b_1", "in_qkv_lora_a_2", "in_qkv_lora_b_2"}
        },
        {"lora_with_mask", {"in_im_mask"}}
    };
    return qkvInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetQKVIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qkvIntermediateTensorCandidates = {
        {"qkv_pack", {"intermediate_qkv"}},
    };
    return qkvIntermediateTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetQKVOutTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qkvOutTensorCandidates = {
        {"default", {"out_q", "out_k", "out_v"}},
    };
    return qkvOutTensorCandidates;
}

template <typename NormParamType>
std::map<std::string, uint32_t> ConstructQKVTensorMap(
    const FusionAttentionParam<NormParamType> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto qkvInTensorCandidates = GetQKVInTensorCandidates();
    auto qkvIntermediateTensorCandidates = GetQKVIntermediateTensorCandidates();
    auto qkvOutTensorCandidates = GetQKVOutTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {};

    bool isPack = CheckPack(param.packQuantType);

    // 添加默认的Tensor
    AddTensorToList(qkvInTensorCandidates, "default", inTensorList);
    if (isPack) {
        AddTensorToList(qkvIntermediateTensorCandidates, "qkv_pack", intermediateTensorList);
    }

    // 添加Lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(qkvInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(qkvInTensorCandidates, "lora", inTensorList);
    }

    // 添加outTensor
    AddTensorToList(qkvOutTensorCandidates, "default", outTensorList);

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

template <typename NormParamType>
atb::Status AddQNormLinearNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier, bool isPack)
{
    atb::Node qNormLinearNode;
    atb_speed::common::NormLinearParam<NormParamType> qNormLinearParam;
    qNormLinearParam.isAntiOutlier = isAntiOutlier;
    if (CheckParamVectorSize(param.layerLinearQuantType, Q_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearQuantType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (CheckParamVectorSize(param.layerLinearTransposeType, Q_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearTransposeType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    qNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[Q_LINEAR_INDEX], param.enableNormQuantOp);
    qNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    qNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
    qNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    qNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    qNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    qNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[0];
    qNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    qNormLinearParam.skipNorm = param.skipNorm;
    qNormLinearParam.normHasBias = param.normHasBias;
    qNormLinearParam.normParamType = param.normParamType;
    qNormLinearParam.normQuantParamType = param.normQuantParamType;
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(qNormLinearParam, &qNormLinearNode.operation));
    std::vector<std::string> qInTensor = {
        "in_qkv_input", "in_qkv_norm_weight", "in_qkv_norm_bias", "in_qkv_norm_new_weight",
        "in_qkv_norm_new_bias",
        "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0", "in_qkv_descale_0", "in_qkv_bias_0",
        "in_qkv_compress_idx_0"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            qInTensor.push_back("in_im_mask");
        }
        qInTensor.push_back("in_seq_len_cum_sum");
        qInTensor.push_back("in_qkv_lora_a_0");
        qInTensor.push_back("in_qkv_lora_b_0");
    }
    qNormLinearNode.inTensorIds = GetTensorIdxList(tensorMap, qInTensor);
    qNormLinearNode.outTensorIds = {GetTensorIdx(tensorMap, isPack ? "intermediate_qkv" : "out_q")};
    opGraph.nodes.push_back(qNormLinearNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSplitQKVNode(const FusionAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitQKVNode;
    atb::infer::SplitParam splitQKVParam;
    if (param.splitWithStride) {
        splitQKVParam = {2, 3, {param.selfAttentionParam.headNum / param.selfAttentionParam.kvHeadNum, 1, 1}};
    } else {
        splitQKVParam = {(param.isFA ? 2 : 1), 3, {
            CheckIntMulOverFlow(param.selfAttentionParam.headNum, param.headDim),
            CheckIntMulOverFlow(param.selfAttentionParam.kvHeadNum, param.headDim),
            CheckIntMulOverFlow(param.selfAttentionParam.kvHeadNum, param.headDim)}};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQKVParam, &splitQKVNode.operation));
    splitQKVNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv")};
    splitQKVNode.outTensorIds = {GetTensorIdxList(tensorMap, {"out_q", "out_k", "out_v"})};
    if (param.splitWithStride) {
        splitQKVNode.inTensorReshapeFuncs.resize(splitQKVNode.inTensorIds.size());
        splitQKVNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            InternlmV2QKVSplit(
                oldShape, newShape,
                param.selfAttentionParam.headNum, param.selfAttentionParam.kvHeadNum, param.headDim);
        };
    }
    opGraph.nodes.push_back(splitQKVNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddSplitMixedQKVNode(const FusionAttentionParam<NormParamType> &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitMixedQKVNode;
    atb::infer::SplitParam splitMixedQKVParam;
    if (param.splitWithStride) {
        splitMixedQKVParam = {-2, 3, {}};
    } else {
        splitMixedQKVParam = {-1, 3, {}};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitMixedQKVParam, &splitMixedQKVNode.operation));
    splitMixedQKVNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_qkv")};
    splitMixedQKVNode.outTensorIds = GetTensorIdxList(tensorMap, {"out_q", "out_k", "out_v"});
    if (param.splitWithStride) {
        splitMixedQKVNode.inTensorReshapeFuncs.resize(splitMixedQKVNode.inTensorIds.size());
        splitMixedQKVNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            size_t dim = 0;
            newShape.dims[dim++] = oldShape.dims[0];                 // PA ntokens | FA batch
            if (param.isFA) {
                newShape.dims[dim++] = oldShape.dims[1];             // FA seqlen
            }
            newShape.dims[dim++] = param.selfAttentionParam.headNum; // headNum
            newShape.dims[dim++] = 3;                                // 3 -> q, k, v
            newShape.dims[dim++] = param.headDim;                    // dk
            newShape.dimNum = dim;
        };
    }
    opGraph.nodes.push_back(splitMixedQKVNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddKNormLinearNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier)
{
    atb::Node kNormLinearNode;
    atb_speed::common::NormLinearParam<NormParamType> kNormLinearParam;
    kNormLinearParam.isAntiOutlier = isAntiOutlier;
    kNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[K_LINEAR_INDEX], param.enableNormQuantOp);
    kNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    kNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
    kNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    kNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    kNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    kNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[1];
    kNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    kNormLinearParam.skipNorm = param.skipNorm;
    kNormLinearParam.normHasBias = param.normHasBias;
    kNormLinearParam.normParamType = param.normParamType;
    kNormLinearParam.normQuantParamType = param.normQuantParamType;
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(kNormLinearParam, &kNormLinearNode.operation));
    std::vector<std::string> kInTensor = {
        "in_qkv_input", "in_qkv_norm_weight", "in_qkv_norm_bias", "in_qkv_norm_new_weight",
        "in_qkv_norm_new_bias",
        "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1", "in_qkv_descale_1", "in_qkv_bias_1",
        "in_qkv_compress_idx_1"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            kInTensor.push_back("in_im_mask");
        }
        kInTensor.push_back("in_seq_len_cum_sum");
        kInTensor.push_back("in_qkv_lora_a_1");
        kInTensor.push_back("in_qkv_lora_b_1");
    }
    kNormLinearNode.inTensorIds = GetTensorIdxList(tensorMap, kInTensor);
    kNormLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "out_k")};
    opGraph.nodes.push_back(kNormLinearNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddVNormLinearNode(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier)
{
    atb::Node vNormLinearNode;
    atb_speed::common::NormLinearParam<NormParamType> vNormLinearParam;
    vNormLinearParam.isAntiOutlier = isAntiOutlier;
    vNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[V_LINEAR_INDEX], param.enableNormQuantOp);
    vNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    vNormLinearParam.fusionLinearParam.hasBias = param.qkvHasBias;
    vNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    vNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    vNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    vNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[V_LINEAR_INDEX];
    vNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    vNormLinearParam.skipNorm = param.skipNorm;
    vNormLinearParam.normHasBias = param.normHasBias;
    vNormLinearParam.normParamType = param.normParamType;
    vNormLinearParam.normQuantParamType = param.normQuantParamType;
    NormLinear<NormParamType>(vNormLinearParam, &vNormLinearNode.operation);
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(vNormLinearParam, &vNormLinearNode.operation));
    std::vector<std::string> vInTensor = {
        "in_qkv_input", "in_qkv_norm_weight", "in_qkv_norm_bias", "in_qkv_norm_new_weight",
        "in_qkv_norm_new_bias",
        "in_qkv_weight_2", "in_qkv_scale_2", "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2",
        "in_qkv_compress_idx_2"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            vInTensor.push_back("in_im_mask");
        }
        vInTensor.push_back("in_seq_len_cum_sum");
        vInTensor.push_back("in_qkv_lora_a_2");
        vInTensor.push_back("in_qkv_lora_b_2");
    }
    vNormLinearNode.inTensorIds = GetTensorIdxList(tensorMap, vInTensor);
    vNormLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "out_v")};
    opGraph.nodes.push_back(vNormLinearNode);
    return atb::NO_ERROR;
}

template <typename NormParamType>
void QKVLinearSplitInferShapeFunc(const FusionAttentionParam<NormParamType> &param,
    atb::GraphParam &opGraph, uint32_t inQKVInputIdx)
{
    if (param.isFA) {
        opGraph.inferShapeFunc = [inQKVInputIdx, param]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0) = inTensorDescs.at(inQKVInputIdx);
            outTensorDescs.at(0).shape.dimNum = 4;  // 0, 4: shape为4维
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(inQKVInputIdx).shape.dims[0];  // 0, 0, 0: batch size
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(inQKVInputIdx).shape.dims[1];  // 0, 1, 1: seq len
            outTensorDescs.at(0).shape.dims[2] = param.selfAttentionParam.headNum;  // 0, 2: headNum
            outTensorDescs.at(0).shape.dims[3] = param.headDim;  // 0, 3: headDim

            outTensorDescs.at(1) = outTensorDescs.at(0);
            outTensorDescs.at(1).shape.dims[2] = param.selfAttentionParam.kvHeadNum;  // 0, 2: kvHeadNum

            outTensorDescs.at(2) = outTensorDescs.at(1);  // 2: 第2个输出tensor的描述和第1个输出tensor的描述一致
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [inQKVInputIdx, param]
                (const atb::SVector<atb::TensorDesc> &inTensorDescs, atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0) = inTensorDescs.at(inQKVInputIdx);
            outTensorDescs.at(0).shape.dimNum = 3;  // 0, 3: shape为3维
            outTensorDescs.at(0).shape.dims[0] = \
                inTensorDescs.at(inQKVInputIdx).shape.dims[0];  // 0, 0, 0: batch size * seq len
            outTensorDescs.at(0).shape.dims[1] = param.selfAttentionParam.headNum;  // 0, 1: headNum
            outTensorDescs.at(0).shape.dims[2] = param.headDim;  // 0, 2: headDim

            outTensorDescs.at(1) = outTensorDescs.at(0);
            outTensorDescs.at(1).shape.dims[1] = param.selfAttentionParam.kvHeadNum;  // 0, 1: kvHeadNum

            outTensorDescs.at(2) = outTensorDescs.at(1);  // 2: 第2个输出tensor的描述和第1个输出tensor的描述一致
            return atb::NO_ERROR;
        };
    }
}

template <typename NormParamType>
atb::Status QKVLinearSplit(const FusionAttentionParam<NormParamType> &param, atb::Operation **operation)
{
    bool isPack = CheckPack(param.packQuantType);
    bool isAntiOutlier = CheckAntiOutlier(param.packQuantType);

    atb::GraphParam opGraph;
    opGraph.name = isPack ? "QKVLinearSplitPack" : "QKVLinearSplitNoPack";
    std::map<std::string, uint32_t> tensorMap = ConstructQKVTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("qkv opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("qkv opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("qkv opGraph.internalTensorNum " << opGraph.internalTensorNum);

    CHECK_PARAM_GT(param.selfAttentionParam.kvHeadNum, 0);
    CHECK_PARAM_GT(param.selfAttentionParam.headNum, 0);
    CHECK_PARAM_GT(param.headDim, 0);
    CHECK_PARAM_LT(param.headDim, 576);  // 576: headDim上界

    CHECK_OPERATION_STATUS_RETURN(AddQNormLinearNode(param, opGraph, tensorMap, isAntiOutlier, isPack));

    if (isPack && param.isGroupedQueryAttention) {  // Split GQA
        CHECK_OPERATION_STATUS_RETURN(AddSplitQKVNode(param, opGraph, tensorMap));
    } else if (isPack && !param.isGroupedQueryAttention) {  // Split MHA
        CHECK_OPERATION_STATUS_RETURN(AddSplitMixedQKVNode(param, opGraph, tensorMap));
    } else {  // isPack: false
        if (CheckParamVectorSize(param.layerLinearQuantType, V_LINEAR_INDEX + 1) != atb::NO_ERROR) {
            ATB_SPEED_LOG_ERROR("The size of param.layerLinearQuantType is wrong, please check");
            return atb::ERROR_INVALID_PARAM;
        }
        if (CheckParamVectorSize(param.layerLinearTransposeType, V_LINEAR_INDEX + 1) != atb::NO_ERROR) {
            ATB_SPEED_LOG_ERROR("The size of param.layerLinearTransposeType is wrong, please check");
            return atb::ERROR_INVALID_PARAM;
        }
        CHECK_OPERATION_STATUS_RETURN(AddKNormLinearNode(param, opGraph, tensorMap, isAntiOutlier));
        CHECK_OPERATION_STATUS_RETURN(AddVNormLinearNode(param, opGraph, tensorMap, isAntiOutlier));
    }

    uint32_t inQKVInputIdx = GetTensorIdx(tensorMap, "in_qkv_input");
    QKVLinearSplitInferShapeFunc(param, opGraph, inQKVInputIdx);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

template atb::Status AddQNormLinearNode(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier, bool isPack);
template atb::Status AddQNormLinearNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier, bool isPack);

template atb::Status AddSplitQKVNode(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSplitQKVNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);

template atb::Status AddSplitMixedQKVNode(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);
template atb::Status AddSplitMixedQKVNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap);

template atb::Status AddKNormLinearNode(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier);
template atb::Status AddKNormLinearNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier);

template atb::Status AddVNormLinearNode(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier);
template atb::Status AddVNormLinearNode(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap, bool isAntiOutlier);

template std::map<std::string, uint32_t> ConstructQKVTensorMap(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);
template std::map<std::string, uint32_t> ConstructQKVTensorMap(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum);

template atb::Status QKVLinearSplit(
    const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::Operation **operation);
template atb::Status QKVLinearSplit(
    const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::Operation **operation);

template void QKVLinearSplitInferShapeFunc(const FusionAttentionParam<atb::infer::RmsNormParam> &param,
    atb::GraphParam &opGraph, uint32_t inQKVInputIdx);
template void QKVLinearSplitInferShapeFunc(const FusionAttentionParam<atb::infer::LayerNormParam> &param,
    atb::GraphParam &opGraph, uint32_t inQKVInputIdx);
} // namespace common
} // namespace atb_speed