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
#include <atb/atb_infer.h>
#include <cmath>
#include "atb_speed/log.h"
#include "operations/aclnn/ops/w8a16_operation.h"
#include "operations/aclnn/ops/w4a16_operation.h"
#include "operations/aclnn/ops/w8a8_operation.h"
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/linear/linear.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetLinearInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearInTensorCandidates = {
        {"default", {
            "in_input", "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"}
        },
        {"lora", {"in_group_list", "in_lora_a", "in_lora_b"}},
        {"lora_with_mask", {"in_im_mask"}}
    };
    return linearInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLinearIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearIntermediateTensorCandidates = {
        {"quant_input", {"intermediate_quant_input"}},
        {"lora", {"intermediate_base_linear_out", "intermediate_lora_a_out", "intermediate_lora_b_out"}},
        {"dynamic_quant", {"intermediate_input_scale"}},
        {"lora_with_mask", {"intermediate_im_mask_out"}},
    };
    return linearIntermediateTensorCandidates;
}

std::map<std::string, uint32_t> ConstructLinearTensorMap(
    const FusionLinearParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto linearInTensorCandidates = GetLinearInTensorCandidates();
    auto linearIntermediateTensorCandidates = GetLinearIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(linearInTensorCandidates, "default", inTensorList);

    // 添加额外的中间Tensor
    if (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT
        || param.quantType == LINEAR_W8A8_DYNAMIC_QUANT) {
        AddTensorToList(linearIntermediateTensorCandidates, "quant_input", intermediateTensorList);
    }

    // 添加动态量化中间Tensor
    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT) {
        AddTensorToList(linearIntermediateTensorCandidates, "dynamic_quant", intermediateTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddElewiseQuant(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    if (param.quantType == LINEAR_W8A8_QUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
        // quant
        atb::Node inputQuantNode;
        atb::infer::ElewiseParam inputQuantParam;
        inputQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(inputQuantParam, &inputQuantNode.operation));
        inputQuantNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_scale", "in_offset"});
        inputQuantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_quant_input")};
        opGraph.nodes.push_back(inputQuantNode);
    }
    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT) {
        atb::Node inputDynamicQuantNode;
        inputDynamicQuantNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input"});
        inputDynamicQuantNode.outTensorIds = GetTensorIdxList(tensorMap, {"intermediate_quant_input",
                                                                          "intermediate_input_scale"});
        inputDynamicQuantNode.operation = new atb_speed::common::DynamicQuantOperation("DynamicQuantNode");
        opGraph.nodes.push_back(inputDynamicQuantNode);
    }
    return atb::NO_ERROR;
}

int64_t AddAclNNWeightQuantBatchMatmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "in_input", "in_weight", "in_scale", "in_offset"
    });
    AclNNWeightQuantBatchMatmulParam aclnnParam;
    aclnnParam.transposeB = param.transposeType == TRANSPOSE;
    if (param.hasBias) {
        linearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        aclnnParam.hasBias = true;
    }
    if (param.quantType == W8A16) {
        aclnnParam.quantGroupSize = param.quantGroupSize;
        linearNode.operation = new atb_speed::common::W8A16Operation("W8A16LinearNode", aclnnParam);
    } else if (param.quantType == W4A16) {
        aclnnParam.quantGroupSize = param.quantGroupSize;  // W4A16 group size默认为64，此时精度更高
        linearNode.operation = new atb_speed::common::W4A16Operation("W4A16LinearNode", aclnnParam);
    }
    if (linearNode.operation == nullptr) {
        return atb::ERROR_INVALID_GRAPH;
    }
    return atb::NO_ERROR;
}

int64_t AddAclNNQuantMatmul(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "in_input", "in_weight", "in_scale", "in_offset"
    });
    AclNNQuantMatmulParam aclnnQuantMatmulParam;
    aclnnQuantMatmulParam.transposeB = param.transposeType == TRANSPOSE;
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "intermediate_quant_input", "in_weight", "in_scale", "intermediate_input_scale",
    });
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[3] = [=](const atb::Dims &oldShape, atb::Dims &newShape) { // 3: 3号scale
        newShape.dimNum = 1; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0];
    };
    linearNode.operation = new atb_speed::common::W8A8Operation("W8A8LinearNode", aclnnQuantMatmulParam);

    return atb::NO_ERROR;
}

int64_t AddAclNNLinear(atb::Node &linearNode, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    if (param.quantType == W8A16 || param.quantType == W4A16) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNWeightQuantBatchMatmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    if (param.quantType == LINEAR_W8A8_DYNAMIC_QUANT) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNQuantMatmul(linearNode, param, tensorMap));
        return atb::NO_ERROR;
    }
    return atb::NO_ERROR;
}

int64_t AddLinear(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node linearNode;
    atb::infer::LinearParam linearParam;
    linearParam.transposeB = param.transposeType == TRANSPOSE;
    if (param.quantType != NO_QUANT) {
        linearParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    }
    // 设置LinearNode outTensor
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    // 稀疏量化
    if (param.quantType == LINEAR_W8A8_SC_DEQUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
        atb::infer::LinearSparseParam linearSparseParam;
        linearSparseParam.tilingK = 8;  // 8: 稀疏量化系数
        linearSparseParam.tilingN = 8;  // 8: 稀疏量化稀疏
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearSparseParam, &linearNode.operation));
        linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
            param.quantType == LINEAR_W8A8_SC_DEQUANT ? "in_input" : "intermediate_quant_input",
            "in_weight", "in_bias", "in_descale", "in_compress_idx"
        });
        opGraph.nodes.push_back(linearNode);
        return atb::NO_ERROR;
    }
    // AclNN Linear (W8A16, W4A16, LINEAR_W8A8_DYNAMIC_QUANT)
    if (param.quantType == W8A16 || param.quantType == W4A16 || param.quantType == LINEAR_W8A8_DYNAMIC_QUANT) {
        CHECK_OPERATION_STATUS_RETURN(AddAclNNLinear(linearNode, param, tensorMap));
        opGraph.nodes.push_back(linearNode);
        return atb::NO_ERROR;
    }
    // 加速库Linear
    if (param.quantType == NO_QUANT && param.hasBias) {
        linearParam.hasBias = true;
        linearNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_weight", "in_bias"});
    } else if (param.quantType == NO_QUANT && !param.hasBias) {
        linearParam.hasBias = false;
        linearNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_weight"});
    } else {
        linearParam.hasBias = true;
        linearNode.inTensorIds = GetTensorIdxList(tensorMap, {
            param.quantType == LINEAR_W8A8_DEQUANT ? "in_input" : "intermediate_quant_input",
            "in_weight", "in_bias", "in_descale"
        });
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParam, &linearNode.operation));
    opGraph.nodes.push_back(linearNode);

    return atb::NO_ERROR;
}

atb::Status CreateFusionLinear(const FusionLinearParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.quantType == NO_QUANT ? "LinearNoQuant" : \
        param.quantType == LINEAR_W8A8_DEQUANT || param.quantType == LINEAR_W8A8_SC_DEQUANT ? "LinearDequantOnly" : \
        param.quantType == W8A16 ? "LinearW8A16" : \
        param.quantType == W4A16 ? "LinearW4A16" : "LinearQuant";
    std::map<std::string, uint32_t> tensorMap = ConstructLinearTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    if (param.transposeType == TRANSPOSE_INVALID) {
        ATB_SPEED_LOG_ERROR("param.transposeType is invalid");
        return atb::ERROR_INVALID_GRAPH;
    }

    CHECK_OPERATION_STATUS_RETURN(AddElewiseQuant(opGraph, param, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(AddLinear(opGraph, param, tensorMap));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        uint32_t inputIdx = GetTensorIdx(tensorMap, "in_input");
        uint32_t weightIdx = GetTensorIdx(tensorMap, "in_weight");
        uint32_t biasIdx = GetTensorIdx(tensorMap, "in_bias");
        outTensorDescs.at(0).format = inTensorDescs.at(inputIdx).format;
        outTensorDescs.at(0).dtype = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
        outTensorDescs.at(0).shape = inTensorDescs.at(inputIdx).shape;
        auto outDimSize = outTensorDescs.at(inputIdx).shape.dimNum;
        CHECK_TENSORDESC_DIMNUM_VALID(outDimSize);
        int nDim = param.transposeType == TransposeType::TRANSPOSE ? 0 : 1;

        if (param.quantType == LINEAR_W8A8_SC_DEQUANT || param.quantType == LINEAR_W8A8_SC_QUANT) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(biasIdx).shape.dims[0];
        } else if (param.quantType == W4A16) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = \
                CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[nDim], 2);  // 2: n维shape * 2
        } else if (inTensorDescs.at(weightIdx).shape.dimNum == 3) { // 3: dimNum
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(weightIdx).shape.dims[nDim + 1];
        } else {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(weightIdx).shape.dims[nDim];
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

std::map<std::string, uint32_t> ConstructLinearWithLoraTensorMap(
    const FusionLinearParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto linearInTensorCandidates = GetLinearInTensorCandidates();
    auto linearIntermediateTensorCandidates = GetLinearIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(linearInTensorCandidates, "default", inTensorList);

    // 添加Lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(linearInTensorCandidates, "lora_with_mask", inTensorList);
            AddTensorToList(linearIntermediateTensorCandidates, "lora_with_mask", intermediateTensorList);
        }
        AddTensorToList(linearInTensorCandidates, "lora", inTensorList);
        AddTensorToList(linearIntermediateTensorCandidates, "lora", intermediateTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddImMask(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mulNode;
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mulParam, &mulNode.operation));
    mulNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_im_mask"});
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_im_mask_out")};
    opGraph.nodes.push_back(mulNode);
    return atb::NO_ERROR;
}

int64_t AddLoraA(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    // 添加Lora A
    atb::Node loraALinearNode;
    if (param.loraEnableGMM) {
        AclNNGroupedMatmulParam aclnnParam;
        aclnnParam.transposeB = true;
        loraALinearNode.operation = new atb_speed::common::GroupedMatmulOperation("loraALinearNode", aclnnParam);
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateFusionLinear(param, &loraALinearNode.operation));
    }
    if (param.useImMask) {
        loraALinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_im_mask_out", "in_lora_a"});
    } else {
        loraALinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"in_input", "in_lora_a"});
    }
    if (param.loraEnableGMM) {
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    } else {
        // Lora权重暂不支持量化，以下Index仅为占位符
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_descale"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        loraALinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_compress_idx"));
    }
    loraALinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_lora_a_out")};
    opGraph.nodes.push_back(loraALinearNode);
    return atb::NO_ERROR;
}

int64_t AddLoraB(atb::GraphParam &opGraph, const FusionLinearParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    // 添加Lora B
    atb::Node loraBLinearNode;
    if (param.loraEnableGMM) {
        AclNNGroupedMatmulParam aclnnParam;
        aclnnParam.transposeB = false;
        loraBLinearNode.operation = new atb_speed::common::GroupedMatmulOperation("loraBLinearNode", aclnnParam);
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateFusionLinear(param, &loraBLinearNode.operation));
    }
    loraBLinearNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_lora_a_out", "in_lora_b"});
    if (param.loraEnableGMM) {
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_group_list"));
    } else {
        // Lora权重暂不支持量化，以下Index仅为占位符
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_scale"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_offset"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_descale"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_bias"));
        loraBLinearNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_compress_idx"));
    }
    loraBLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_lora_b_out")};
    opGraph.nodes.push_back(loraBLinearNode);
    return atb::NO_ERROR;
}

atb::Status CreateFusionLinearWithLora(const FusionLinearParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    std::map<std::string, uint32_t> tensorMap = ConstructLinearWithLoraTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    opGraph.name = "LinearWithLora";

    // 添加Base模型的Linear
    atb::Node baseLinearNode;
    atb_speed::common::FusionLinearParam baseLinearParam = param;
    baseLinearParam.supportLora = false;
    baseLinearParam.loraEnableGMM = false;
    CHECK_OPERATION_STATUS_RETURN(CreateFusionLinear(baseLinearParam, &baseLinearNode.operation));
    baseLinearNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "in_input", "in_weight", "in_scale", "in_offset",
        "in_descale", "in_bias", "in_compress_idx"
    });
    baseLinearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_base_linear_out")};
    opGraph.nodes.push_back(baseLinearNode);

    atb_speed::common::FusionLinearParam loraLinearParam;
    loraLinearParam.isBF16 = param.isBF16;
    loraLinearParam.hasBias = false;
    loraLinearParam.transposeType = TRANSPOSE;
    loraLinearParam.loraEnableGMM = param.loraEnableGMM;
    loraLinearParam.useImMask = param.useImMask;
    if (param.useImMask) {
        CHECK_OPERATION_STATUS_RETURN(AddImMask(opGraph, tensorMap));
    }
    CHECK_OPERATION_STATUS_RETURN(AddLoraA(opGraph, loraLinearParam, tensorMap));
    loraLinearParam.transposeType = NOT_TRANSPOSE;
    CHECK_OPERATION_STATUS_RETURN(AddLoraB(opGraph, loraLinearParam, tensorMap));

    // 合并Base模型的Linear输出和Lora Linear的输出
    atb::Node addNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
    addNode.inTensorIds = GetTensorIdxList(tensorMap, {"intermediate_base_linear_out", "intermediate_lora_b_out"});
    addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(addNode);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

atb::Status FusionLinear(const FusionLinearParam &param, atb::Operation **operation)
{
    if (param.supportLora) {
        return CreateFusionLinearWithLora(param, operation);
    } else {
        return CreateFusionLinear(param, operation);
    }
}
} // namespace common
} // namespace atb_speed