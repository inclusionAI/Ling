/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "integrated_gmm.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/aclnn/ops/grouped_matmul_operation.h"
#include "operations/aclnn/ops/dynamic_quant_operation.h"

namespace atb_speed {
namespace common {
enum QuantGmmTensorId : int {
    IN_HIDDENSTATUS = 0,
    IN_WEIGHT_EXPERT,
    IN_BIAS_EXPERT,
    IN_DESCALE_EXPERT,
    IN_OFFSET_EXPERT,
    IN_SCALE_EXPERT,
    IN_COMPRESS_IDX_EXPERT,
    IN_GROUP_LIST,
    OUT_GMM_RESULT,
    INTERMEDIATE_QUANT_OUT,
    INTERMEDIATE_DYNAMIC_SCALE,
};

static const uint64_t IN_TENSOR_COUNT = 8;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t INTERMEDIATE_TENSOR_COUNT_IF_PERTOKEN = 2;
static const uint64_t NODE_COUNT = 1;
static const uint64_t NODE_COUNT_IF_PERTOKEN = 2;
static const int IDX2 = 2;
static const int IDX3 = 3;

int64_t SetAclnnDynamicQuantNode(atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &dynamicQuantNode = opGraph.nodes.at(nodeId++);
    dynamicQuantNode.operation = new atb_speed::common::DynamicQuantOperation("DynamicQuantNode");
    dynamicQuantNode.inTensorIds = {IN_HIDDENSTATUS};
    dynamicQuantNode.outTensorIds = {INTERMEDIATE_QUANT_OUT,
                                     INTERMEDIATE_DYNAMIC_SCALE};
    ATB_SPEED_LOG_DEBUG("create dynamic quant");
    return atb::NO_ERROR;
}

int CalcGmmQuantType(const IntegratedGmmParam &param)
{
    int gmmQuantType = 0;
    int tempQuantType = 0;
    if (param.isUp) {
        tempQuantType = atb_speed::common::GetLinearQuantType(
            param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
                param.packQuantType : param.denseQuantType,
            param.moeLinearQuantType[IntegratedGmmIdx::MOE_MLP_GATE_IDX], false);
    } else {
        tempQuantType = atb_speed::common::GetLinearQuantType(
            param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED ?
                param.packQuantType : param.denseQuantType,
            param.moeLinearQuantType[IntegratedGmmIdx::MOE_MLP_DOWN_IDX], false);
    }
    if (tempQuantType == LinearQuantType::NO_QUANT) {
        gmmQuantType = GmmQuantType::NONE;
    } else if (tempQuantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_QUANT || \
                tempQuantType == LinearQuantType::LINEAR_W8A8_DYNAMIC_DEQUANT) {
        gmmQuantType = GmmQuantType::W8A8_TOKEN;
    } else if (tempQuantType == LinearQuantType::W8A16) {
        gmmQuantType = GmmQuantType::W8A16_CHANNEL;
    } else {
        gmmQuantType = GmmQuantType::W8A8_CHANNEL;
    }
    ATB_SPEED_LOG_DEBUG(gmmQuantType);
    return gmmQuantType;
}

atb::Status CreateW8A8Token(atb::Node &gmmNode)
{
    ATB_SPEED_LOG_DEBUG("push back W8A8_TOKEN");
    gmmNode.inTensorIds.push_back(IN_SCALE_EXPERT);
    gmmNode.inTensorIds.push_back(INTERMEDIATE_DYNAMIC_SCALE);
    gmmNode.inTensorIds.push_back(IN_GROUP_LIST);
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    gmmNode.inTensorReshapeFuncs[IDX2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2;
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("inTensorReshapeFuncs success");
    return atb::NO_ERROR;
}

atb::Status CreateW8A16Channel(atb::Node &gmmNode)
{
    ATB_SPEED_LOG_DEBUG("push back W8A16_CHANNEL");
    gmmNode.inTensorIds.push_back(IN_SCALE_EXPERT);
    gmmNode.inTensorIds.push_back(IN_OFFSET_EXPERT);
    gmmNode.inTensorIds.push_back(IN_GROUP_LIST);
    gmmNode.inTensorReshapeFuncs.resize(gmmNode.inTensorIds.size());
    gmmNode.inTensorReshapeFuncs[IDX2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    gmmNode.inTensorReshapeFuncs[IDX3] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        ATB_SPEED_LOG_DEBUG(oldShape.dimNum);
        newShape.dimNum = IDX2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("inTensorReshapeFuncs success");
    return atb::NO_ERROR;
}

// Op1 - GMM
atb::Status CreateGmm(atb::GraphParam &opGraph, size_t &nodeId,
                      const IntegratedGmmParam &param, int gmmQuantType)
{
    atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AclNNGroupedMatmulParam gmmParam;
    gmmParam.quantType = gmmQuantType;
    gmmParam.outDataType = param.outDataType;
    gmmParam.transposeB = param.transposeB;
    ATB_SPEED_LOG_DEBUG("Calc GmmQuantType success");
    gmmNode.operation = new atb_speed::common::GroupedMatmulOperation("gmmNode", gmmParam);
    gmmNode.outTensorIds = {OUT_GMM_RESULT};
    if (gmmParam.quantType == GmmQuantType::W8A8_TOKEN) {
        gmmNode.inTensorIds = {INTERMEDIATE_QUANT_OUT, IN_WEIGHT_EXPERT};
    } else {
        gmmNode.inTensorIds = {IN_HIDDENSTATUS, IN_WEIGHT_EXPERT};
    }
    if (param.hasBias) {
        gmmNode.inTensorIds.push_back(IN_BIAS_EXPERT);
    }
    if (gmmParam.quantType == GmmQuantType::W8A16_CHANNEL) {
        CHECK_OPERATION_STATUS_RETURN(CreateW8A16Channel(gmmNode));
    } else if (gmmParam.quantType == GmmQuantType::W8A8_CHANNEL) {
        ATB_SPEED_LOG_ERROR("MoE does not support W8A8_CHANNEL");
        gmmNode.inTensorIds.push_back(IN_SCALE_EXPERT);
        gmmNode.inTensorIds.push_back(IN_COMPRESS_IDX_EXPERT);
        gmmNode.inTensorIds.push_back(IN_GROUP_LIST);
    } else if (gmmParam.quantType == GmmQuantType::W8A8_TOKEN) {
        CHECK_OPERATION_STATUS_RETURN(CreateW8A8Token(gmmNode));
    } else {
        gmmNode.inTensorIds.push_back(IN_GROUP_LIST);
    }
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateIntegratedGmmOperation(const IntegratedGmmParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "integrated_gmm";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    uint64_t nodeCount = NODE_COUNT;
    int gmmQuantType = CalcGmmQuantType(param);
    if (gmmQuantType == GmmQuantType::W8A8_TOKEN) {
        opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT_IF_PERTOKEN;
        nodeCount = NODE_COUNT_IF_PERTOKEN;
    }
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    if (gmmQuantType == GmmQuantType::W8A8_TOKEN) {
        CHECK_OPERATION_STATUS_RETURN(SetAclnnDynamicQuantNode(opGraph, nodeId));
    }
    CHECK_OPERATION_STATUS_RETURN(CreateGmm(opGraph, nodeId, param, gmmQuantType));
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed