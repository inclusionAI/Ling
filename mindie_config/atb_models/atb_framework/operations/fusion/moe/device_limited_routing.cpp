/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "device_limited_routing.h"
#include <atb/atb_infer.h>
#include <memory>

namespace atb_speed {
namespace deviceLimitedRouting {
enum DeviceLimitedRoutingTensorId : int {
    IN_ROUTER_LOGITS,
    IN_EXPERT_GROUP,
    IN_ONE_HOT,
    IN_ZERO_HOT,
    OUT_ROUTER_LOGITS,
    INTERMIDATE_GROUP_MAX_LOGITS,
    DUMMY_IDX,
    DUMMY_LOGITS,
    INTERMIDATE_TOP_GROUP_IDX,
    INTERMIDATE_TOP_EXPERTS_IDX,
    INTERMIDATE_EXPERT_MASK,
    INTERMIDATE_EXPERT_MASK_FLOAT16,
    INTERMIDATE_EXPERT_MASK_FINAL
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 8;
static const uint64_t OPERATION_COUNT = 7;

// Op0 - Sort0
atb::Status CreateGroupTopOne(
    atb::Node &groupTopOneNode, atb::GraphParam opGraph,
    const DeviceLimitedRoutingParam &param,
    std::shared_ptr<int64_t> batchDimPtr)
{
    CHECK_PARAM_NE(param.numOfGroups, 0);
    atb::infer::SortParam topKExpertParam;
    topKExpertParam.num = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(topKExpertParam, &groupTopOneNode.operation));
    groupTopOneNode.inTensorIds = {IN_ROUTER_LOGITS};
    groupTopOneNode.outTensorIds = {INTERMIDATE_GROUP_MAX_LOGITS, DUMMY_IDX};
    groupTopOneNode.inTensorReshapeFuncs.resize(groupTopOneNode.inTensorIds.size());
    groupTopOneNode.inTensorReshapeFuncs[0] = [batchDimPtr, param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = param.numOfGroups;
        newShape.dims[2] = oldShape.dims[1] / param.numOfGroups; // 2:second dimension
    };
    opGraph.nodes.push_back(groupTopOneNode);
    ATB_SPEED_LOG_DEBUG("Reduction calculation success");
    return atb::NO_ERROR;
}

// Op1 - Sort1
atb::Status CreateTopkGroup(
    const DeviceLimitedRoutingParam &param,
    atb::Node &topKGroupNode, atb::GraphParam opGraph,
    std::shared_ptr<int64_t> batchDimPtr)
{
    atb::infer::SortParam topKGroupParam;
    topKGroupParam.num = param.topkGroups;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(topKGroupParam, &topKGroupNode.operation));
    topKGroupNode.inTensorIds = {INTERMIDATE_GROUP_MAX_LOGITS};
    topKGroupNode.outTensorIds = {DUMMY_LOGITS, INTERMIDATE_TOP_GROUP_IDX};
    topKGroupNode.inTensorReshapeFuncs.resize(topKGroupNode.inTensorIds.size());
    topKGroupNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
    };
    opGraph.nodes.push_back(topKGroupNode);
    ATB_SPEED_LOG_DEBUG("Group selection success");
    return atb::NO_ERROR;
}

// Op2 - GroupId -> ExpertId
atb::Status CreateGather(std::shared_ptr<int64_t> batchDimPtr, atb::Node &gatherNode, atb::GraphParam opGraph)
{
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode.operation));
    gatherNode.inTensorIds = {IN_EXPERT_GROUP, INTERMIDATE_TOP_GROUP_IDX};
    gatherNode.outTensorIds = {INTERMIDATE_TOP_EXPERTS_IDX};
    gatherNode.inTensorReshapeFuncs.resize(gatherNode.inTensorIds.size());
    gatherNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // 3:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    opGraph.nodes.push_back(gatherNode);
    ATB_SPEED_LOG_DEBUG("Gather0 calculation success");
    return atb::NO_ERROR;
}


// Op3 - ExpertId -> OneHotMask
atb::Status CreateOneHot(
    const DeviceLimitedRoutingParam &param,
    std::shared_ptr<int64_t> batchDimPtr,
    atb::Node &oneHotNode, atb::GraphParam opGraph)
{
    CHECK_PARAM_NE(param.topkGroups.at(0), 0);
    atb::infer::OnehotParam onehotParam;
    onehotParam.axis = 2; // 2:specify axis for oneHotOperation
    onehotParam.depth = param.numOfExperts;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(onehotParam, &oneHotNode.operation));
    oneHotNode.inTensorIds = {INTERMIDATE_TOP_EXPERTS_IDX, IN_ONE_HOT, IN_ZERO_HOT};
    oneHotNode.outTensorIds = {INTERMIDATE_EXPERT_MASK};
    oneHotNode.inTensorReshapeFuncs.resize(oneHotNode.inTensorIds.size());
    oneHotNode.inTensorReshapeFuncs[0] = [batchDimPtr, param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0] / param.topkGroups.at(0);
        newShape.dims[1] = oldShape.dims[1] * param.topkGroups.at(0);
    };
    opGraph.nodes.push_back(oneHotNode);
    ATB_SPEED_LOG_DEBUG("Expert Mask created success");
    return atb::NO_ERROR;
}


// Op4 - CastforReduceSum
atb::Status CreateCast(atb::Node &castNode, atb::GraphParam opGraph)
{
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {INTERMIDATE_EXPERT_MASK};
    castNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_FLOAT16};
    opGraph.nodes.push_back(castNode);
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

// Op5 - Finalize Device-limited Mask
atb::Status CreateMask(atb::Node &maskNode, atb::GraphParam opGraph)
{
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &maskNode.operation));
    maskNode.inTensorIds = {INTERMIDATE_EXPERT_MASK_FLOAT16};
    maskNode.outTensorIds = {INTERMIDATE_EXPERT_MASK_FINAL};
    opGraph.nodes.push_back(maskNode);
    ATB_SPEED_LOG_DEBUG("Mask reduction calculation success");
    return atb::NO_ERROR;
}

// Op6 - Finalize Router Logits
atb::Status CreateElewiseMul(atb::Node &mulNode, atb::GraphParam opGraph)
{
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &mulNode.operation));
    mulNode.inTensorIds = {IN_ROUTER_LOGITS, INTERMIDATE_EXPERT_MASK_FINAL};
    mulNode.outTensorIds = {OUT_ROUTER_LOGITS};
    opGraph.nodes.push_back(mulNode);
    ATB_SPEED_LOG_DEBUG("ElewiseMul0 calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateDeviceLimitedRoutingOperation(const DeviceLimitedRoutingParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "DeviceLimitedRouting";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    uint64_t nodeSize = OPERATION_COUNT;
    opGraph.nodes.resize(nodeSize);
    size_t nodeId = 0;

    atb::Node &groupTopOneNode = opGraph.nodes.at(nodeId++);
    atb::Node &topKGroupNode = opGraph.nodes.at(nodeId++);
    atb::Node &gatherNode = opGraph.nodes.at(nodeId++);
    atb::Node &oneHotNode = opGraph.nodes.at(nodeId++);
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::Node &maskNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);

    CHECK_OPERATION_STATUS_RETURN(CreateGroupTopOne(groupTopOneNode, opGraph, param, batchDimPtr));
    CHECK_OPERATION_STATUS_RETURN(CreateTopkGroup(param, topKGroupNode, opGraph, batchDimPtr));
    CHECK_OPERATION_STATUS_RETURN(CreateGather(batchDimPtr, gatherNode, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateOneHot(param, batchDimPtr, oneHotNode, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateCast(castNode, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateMask(maskNode, opGraph));
    CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul(mulNode, opGraph));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
}
