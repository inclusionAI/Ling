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
#include "parallel_lmhead.h"

#include <atb/atb_infer.h>

#include "parallel_layer.h"

namespace atb_speed {
namespace common {
atb::Status InnerInferShape(const ParallelLmHeadParam &param, atb::GraphParam &opGraph)
{
    if (param.gatherAhead) {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(2).shape.dims[0]; // 2 下标
            outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0] * param.rankSize;
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0) = inTensorDescs.at(0);
            auto dimLast = inTensorDescs.at(0).shape.dimNum - 1;
            outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(1).shape.dims[0] * param.rankSize;
            return atb::NO_ERROR;
        };
    }
    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateParallelLmHeadBase(const ParallelLmHeadParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = config.inTensorNum;
    opGraph.outTensorNum = config.outTensorNum;
    opGraph.internalTensorNum = config.interTensorNum;
    opGraph.nodes.resize(config.nodeCount);
    if (param.gatherAhead) {
        opGraph.name = "Parallel_LmHead_GatherAhead";
    } else {
        opGraph.name = "Parallel_LmHead";
    }
    size_t nodeId = 0;
    if (param.gatherAhead) {
        auto &gatherNode = opGraph.nodes.at(nodeId++);
        atb::infer::GatherParam gatherParam;
        CREATE_OPERATION(gatherParam, &gatherNode.operation);
        gatherNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_LMHEAD_INDICES_ID };
        gatherNode.outTensorIds = { config.INTERMIDATE_GATHER_OUT_ID };
    }
    atb::Node &parallelLinearNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::ParallelParam parallelParam;
    parallelParam.rank = param.rank;
    parallelParam.rankSize = param.rankSize;
    parallelParam.isBias = false;
    parallelParam.backend = param.backend;
    parallelParam.isBF16 = param.isBF16;
    atb_speed::common::ColumnParallelLinear(parallelParam, &parallelLinearNode.operation);
    parallelLinearNode.inTensorIds = { param.gatherAhead ? config.INTERMIDATE_GATHER_OUT_ID : config.IN_HIDDENSTATES_ID,
        config.IN_WEIGHT_ID };
    parallelLinearNode.outTensorIds = { param.rankSize > 1 ? config.INTERMEDIATE_ALLGATHER_OUT_ID :
                                                             config.OUT_LOGITS_ID };
    if (param.rankSize > 1) {
        atb::Node &transposeNode = opGraph.nodes.at(nodeId++);
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = { 1, 0, 2 }; // 2 是维度重新排列顺序
        } else {
            transposeParam.perm = { 1, 2, 0, 3 }; // 2 3 是维度重新排列顺序
        }
        CREATE_OPERATION(transposeParam, &transposeNode.operation);
        transposeNode.inTensorIds = { config.INTERMEDIATE_ALLGATHER_OUT_ID };
        transposeNode.outTensorIds = { config.OUT_LOGITS_ID };
    }
    CHECK_OPERATION_STATUS_RETURN(InnerInferShape(param, opGraph));
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status ParallelLmHead(const ParallelLmHeadParam &param, atb::Operation **operation)
{
    if (param.rankSize > 1) {
        if (!param.gatherAhead) {
            return CreateParallelLmHeadBase(param, operation,
                ParallelLmHeadConfig(2, 1, 1, 2)); // 2 输入张量 1 输出张量 1 中间张量 2 节点 数量
        } else if (param.unpadInputs) {
            return CreateParallelLmHeadBase(param, operation,
                ParallelLmHeadGatherAheadConfig(3, 1, 2, 3)); // 3 输入张量 1 输出张量 2 中间张量 3 节点 数量
        } else {
            ATB_SPEED_LOG_ERROR("[gatherAhead] can only used with [unpadInputs]");
            return atb::ERROR_INVALID_PARAM;
        }
    } else {
        if (!param.gatherAhead) {
            return CreateParallelLmHeadBase(param, operation,
                ParallelLmHeadConfig(2, 1, 0, 1)); // 2 输入张量 1 输出张量 0 中间张量 1 节点 数量
        } else if (param.unpadInputs) {
            return CreateParallelLmHeadBase(param, operation,
                ParallelLmHeadGatherAheadConfig(3, 1, 1, 2)); // 3 输入张量 1 输出张量 1 中间张量 2 节点 数量
        } else {
            ATB_SPEED_LOG_ERROR("[gatherAhead] can only used with [unpadInputs]");
            return atb::ERROR_INVALID_PARAM;
        }
    }
}
} // namespace common
} // namespace atb_speed