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
#include "parallel_layer.h"

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

namespace atb_speed {
namespace common {
enum ParallelType : int {
    ROW_PARALLEL = 0,
    COLUMN_PARALLEL,
};

atb::Status InnerParallelLinearBase(const ParallelParam &param, atb::GraphParam &opGraph,
    const ParallelType parallelType)
{
    if (parallelType == ROW_PARALLEL) {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) { // 维度数量 3
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            }
            if (param.transposeB) {
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[0];
            } else {
                outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[1];
            }
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
            outTensorDescs.at(0).format = inTensorDescs.at(0).format;
            auto dimNum = inTensorDescs.at(0).shape.dimNum;
            outTensorDescs.at(0).shape.dimNum = dimNum + 1; // add rank dim
            outTensorDescs.at(0).shape.dims[0] = param.rankSize;
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {                                                          // 维度数量 3
                outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dim 2
            }
            if (param.transposeB) {
                outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[0]; // last dim
            } else {
                outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[1]; // last dim
            }
            return atb::NO_ERROR;
        };
    }
    return atb::NO_ERROR;
}

template <class T>
atb::Status ParallelLinearBase(const ParallelParam &param, atb::Operation **operation, T config,
    const ParallelType parallelType)
{
    atb::GraphParam opGraph;
    opGraph.name = "ParallelLinearBase";
    opGraph.inTensorNum = static_cast<unsigned>(config.inTensorNum);
    opGraph.outTensorNum = static_cast<unsigned>(config.outTensorNum);
    opGraph.internalTensorNum = static_cast<unsigned>(config.interTensorNum);
    opGraph.nodes.resize(config.nodeCount);

    size_t nodeId = 0;
    atb::Node &matmulNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam matmulParam = { param.transposeA, param.transposeB, false };
    CREATE_OPERATION(matmulParam, &matmulNode.operation);
    matmulNode.inTensorIds = { config.IN_INPUT, config.IN_WEIGHT };
    matmulNode.outTensorIds = { config.INTERMIDATE_MATMULOUT };

    if (param.rankSize > 1) {
        atb::Node &parallelNode = opGraph.nodes.at(nodeId++);

        if (parallelType == ROW_PARALLEL) {
            atb::infer::AllReduceParam allReduceParam;
            allReduceParam.rank = param.rank;
            allReduceParam.rankSize = param.rankSize;
            allReduceParam.backend = param.backend;
            CREATE_OPERATION(allReduceParam, &parallelNode.operation);
        } else {
            atb::infer::AllGatherParam allGatherParam;
            allGatherParam.rank = param.rank;
            allGatherParam.rankSize = param.rankSize;
            allGatherParam.backend = param.backend;
            CREATE_OPERATION(allGatherParam, &parallelNode.operation);
        }

        parallelNode.inTensorIds = { config.INTERMIDATE_MATMULOUT };
        parallelNode.outTensorIds = { config.INTERMIDATE_ALLREDUCEOUT };
    }

    if (param.isBias) {
        atb::Node &addNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &addNode.operation);
        addNode.inTensorIds = { param.rankSize > 1 ? config.INTERMIDATE_ALLREDUCEOUT : config.INTERMIDATE_MATMULOUT,
            config.IN_BIAS };
        addNode.outTensorIds = { config.OUT_LINEAROUT };
    }
    InnerParallelLinearBase(param, opGraph, parallelType);
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status ParallelLinear(const ParallelParam &param, atb::Operation **operation, const ParallelType parallelType)
{
    if (param.isBias && (param.rankSize > 1)) {
        return ParallelLinearBase(param, operation,
            LinearWithBiasAndParallel(3, 1, 2, 3), // 3是输入张量数量 1是输出张量数量 2是中间张量数量 3是节点数量
            parallelType);                         // 3:in 1:out 2:inter 3:node
    } else if (param.isBias) {
        return ParallelLinearBase(param, operation,
            LinearWithBias(3, 1, 1, 2), // 3是输入张量数量 1是输出张量数量 1是中间张量数量 2是节点数量
            parallelType);              // 3:in 1:out 1:inter 2:node
    } else if (param.rankSize > 1) {
        return ParallelLinearBase(param, operation,
            LinearWithParallel(2, 1, 1, 2), // 2是输入张量数量 1是输出张量数量 1是中间张量数量 2是节点数量
            parallelType);                  // 2:in 1:out 1:inter 2:node
    } else {
        return ParallelLinearBase(param, operation, LinearOnly(2, 1, 0, 1), parallelType); // 2:in 1:out 0:inter 1:node
    }
}

atb::Status RowParallelLinear(const ParallelParam &param, atb::Operation **operation)
{
    return ParallelLinear(param, operation, ROW_PARALLEL);
}

atb::Status ColumnParallelLinear(const ParallelParam &param, atb::Operation **operation)
{
    return ParallelLinear(param, operation, COLUMN_PARALLEL);
}

atb::Status VocabParallelEmbedding(const atb::Operation **operation)
{
    (void)&operation;
    return 0;
}
} // namespace common
} // namespace atb_speed
