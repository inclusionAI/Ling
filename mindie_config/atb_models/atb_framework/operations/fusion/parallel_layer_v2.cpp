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

#include <atb/atb_infer.h>
#include "nlohmann/json.hpp"
#include "parallel_layer_v2.h"


namespace atb_speed {
namespace common {
enum ParallelType : int {
    ROW_PARALLEL = 0,
    COLUMN_PARALLEL,
};

enum InTensorId : int {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_BIAS,
    IN_DEQSCALE,
    IN_INDEX_IDS,
    IN_OFFSET,
    IN_COMPRESSINFO,
    OUT_LINEAR,
    INTER_ID,
};

atb::Status CalNodeNum(size_t &nodeCount, size_t &internalTensorNum, const ParallelParamV2 &param,
    const ParallelType parallelType)
{
    if (param.isQuant) {
        if (param.quantParam.isQuantOp) {
            nodeCount += 1;
            internalTensorNum += 1;
        }
    } else {
        if (param.isBias) {
            nodeCount += 1;
            internalTensorNum += 1;
        }
    }

    if (param.commParam.rankSize > 1) {
        nodeCount += 1;
        internalTensorNum += 1;
        if (parallelType == COLUMN_PARALLEL && param.isAllGatherTranspose) {
            nodeCount += 1;
            internalTensorNum += 1;
        }
    }
    return atb::NO_ERROR;
}

atb::Status AddmatmulNode(const ParallelParamV2 &param, atb::GraphParam &opGraph, size_t &nodeId, uint32_t &inteId)
{
    if (!param.isQuant) {
        atb::Node &matmulNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam matmulParam = { param.transposeA, param.transposeB, false };
        CREATE_OPERATION(matmulParam, &matmulNode.operation);
        matmulNode.inTensorIds = { IN_INPUT, IN_WEIGHT };
        matmulNode.outTensorIds = { (param.commParam.rankSize > 1 || param.isBias) ?
            inteId :
            static_cast<uint32_t>(OUT_LINEAR) };
    } else {
        if (param.quantParam.isQuantOp) {
            atb::Node &quantNode = opGraph.nodes.at(nodeId++);
            atb::infer::ElewiseParam quantParam;
            quantParam.elewiseType = param.quantParam.elewiseType;
            quantParam.quantParam.inputScale = param.quantParam.inputScale;
            quantParam.quantParam.inputOffset = param.quantParam.inputOffset;
            CREATE_OPERATION(quantParam, &quantNode.operation);
            quantNode.inTensorIds = { IN_INPUT };
            quantNode.outTensorIds = { inteId };
        }

        if (param.isSparse) {
            atb::Node &matmulNode = opGraph.nodes.at(nodeId++);
            atb::infer::LinearSparseParam linearSparseParam = { false, true, 8, 8 }; // 8 压缩参数
            CREATE_OPERATION(linearSparseParam, &matmulNode.operation);
            matmulNode.inTensorIds = { param.quantParam.isQuantOp ? inteId++ : static_cast<uint32_t>(IN_INPUT),
                static_cast<uint32_t>(IN_WEIGHT), static_cast<uint32_t>(IN_BIAS), static_cast<uint32_t>(IN_DEQSCALE),
                static_cast<uint32_t>(IN_INDEX_IDS) };
            matmulNode.outTensorIds = { param.commParam.rankSize > 1 ? inteId : static_cast<uint32_t>(OUT_LINEAR) };
        } else {
            atb::Node &matmulNode = opGraph.nodes.at(nodeId++);
            atb::infer::LinearParam matmulParam;
            matmulParam.transposeA = param.transposeA;
            matmulParam.transposeB = param.transposeB;
            matmulParam.outDataType = ACL_FLOAT16;
            CREATE_OPERATION(matmulParam, &matmulNode.operation);
            matmulNode.inTensorIds = { param.quantParam.isQuantOp ? inteId++ : static_cast<uint32_t>(IN_INPUT),
                static_cast<uint32_t>(IN_WEIGHT), static_cast<uint32_t>(IN_BIAS), static_cast<uint32_t>(IN_DEQSCALE) };
            matmulNode.outTensorIds = { param.commParam.rankSize > 1 ? inteId : static_cast<uint32_t>(OUT_LINEAR) };
        }
    }
    return atb::NO_ERROR;
}

atb::Status CalMulRank(const ParallelParamV2 &param, atb::GraphParam &opGraph, size_t &nodeId, uint32_t &inteId,
    const ParallelType parallelType)
{
    if (param.commParam.rankSize > 1) {
        atb::Node &parallelNode = opGraph.nodes.at(nodeId++);

        if (parallelType == ROW_PARALLEL) {
            atb::infer::AllReduceParam allReduceParam;
            allReduceParam.rank = param.commParam.rank;
            allReduceParam.rankSize = param.commParam.rankSize;
            allReduceParam.backend = param.commParam.backend;
            CREATE_OPERATION(allReduceParam, &parallelNode.operation);
            parallelNode.inTensorIds = { inteId++ };
            parallelNode.outTensorIds = { param.isBias && !param.isQuant ? inteId : static_cast<uint32_t>(OUT_LINEAR) };
        } else {
            atb::infer::AllGatherParam allGatherParam;
            allGatherParam.rank = param.commParam.rank;
            allGatherParam.rankSize = param.commParam.rankSize;
            allGatherParam.backend = param.commParam.backend;
            CREATE_OPERATION(allGatherParam, &parallelNode.operation);
            parallelNode.inTensorIds = { inteId++ };
            parallelNode.outTensorIds = { (param.isBias && !param.isQuant) || param.isAllGatherTranspose ?
                inteId :
                static_cast<uint32_t>(OUT_LINEAR) };

            // (world_size,bs,seq,vocab_size//world_size)
            // -> (bs,seq,world_size,vocab_size//world_size)
            // -> (bs,seq,vocab_size)
            if (param.isAllGatherTranspose) {
                atb::Node &gatherTransposeNode = opGraph.nodes.at(nodeId++);
                atb::infer::TransposeParam gatherTransposeParam;
                gatherTransposeParam.perm = { 1, 2, 0, 3 };
                CREATE_OPERATION(gatherTransposeParam, &gatherTransposeNode.operation);
                gatherTransposeNode.inTensorIds = { inteId++ };
                gatherTransposeNode.outTensorIds = { param.isBias && !param.isQuant ?
                    inteId :
                    static_cast<uint32_t>(OUT_LINEAR) };
            }
        }
    }
    return atb::NO_ERROR;
}

atb::Status CalBias(const ParallelParamV2 &param, atb::GraphParam &opGraph, size_t &nodeId, const uint32_t &inteId)
{
    if (param.isBias && !param.isQuant) {
        atb::Node &addNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &addNode.operation);
        addNode.inTensorIds = { inteId, IN_BIAS };
        addNode.outTensorIds = { OUT_LINEAR };
    }
    return atb::NO_ERROR;
}

atb::Status RowParallelInferShape(const ParallelParamV2 &param, atb::GraphParam &opGraph)
{
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (param.isQuant) {
            outTensorDescs.at(0).dtype = ACL_FLOAT16;
        } else {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        }
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        auto dimNum = inTensorDescs.at(0).shape.dimNum;
        auto wdim = inTensorDescs.at(1).shape.dimNum;
        outTensorDescs.at(0).shape.dimNum = dimNum;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        if (param.isQuant && param.isSparse) {
            if (dimNum == 3) { // 3是张量的维度数
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
                outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(2).shape.dims[0]; // 2 dim维度数下标 2 下标
            } else {
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(2).shape.dims[0]; // 2 下标
            }
        } else if (param.isQuant) {
            if (dimNum == 3) { // 3是张量的维度数
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            }
            outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[wdim - 2]; // ND,NZ统一为-2轴
        } else {
            if (dimNum == 3) { // 3是张量的维度数
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            }
            outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(1).shape.dims[0];
        }
        return atb::NO_ERROR;
    };
    return atb::NO_ERROR;
}

atb::Status NoRowParallelInferShape(const ParallelParamV2 &param, atb::GraphParam &opGraph)
{
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        if (!param.isQuant) {
            outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        } else {
            outTensorDescs.at(0).dtype = ACL_FLOAT16;
        }
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        auto dimNum = inTensorDescs.at(0).shape.dimNum;
        if (param.isAllGatherTranspose) {
            outTensorDescs.at(0).shape.dimNum = dimNum;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {                                                          // 3是张量的维度数
                outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1]; // dim 2
            }
            outTensorDescs.at(0).shape.dims[dimNum - 1] =
                inTensorDescs.at(1).shape.dims[0] * param.commParam.rankSize; // last dim
        } else {
            outTensorDescs.at(0).shape.dimNum = dimNum + 1; // add rank dim
            outTensorDescs.at(0).shape.dims[0] = param.commParam.rankSize;
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
            if (dimNum == 3) {                                                          // 3是张量的维度数
                outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(0).shape.dims[1]; // dim 2
            }
            outTensorDescs.at(0).shape.dims[dimNum] = inTensorDescs.at(1).shape.dims[0]; // last dim
        }

        return atb::NO_ERROR;
    };
    return atb::NO_ERROR;
}

atb::Status ParallelLinearBaseV2(const ParallelParamV2 &param, atb::Operation **operation,
    const ParallelType parallelType)
{
    atb::GraphParam opGraph;
    opGraph.name = "ParallelLinearBaseV2";
    opGraph.inTensorNum = 7; // 7是输入张量数量
    opGraph.outTensorNum = 1;
    // 判断node个数
    size_t nodeCount = 1;
    size_t internalTensorNum = 0;
    CHECK_OPERATION_STATUS_RETURN(CalNodeNum(nodeCount, internalTensorNum, param, parallelType));
    opGraph.internalTensorNum = internalTensorNum;
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    uint32_t inteId = INTER_ID;
    CHECK_OPERATION_STATUS_RETURN(AddmatmulNode(param, opGraph, nodeId, inteId));
    CHECK_OPERATION_STATUS_RETURN(CalMulRank(param, opGraph, nodeId, inteId, parallelType));
    CHECK_OPERATION_STATUS_RETURN(CalBias(param, opGraph, nodeId, inteId));
    if (parallelType == ROW_PARALLEL) {
        CHECK_OPERATION_STATUS_RETURN(RowParallelInferShape(param, opGraph));
    } else {
        CHECK_OPERATION_STATUS_RETURN(NoRowParallelInferShape(param, opGraph));
    }
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}


atb::Status ParallelLinearV2(const ParallelParamV2 &param, atb::Operation **operation, const ParallelType parallelType)
{
    return ParallelLinearBaseV2(param, operation, parallelType); // 5:in 1:out 3:inter
}


atb::Status RowParallelLinearV2(const ParallelParamV2 &param, atb::Operation **operation)
{
    return ParallelLinearV2(param, operation, ROW_PARALLEL);
}

atb::Status ColumnParallelLinearV2(const ParallelParamV2 &param, atb::Operation **operation)
{
    return ParallelLinearV2(param, operation, COLUMN_PARALLEL);
}

atb::Status VocabParallelEmbeddingV2(const atb::Operation **operation)
{
    (void)operation;
    return 0;
}
} // namespace common
} // namespace atb_speed
