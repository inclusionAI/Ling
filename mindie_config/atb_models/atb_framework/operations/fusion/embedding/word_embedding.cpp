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

#include "atb_speed/utils/check_util.h"
#include "operations/fusion/embedding/word_embedding.h"

namespace atb_speed {
namespace common {

enum WordEmbeddingTensorIdx : uint32_t {
    IN_EMBEDDING_WEIGHTS = 0,
    IN_INPUT_IDS,
    OUT_HIDDEN_STATES,
    INTERMEDIATE_GATHER,
    INTERMEDIATE_ALLGATHER_OUT_ID,
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_NO_ALL_GATHER_COUNT = 0;
static const uint64_t INTERMEDIATE_TENSOR_ALL_GATHER_COUNT = 2;
static const uint64_t NODE_NO_ALL_GATHER_COUNT = 1;
static const uint64_t NODE_ALL_GATHER_COUNT = 3;

int64_t AddAllGatherTranspose(atb::GraphParam &opGraph, const WordEmbeddingParam &param)
{
    atb::Node allGatherNode;
    atb::infer::AllGatherParam allGatherParam;
    allGatherParam.rank = param.tensorParallelInfo.rank;
    allGatherParam.rankSize = param.tensorParallelInfo.worldSize;
    allGatherParam.backend = param.tensorParallelInfo.backend;
    allGatherParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
    allGatherNode.inTensorIds = {WordEmbeddingTensorIdx::INTERMEDIATE_GATHER};
    allGatherNode.outTensorIds = {WordEmbeddingTensorIdx::INTERMEDIATE_ALLGATHER_OUT_ID};
    opGraph.nodes.push_back(allGatherNode);

    atb::Node transposeNode;
    atb::infer::TransposeParam transposeParam;
    if (param.unpadInputs) {
        transposeParam.perm = {1, 0, 2};
    } else {
        transposeParam.perm = {1, 2, 0, 3};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
    transposeNode.inTensorIds = {WordEmbeddingTensorIdx::INTERMEDIATE_ALLGATHER_OUT_ID};
    transposeNode.outTensorIds = {WordEmbeddingTensorIdx::OUT_HIDDEN_STATES};
    opGraph.nodes.push_back(transposeNode);
    return atb::NO_ERROR;
}

atb::Status WordEmbedding(const WordEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    // 若权重按列切分，则需使用all gather方式收集完整的hidden states
    // 相比不使用all gather会多两个internalTensor和两个node
    opGraph.internalTensorNum = \
        param.tensorParallelInfo.worldSize > 1 ? \
        INTERMEDIATE_TENSOR_ALL_GATHER_COUNT : INTERMEDIATE_TENSOR_NO_ALL_GATHER_COUNT;
    opGraph.name = "WordEmbedding";

    atb::Node inputIdEmbeddingNode;
    atb::infer::GatherParam inputembedinggatherparam;
    inputembedinggatherparam.axis = param.axis;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(inputembedinggatherparam, &inputIdEmbeddingNode.operation));
    inputIdEmbeddingNode.inTensorIds = {
        WordEmbeddingTensorIdx::IN_EMBEDDING_WEIGHTS, WordEmbeddingTensorIdx::IN_INPUT_IDS
    };
    inputIdEmbeddingNode.outTensorIds = {
        param.tensorParallelInfo.worldSize > 1 ? \
        WordEmbeddingTensorIdx::INTERMEDIATE_GATHER : WordEmbeddingTensorIdx::OUT_HIDDEN_STATES
    };
    opGraph.nodes.push_back(inputIdEmbeddingNode);

    if (param.tensorParallelInfo.worldSize > 1) {
        CHECK_OPERATION_STATUS_RETURN(AddAllGatherTranspose(opGraph, param));
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = inTensorDescs.at(IN_EMBEDDING_WEIGHTS).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(IN_EMBEDDING_WEIGHTS).format;
        if (param.unpadInputs) {
            outTensorDescs.at(0).shape.dimNum = 2;  // 2: 第一个输出tensor的维度为2
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_INPUT_IDS).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = CheckIntMulOverFlow(
                inTensorDescs.at(IN_EMBEDDING_WEIGHTS).shape.dims[1], param.tensorParallelInfo.worldSize);
        } else {
            outTensorDescs.at(0).shape.dimNum = 3;  // 3: 第一个输出tensor的维度为3
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_INPUT_IDS).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(IN_INPUT_IDS).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] =  // 2: 第2维
                CheckIntMulOverFlow(
                    inTensorDescs.at(IN_EMBEDDING_WEIGHTS).shape.dims[1],
                    param.tensorParallelInfo.worldSize);
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}
}  // namespace common
}  // namespace atb_speed