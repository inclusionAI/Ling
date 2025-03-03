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
#include "operations/fusion/infer_shape_functions.h"
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/embedding/positional_embedding.h"

namespace atb_speed {
namespace common {

enum PositionalEmbeddingGatherTensorIdx : uint32_t {
    IN_POSITION_IDS = 0,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    OUT_COS_EMBEDDING,
    OUT_SIN_EMBEDDING,
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 2;

atb::Status PositionalEmbeddingGather(atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = "PositionalEmbeddingGather";

    atb::Node cosEmbeddingNode;
    atb::infer::GatherParam cosEmbeddingGatherParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(cosEmbeddingGatherParam, &cosEmbeddingNode.operation));
    cosEmbeddingNode.inTensorIds = {
        PositionalEmbeddingGatherTensorIdx::IN_COS_TABLE, PositionalEmbeddingGatherTensorIdx::IN_POSITION_IDS
    };
    cosEmbeddingNode.outTensorIds = {PositionalEmbeddingGatherTensorIdx::OUT_COS_EMBEDDING};
    opGraph.nodes.push_back(cosEmbeddingNode);

    atb::Node sinEmbeddingNode;
    atb::infer::GatherParam sinEmbeddingGatherParam;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(sinEmbeddingGatherParam, &sinEmbeddingNode.operation));
    sinEmbeddingNode.inTensorIds = {
        PositionalEmbeddingGatherTensorIdx::IN_SIN_TABLE, PositionalEmbeddingGatherTensorIdx::IN_POSITION_IDS
    };
    sinEmbeddingNode.outTensorIds = {PositionalEmbeddingGatherTensorIdx::OUT_SIN_EMBEDDING};
    opGraph.nodes.push_back(sinEmbeddingNode);

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_COS_TABLE);
        if (inTensorDescs.at(IN_COS_TABLE).shape.dimNum >= 3) {  // 3: 如果IN_COS_TABLE的维度大于3
            outTensorDescs.at(0).shape.dimNum = 3;  // 3: 第一个输出tensor的维度为3
        } else {
            outTensorDescs.at(0).shape.dimNum = 2;  // 2: 第一个输出tensor的维度为2
        }
        outTensorDescs.at(0).shape.dims[0] = 1;
        // unpadInputs=True场景下，for loop只循环一次；unpadInputs=False场景下，for loop循环两次，将bsz和seqLen合轴
        CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(IN_POSITION_IDS).shape.dimNum);
        for (uint64_t i = 0; i < inTensorDescs.at(IN_POSITION_IDS).shape.dimNum; i++) {
            outTensorDescs.at(0).shape.dims[0] = CheckIntMulOverFlow(
                outTensorDescs.at(0).shape.dims[0], inTensorDescs.at(IN_POSITION_IDS).shape.dims[i]);
        }
        
        outTensorDescs.at(1) = outTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}
static const uint64_t POS_EMB_IN_TENSOR_COUNT = 5;
static const uint64_t POS_EMB_OUT_TENSOR_COUNT = 2;
static const uint64_t POS_EMB_INTERMEDIATE_TENSOR_2D_COUNT = 6;
static const uint64_t POS_EMB_INTERMEDIATE_TENSOR_1D_COUNT = 0;
static const uint64_t POS_EMB_NODE_2D_COUNT = 5;
static const uint64_t POS_EMB_NODE_1D_COUNT = 1;

static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;
static const int64_t DIM_LAST = -1;
static const uint64_t DIM_0 = 0;
static const uint64_t DIM_1 = 1;
static const uint64_t DIM_2 = 2;
static const uint64_t DIM_3 = 3;
static const uint64_t SPLIT_NUM_2 = 2;
static const uint64_t SPLIT_NUM_3 = 3;

static void SqueezeRopeIntensor(const atb::Dims &oldShape, atb::Dims &newShape)
{
    if (oldShape.dimNum == DIM_NUM_4) {
        newShape.dimNum = DIM_NUM_2;
        newShape.dims[0] = CheckIntMulOverFlow(oldShape.dims[0], oldShape.dims[1]);
        newShape.dims[1] = CheckIntMulOverFlow(oldShape.dims[DIM_2], oldShape.dims[DIM_3]);
    } else if (oldShape.dimNum == DIM_NUM_3) {
        newShape.dimNum = DIM_NUM_2;
        newShape.dims[0] = CheckIntMulOverFlow(oldShape.dims[0], oldShape.dims[1]);
        newShape.dims[1] = oldShape.dims[DIM_2];
    } else {
        newShape = oldShape;
    }
}

enum class RotaryPositionEmbeddingTensorId : int {
    IN_QUERY = 0,
    IN_KEY,
    IN_ROPE_COS,
    IN_ROPE_SIN,
    IN_SEQLEN,

    OUT_QUERY,
    OUT_KEY,

    INTERMEDIATE_QCHUNK0,
    INTERMEDIATE_QCHUNK1,
    INTERMEDIATE_KCHUNK0,
    INTERMEDIATE_KCHUNK1,
    INTERMEDIATE_QOUT,
    INTERMEDIATE_KOUT,
};

#define POS_EMB_CAST(x) static_cast<int>(RotaryPositionEmbeddingTensorId::x)

int64_t AddInferShapeFunc(atb::GraphParam &opGraph, const RotaryPositionEmbeddingParam &param)
{
    if (param.isFA) {
        opGraph.inferShapeFunc = [=](
            const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs
        ) {
            outTensorDescs.at(0) = inTensorDescs.at(POS_EMB_CAST(IN_QUERY));
            outTensorDescs.at(0).shape.dimNum = DIM_NUM_4;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(POS_EMB_CAST(IN_QUERY)).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(POS_EMB_CAST(IN_QUERY)).shape.dims[1];
            outTensorDescs.at(0).shape.dims[DIM_2] = param.headNum;
            outTensorDescs.at(0).shape.dims[DIM_3] = param.headDim;
            outTensorDescs.at(1) = inTensorDescs.at(POS_EMB_CAST(IN_KEY));
            outTensorDescs.at(1).shape.dimNum = DIM_NUM_4;
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(POS_EMB_CAST(IN_KEY)).shape.dims[0];
            outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(POS_EMB_CAST(IN_KEY)).shape.dims[1];
            outTensorDescs.at(1).shape.dims[DIM_2] = param.kvHeadNum;
            outTensorDescs.at(1).shape.dims[DIM_3] = param.headDim;
            return atb::NO_ERROR;
        };
    } else {
        opGraph.inferShapeFunc = [=] (
            const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs
        ) {
            outTensorDescs.at(0) = inTensorDescs.at(POS_EMB_CAST(IN_QUERY));
            outTensorDescs.at(0).shape.dimNum = DIM_NUM_3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(POS_EMB_CAST(IN_QUERY)).shape.dims[0];
            outTensorDescs.at(0).shape.dims[DIM_1] = param.headNum;
            outTensorDescs.at(0).shape.dims[DIM_2] = param.headDim;
            outTensorDescs.at(1) = inTensorDescs.at(POS_EMB_CAST(IN_KEY));
            outTensorDescs.at(1).shape.dimNum = DIM_NUM_3;
            outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(POS_EMB_CAST(IN_KEY)).shape.dims[0];
            outTensorDescs.at(1).shape.dims[DIM_1] = param.kvHeadNum;
            outTensorDescs.at(1).shape.dims[DIM_2] = param.headDim;
            return atb::NO_ERROR;
        };
    }
    return atb::NO_ERROR;
}

int64_t AddSplitKV(atb::GraphParam &opGraph, const RotaryPositionEmbeddingParam &param)
{
    atb::Node splitQNode;
    atb::infer::SplitParam splitQParam;
    splitQParam.splitDim = DIM_LAST;
    splitQParam.splitNum = SPLIT_NUM_2;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitQParam, &splitQNode.operation));
    splitQNode.inTensorIds = {POS_EMB_CAST(IN_QUERY)};
    splitQNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_QCHUNK0), POS_EMB_CAST(INTERMEDIATE_QCHUNK1)};
    if (!param.isFA) {
        splitQNode.inTensorReshapeFuncs.resize(splitQNode.inTensorIds.size());
        splitQNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnsqueezeHeadNumHeadDim(oldShape, newShape, param.headNum, param.headDim);
        };
    }
    opGraph.nodes.push_back(splitQNode);

    atb::Node splitKNode;
    atb::infer::SplitParam splitKParam;
    splitKParam.splitDim = DIM_LAST;
    splitKParam.splitNum = SPLIT_NUM_2;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitKParam, &splitKNode.operation));
    splitKNode.inTensorIds = {POS_EMB_CAST(IN_KEY)};
    splitKNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_KCHUNK0), POS_EMB_CAST(INTERMEDIATE_KCHUNK1)};
    if (!param.isFA) {
        splitKNode.inTensorReshapeFuncs.resize(splitKNode.inTensorIds.size());
        splitKNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnsqueezeHeadNumHeadDim(oldShape, newShape, param.kvHeadNum, param.headDim);
        };
    }
    opGraph.nodes.push_back(splitKNode);
    return atb::NO_ERROR;
}

int64_t AddCatLV(atb::GraphParam &opGraph, const RotaryPositionEmbeddingParam &param)
{
    atb::Node cat1Node;
    atb::infer::ConcatParam cat1Param;
    cat1Param.concatDim = DIM_LAST;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(cat1Param, &cat1Node.operation));
    cat1Node.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_QOUT), POS_EMB_CAST(INTERMEDIATE_QCHUNK1)};
    cat1Node.outTensorIds = {POS_EMB_CAST(OUT_QUERY)};
    if (!param.isFA) {
        cat1Node.inTensorReshapeFuncs.resize(cat1Node.inTensorIds.size());
        cat1Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnsqueezeHeadNumHeadDim(oldShape, newShape, param.headNum, param.headDim / 2);  // 2: headDim被切分，所以除以2
        };
    }
    opGraph.nodes.push_back(cat1Node);

    atb::Node cat2Node;
    atb::infer::ConcatParam cat2Param;
    cat2Param.concatDim = DIM_LAST;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(cat2Param, &cat2Node.operation));
    cat2Node.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_KOUT), POS_EMB_CAST(INTERMEDIATE_KCHUNK1)};
    cat2Node.outTensorIds = {POS_EMB_CAST(OUT_KEY)};
    if (!param.isFA) {
        cat2Node.inTensorReshapeFuncs.resize(cat2Node.inTensorIds.size());
        cat2Node.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            UnsqueezeHeadNumHeadDim(oldShape, newShape, param.kvHeadNum, param.headDim / 2);  // 2: headDim被切分，所以除以2
        };
    }
    opGraph.nodes.push_back(cat2Node);
    return atb::NO_ERROR;
}

atb::Status RotaryPositionEmbedding(const RotaryPositionEmbeddingParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "RotaryPositionEmbedding";
    opGraph.inTensorNum = POS_EMB_IN_TENSOR_COUNT;
    opGraph.outTensorNum = POS_EMB_OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = param.rotaryType == HALF_ROTARY ?
                                     POS_EMB_INTERMEDIATE_TENSOR_2D_COUNT : POS_EMB_INTERMEDIATE_TENSOR_1D_COUNT;

    if (param.rotaryType == HALF_ROTARY) {
        // split q and k to half
        CHECK_OPERATION_STATUS_RETURN(AddSplitKV(opGraph, param));

        atb::Node ropeNode;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.ropeParam, &ropeNode.operation));
        ropeNode.inTensorIds = {POS_EMB_CAST(INTERMEDIATE_QCHUNK0), POS_EMB_CAST(INTERMEDIATE_KCHUNK0),
                                POS_EMB_CAST(IN_ROPE_COS), POS_EMB_CAST(IN_ROPE_SIN),
                                POS_EMB_CAST(IN_SEQLEN)};
        ropeNode.outTensorIds = {POS_EMB_CAST(INTERMEDIATE_QOUT), POS_EMB_CAST(INTERMEDIATE_KOUT)};
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        if (param.isFA) {
            ropeNode.inTensorReshapeFuncs.at(DIM_2) = &SqueezeRopeIntensor;
            ropeNode.inTensorReshapeFuncs.at(DIM_3) = &SqueezeRopeIntensor;
        } else {
            ropeNode.inTensorReshapeFuncs.at(DIM_0) = &SqueezeHeadNumHeadDim;
            ropeNode.inTensorReshapeFuncs.at(DIM_1) = &SqueezeHeadNumHeadDim;
            ropeNode.inTensorReshapeFuncs.at(DIM_2) = &SqueezeHeadNumHeadDim;
            ropeNode.inTensorReshapeFuncs.at(DIM_3) = &SqueezeHeadNumHeadDim;
        }
        opGraph.nodes.push_back(ropeNode);

        CHECK_OPERATION_STATUS_RETURN(AddCatLV(opGraph, param));
    } else {
        atb::Node ropeNode;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.ropeParam, &ropeNode.operation));
        ropeNode.inTensorIds = {POS_EMB_CAST(IN_QUERY), POS_EMB_CAST(IN_KEY), POS_EMB_CAST(IN_ROPE_COS),
                                POS_EMB_CAST(IN_ROPE_SIN), POS_EMB_CAST(IN_SEQLEN)};
        ropeNode.outTensorIds = {POS_EMB_CAST(OUT_QUERY), POS_EMB_CAST(OUT_KEY)};
        ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
        ropeNode.inTensorReshapeFuncs.at(DIM_2) = &SqueezeRopeIntensor;
        ropeNode.inTensorReshapeFuncs.at(DIM_3) = &SqueezeRopeIntensor;
        opGraph.nodes.push_back(ropeNode);
    }
    CHECK_OPERATION_STATUS_RETURN(AddInferShapeFunc(opGraph, param));

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));

    return atb::NO_ERROR;
}

}  // namespace common
}  // namespace atb_speed