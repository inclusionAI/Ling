
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
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"

#include "operations/fusion/lmhead/lmhead.h"

namespace atb_speed {
namespace common {

enum LmHeadTensorIdx : uint32_t {
    IN_HIDDENSTATES = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_DESCALE,
    IN_BIAS,
    IN_COMPRESS_IDX,
    IN_INDICES,
    OUT_LOGITS,
};

static const uint64_t IN_TENSOR_COUNT = 8;
static const uint64_t OUT_TENSOR_COUNT = 1;

template <class T>
int64_t AddSlice(atb::GraphParam &opGraph, const LmHeadParam &param, T &config)
{
    atb::Node sliceNode;
    atb::infer::SliceParam slicePassParam;
    if (param.unpadInputs) {
        slicePassParam.offsets = {
            0, CheckIntMulOverFlow(param.hiddenSizePerAttentionHead,
                                   param.linearParallelParam.tensorParallelInfo.rank)
        };
        slicePassParam.size = {-1, param.hiddenSizePerAttentionHead};
    } else {
        slicePassParam.offsets = {
            0, 0, CheckIntMulOverFlow(param.hiddenSizePerAttentionHead,
                                      param.linearParallelParam.tensorParallelInfo.rank)
        };
        slicePassParam.size = {-1, -1, param.hiddenSizePerAttentionHead};
    }
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(slicePassParam, &sliceNode.operation));
    if (param.gatherAhead) {
        sliceNode.inTensorIds = {config.INTERMEDIATE_GATHER_OUT};
    } else {
        sliceNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES};
    }
    sliceNode.outTensorIds = {config.INTERMEDIATE_SLICE_OUT};
    opGraph.nodes.push_back(sliceNode);

    return atb::NO_ERROR;
}

template <class T>
int64_t AddLinearParallel(
    atb::GraphParam &opGraph, const LmHeadParam &param,
    T &config, atb_speed::common::LinearParallelType parallelType)
{
    atb::Node linearParallelNode;
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(param.linearParallelParam, &linearParallelNode.operation));
    if (parallelType == ROW_PARALLEL) {
        linearParallelNode.inTensorIds = {
            config.INTERMEDIATE_SLICE_OUT, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_BIAS,
            LmHeadTensorIdx::IN_COMPRESS_IDX,
        };
    } else if (param.gatherAhead) {
        linearParallelNode.inTensorIds = {
            config.INTERMEDIATE_GATHER_OUT, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_BIAS,
            LmHeadTensorIdx::IN_COMPRESS_IDX,
        };
    } else {
        linearParallelNode.inTensorIds = {
            LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_WEIGHT, LmHeadTensorIdx::IN_SCALE,
            LmHeadTensorIdx::IN_OFFSET, LmHeadTensorIdx::IN_DESCALE, LmHeadTensorIdx::IN_BIAS,
            LmHeadTensorIdx::IN_COMPRESS_IDX,
        };
    }
    linearParallelNode.outTensorIds = {LmHeadTensorIdx::OUT_LOGITS};
    opGraph.nodes.push_back(linearParallelNode);

    return atb::NO_ERROR;
}

template <class T>
atb::Status CreateLmHead(
    const LmHeadParam &param, atb::Operation **operation, T config,
    atb_speed::common::LinearParallelType parallelType)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum =
        param.gatherAhead ? config.intermediateTensorCount : config.intermediateTensorCount - 1;
    opGraph.name = "LmHead";

    if (param.gatherAhead) {
        atb::Node gatherNode;
        atb::infer::GatherParam gatherParam;
        gatherParam.axis = param.unpadInputs ? 0 : 1;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(gatherParam, &gatherNode.operation));
        gatherNode.inTensorIds = {LmHeadTensorIdx::IN_HIDDENSTATES, LmHeadTensorIdx::IN_INDICES};
        gatherNode.outTensorIds = {config.INTERMEDIATE_GATHER_OUT};
        opGraph.nodes.push_back(gatherNode);
    }

    if (parallelType == ROW_PARALLEL) {
        CHECK_OPERATION_STATUS_RETURN(AddSlice(opGraph, param, config));
    }

    CHECK_OPERATION_STATUS_RETURN(AddLinearParallel(opGraph, param, config, parallelType));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(IN_HIDDENSTATES);
        CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(IN_HIDDENSTATES).shape.dimNum);
        auto dimLast = inTensorDescs.at(IN_HIDDENSTATES).shape.dimNum - 1;
        if (param.gatherAhead) {
            outTensorDescs.at(0).shape.dims[param.unpadInputs ? 0 : 1] = inTensorDescs.at(IN_INDICES).shape.dims[0];
        }
        if (parallelType == COLUMN_PARALLEL) {
            outTensorDescs.at(0).shape.dims[dimLast] = \
                CheckIntMulOverFlow(inTensorDescs.at(IN_WEIGHT).shape.dims[0],
                param.linearParallelParam.tensorParallelInfo.worldSize);
        } else {
            outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(IN_WEIGHT).shape.dims[0];
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

class LmHeadNoParallelConfig {
public:
    uint64_t nodeCount = 2;
    uint64_t intermediateTensorCount = 1;

    enum LmHeadNoParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

class LmHeadRowParallelConfig {
public:

    uint64_t nodeCount = 3;
    uint64_t intermediateTensorCount = 2;

    enum LmHeadRowParallelTensorIdx : uint32_t {
        INTERMEDIATE_SLICE_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_GATHER_OUT,
    };
};

class LmHeadColumnParallelConfig {
public:

    uint64_t nodeCount = 2;
    uint64_t intermediateTensorCount = 1;

    enum LmHeadColumnParallelTensorIdx : uint32_t {
        INTERMEDIATE_GATHER_OUT = LmHeadTensorIdx::OUT_LOGITS + 1,
        INTERMEDIATE_SLICE_OUT  // no usage
    };
};

atb::Status LmHead(const LmHeadParam &param, atb::Operation **operation)
{
    if (param.linearParallelParam.tensorParallelInfo.worldSize <= 1) {
        LmHeadNoParallelConfig lmHeadNoParallelConfig;
        return CreateLmHead(param, operation, lmHeadNoParallelConfig, UNDEFINED);
    } else if (param.linearParallelParam.parallelType == ROW_PARALLEL) {
        LmHeadRowParallelConfig lmHeadRowParallelConfig;
        return CreateLmHead(param, operation, lmHeadRowParallelConfig, ROW_PARALLEL);
    } else if (param.linearParallelParam.parallelType == COLUMN_PARALLEL) {
        LmHeadColumnParallelConfig lmHeadColumnParallelConfig;
        return CreateLmHead(param, operation, lmHeadColumnParallelConfig, COLUMN_PARALLEL);
    } else {
        ATB_SPEED_LOG_ERROR("LmHead operation doesn't support parallelType: "
            << param.linearParallelParam.parallelType
            << " Possible values are 1 (row parallel) or 2 (column parallel).");
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed