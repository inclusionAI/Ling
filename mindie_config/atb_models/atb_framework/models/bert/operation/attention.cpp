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

#include <cmath>
#include "atb_speed/log.h"
#include "atb_speed/utils/check_util.h"
#include "operations/aclnn/ops/layer_norm_operation.h"
#include "models/bert/operation/attention.h"


namespace atb_speed::bert {

    enum AttentionTensorId : int {
        // input tensors
        IN_HIDDENSTATES = 0,
        IN_QLINEAR_WEIGHT,
        IN_QLINEAR_BIAS,
        IN_KLINEAR_WEIGHT,
        IN_KLINEAR_BIAS,
        IN_VLINEAR_WEIGHT,
        IN_VLINEAR_BIAS,
        IN_SELFOUTLINER_WEIGHT,
        IN_SELFOUTLINEAR_BIAS,
        IN_SELFOUTNORM_WEIGHT,
        IN_SELFOUTNORM_BIAS,
        // layer inputs
        IN_ATTENTIONMASK,
        IN_PASTKEY,
        IN_PASTVALUE,
        IN_TOKENOFFSET,
        IN_SEQLEN,
        IN_LAYERID,
        // output tensors
        OUT_SELF_RESULT,
        // intermediate tensors
        INTERMEDIATE_QLINER_OUT,
        INTERMEDIATE_KLINER_OUT,
        INTERMEDIATE_VLINER_OUT,
        INTERMEDIATE_SELFATTENTION_OUT,
        INTERMEDIATE_OUTLINEAR_OUT,
        INTERMEDIATE_OUTADD_OUT
    };

    static const uint64_t IN_TENSOR_COUNT = 17;
    static const uint64_t OUT_TENSOR_COUNT = 1;
    static const uint64_t INTERNAL_TENSOR_COUNT = 6;
    static const uint64_t SELF_ATTENTION_Q_INPUT_INDEX = 0;
    static const uint64_t SELF_ATTENTION_K_INPUT_INDEX = 1;
    static const uint64_t SELF_ATTENTION_V_INPUT_INDEX = 2;
    static const uint64_t SELF_ATTENTION_OUT_INDEX = 0;
    static const uint64_t SELF_ATTENTION_QKV_INPUT_SIZE = 3;
    static const uint64_t SELF_OUT_LINEAR_SIZE = 2;
    static const uint64_t SELF_OUT_ADD_SIZE = 3;

    int64_t CacheTensorReshapePAEncoder(atb::Node &selfAttentionKVCacheNode, const AttentionParam &param)
    {
        if (param.headNum == 0) {
            return atb::ERROR_INVALID_PARAM;
        }

        selfAttentionKVCacheNode.inTensorReshapeFuncs.resize(selfAttentionKVCacheNode.inTensorIds.size());
        selfAttentionKVCacheNode.inTensorReshapeFuncs.at(SELF_ATTENTION_Q_INPUT_INDEX) = [=](
            const atb::Dims &oldShape,
            atb::Dims &newShape
        ) -> void {
            newShape.dimNum = SELF_ATTENTION_QKV_INPUT_SIZE;
            size_t newShapeDimIndex = 0;
            size_t oldShapeDimIndex = 1;
            newShape.dims[newShapeDimIndex++] = CheckIntMulOverFlow(
                oldShape.dims[oldShapeDimIndex - 1], oldShape.dims[oldShapeDimIndex]);
            newShape.dims[newShapeDimIndex++] = param.headNum;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[++oldShapeDimIndex] / param.headNum;
        };
        selfAttentionKVCacheNode.inTensorReshapeFuncs.at(SELF_ATTENTION_K_INPUT_INDEX) = [=](
            const atb::Dims &oldShape,
            atb::Dims &newShape
        ) -> void {
            newShape.dimNum = SELF_ATTENTION_QKV_INPUT_SIZE;
            size_t newShapeDimIndex = 0;
            size_t oldShapeDimIndex = 1;
            newShape.dims[newShapeDimIndex++] = CheckIntMulOverFlow(
                oldShape.dims[oldShapeDimIndex - 1], oldShape.dims[oldShapeDimIndex]);
            newShape.dims[newShapeDimIndex++] = param.headNum;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[++oldShapeDimIndex] / param.headNum;
        };
        selfAttentionKVCacheNode.inTensorReshapeFuncs.at(SELF_ATTENTION_V_INPUT_INDEX) = [=](
            const atb::Dims &oldShape,
            atb::Dims &newShape
        ) -> void {
            newShape.dimNum = SELF_ATTENTION_QKV_INPUT_SIZE;
            size_t newShapeDimIndex = 0;
            size_t oldShapeDimIndex = 1;
            newShape.dims[newShapeDimIndex++] = CheckIntMulOverFlow(
                oldShape.dims[oldShapeDimIndex - 1], oldShape.dims[oldShapeDimIndex]);
            newShape.dims[newShapeDimIndex++] = param.headNum;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[++oldShapeDimIndex] / param.headNum;
        };

        return atb::NO_ERROR;
    }

    int64_t CacheTensorReshapeSelfAdd(atb::Node &outAddNode)
    {
        int64_t batchSize = 0;

        outAddNode.inTensorReshapeFuncs.resize(outAddNode.inTensorIds.size());
        outAddNode.inTensorReshapeFuncs.at(0) = [&](const atb::Dims &oldShape,
            atb::Dims &newShape) {
            newShape.dimNum = SELF_OUT_ADD_SIZE;
            size_t newShapeDimIndex = 0;
            size_t oldShapeDimIndex = 0;
            batchSize = oldShape.dims[oldShapeDimIndex];
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
        };
        outAddNode.inTensorReshapeFuncs.at(1) = [&](const atb::Dims &oldShape,
            atb::Dims &newShape) {
            if (batchSize == 0) {
                return atb::ERROR_INVALID_PARAM;
            }
            newShape.dimNum = SELF_OUT_ADD_SIZE;
            size_t newShapeDimIndex = 0;
            size_t oldShapeDimIndex = 0;
            newShape.dims[newShapeDimIndex++] = batchSize;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++] / batchSize;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            return atb::NO_ERROR;
        };

        return atb::NO_ERROR;
    }

    int64_t SelfAttention(atb::GraphParam &opGraph, const AttentionParam &param)
    {
        atb::Node qLinerNode;
        atb::Node kLinerNode;
        atb::Node vLinerNode;
        atb::Node selfAttentionKVCacheNode;

        // QKV Linear
        atb::infer::LinearParam selfLinearParam;
        selfLinearParam.hasBias = true;
        CREATE_OPERATION(selfLinearParam, &qLinerNode.operation);
        qLinerNode.inTensorIds = { IN_HIDDENSTATES, IN_QLINEAR_WEIGHT, IN_QLINEAR_BIAS };
        qLinerNode.outTensorIds = { INTERMEDIATE_QLINER_OUT };
        opGraph.nodes.push_back(qLinerNode);
        CREATE_OPERATION(selfLinearParam, &kLinerNode.operation);
        kLinerNode.inTensorIds = { IN_HIDDENSTATES, IN_KLINEAR_WEIGHT, IN_KLINEAR_BIAS };
        kLinerNode.outTensorIds = { INTERMEDIATE_KLINER_OUT };
        opGraph.nodes.push_back(kLinerNode);
        CREATE_OPERATION(selfLinearParam, &vLinerNode.operation);
        vLinerNode.inTensorIds = { IN_HIDDENSTATES, IN_VLINEAR_WEIGHT, IN_VLINEAR_BIAS };
        vLinerNode.outTensorIds = { INTERMEDIATE_VLINER_OUT };
        opGraph.nodes.push_back(vLinerNode);

        // Attention Mask
        atb::infer::SelfAttentionParam selfAttentionParam;
        selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
        selfAttentionParam.headNum = param.headNum;
        selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
        selfAttentionParam.qkScale = static_cast<float>(1.0 / sqrt(param.dk));
        CREATE_OPERATION(selfAttentionParam, &selfAttentionKVCacheNode.operation);
        selfAttentionKVCacheNode.inTensorIds = {
            INTERMEDIATE_QLINER_OUT,
            INTERMEDIATE_KLINER_OUT,
            INTERMEDIATE_VLINER_OUT,
            IN_ATTENTIONMASK,
            IN_SEQLEN
        };
        selfAttentionKVCacheNode.outTensorIds = { INTERMEDIATE_SELFATTENTION_OUT };
        CHECK_OPERATION_STATUS_RETURN(CacheTensorReshapePAEncoder(selfAttentionKVCacheNode, param));
        opGraph.nodes.push_back(selfAttentionKVCacheNode);

        return atb::NO_ERROR;
    }

    int64_t SelfOutput(atb::GraphParam &opGraph, const AttentionParam &param)
    {
        atb::Node outLinearNode;
        atb::Node outAddNode;
        atb::Node outNormNode;

        // Linear
        atb::infer::LinearParam outLinearParam;
        outLinearParam.hasBias = true;
        CREATE_OPERATION(outLinearParam, &outLinearNode.operation);
        outLinearNode.inTensorIds = { INTERMEDIATE_SELFATTENTION_OUT, IN_SELFOUTLINER_WEIGHT, IN_SELFOUTLINEAR_BIAS };
        outLinearNode.outTensorIds = { INTERMEDIATE_OUTLINEAR_OUT };
        outLinearNode.inTensorReshapeFuncs.resize(outLinearNode.inTensorIds.size());
        outLinearNode.inTensorReshapeFuncs.at(SELF_ATTENTION_OUT_INDEX) = [=](const atb::Dims &oldShape,
            atb::Dims &newShape) {
            newShape.dimNum = SELF_OUT_LINEAR_SIZE;
            size_t newShapeDimIndex = 0;
            size_t oldShapeDimIndex = 0;
            newShape.dims[newShapeDimIndex++] = oldShape.dims[oldShapeDimIndex++];
            newShape.dims[newShapeDimIndex++] = CheckIntMulOverFlow(
                oldShape.dims[oldShapeDimIndex], oldShape.dims[oldShapeDimIndex + 1]);
        };
        opGraph.nodes.push_back(outLinearNode);

        // Add
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &outAddNode.operation);
        outAddNode.inTensorIds = { IN_HIDDENSTATES, INTERMEDIATE_OUTLINEAR_OUT };
        outAddNode.outTensorIds = { INTERMEDIATE_OUTADD_OUT };
        CHECK_OPERATION_STATUS_RETURN(CacheTensorReshapeSelfAdd(outAddNode));
        opGraph.nodes.push_back(outAddNode);

        // Layer Norm
        atb_speed::common::AclNNLayerNormParam outNormParam;
        outNormParam.layerNormEps = param.layerNormEps;
        outNormParam.layerNormImplMode = param.layerNormImplMode;
        outNormNode.operation = new atb_speed::common::LayerNormOperation(
            "outNormNode",
            outNormParam
        );
        outNormNode.inTensorIds = { INTERMEDIATE_OUTADD_OUT, IN_SELFOUTNORM_WEIGHT, IN_SELFOUTNORM_BIAS };
        outNormNode.outTensorIds = { OUT_SELF_RESULT };
        opGraph.nodes.push_back(outNormNode);

        return atb::NO_ERROR;
    }

    atb::Status Attention(const AttentionParam &param, atb::Operation **operation)
    {
        ATB_SPEED_LOG_INFO(__func__ << " called, headNum: " << param.headNum);
        atb::GraphParam opGraph;
        opGraph.name = "Attention";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;

        CHECK_OPERATION_STATUS_RETURN(SelfAttention(opGraph, param));
        CHECK_OPERATION_STATUS_RETURN(SelfOutput(opGraph, param));

        CREATE_OPERATION(opGraph, operation);
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::bert
