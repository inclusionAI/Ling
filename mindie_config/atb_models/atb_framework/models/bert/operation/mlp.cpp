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

#include "atb_speed/log.h"
#include "operations/aclnn/ops/gelu_operation.h"
#include "operations/aclnn/ops/layer_norm_operation.h"
#include "models/bert/operation/mlp.h"


namespace atb_speed::bert {

    enum MlpTensorId : int {
        // input tensors
        IN_HIDDENSTATES = 0,
        IN_INTERLINEAR_WEIGHT,
        IN_INTERLINEAR_BIAS,
        IN_OUTLINEAR_WEIGHT,
        IN_OUTLINEAR_BIAS,
        IN_NORM_WEIGHT,
        IN_NORM_BIAS,
        // output tensors
        OUT_FEEDFORWARD_RESULT,
        // intermediate tensors
        INTERMEDIATE_INTERLINEAR_OUT,
        INTERMEDIATE_ACT_OUT,
        INTERMEDIATE_OUTLINEAR_OUT,
        INTERMEDAITE_ADD_OUT
    };

    static const uint64_t IN_TENSOR_COUNT = 7;
    static const uint64_t OUT_TENSOR_COUNT = 1;
    static const uint64_t INTERNAL_TENSOR_COUNT = 4;
    static const uint64_t NODE_COUNT = 5;

    atb::Status Mlp(const MlpParam &param, atb::Operation **operation)
    {
        ATB_SPEED_LOG_INFO(__func__ << " called");
        atb::GraphParam opGraph;
        opGraph.name = "Mlp";
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.outTensorNum = OUT_TENSOR_COUNT;
        opGraph.internalTensorNum = INTERNAL_TENSOR_COUNT;
        opGraph.nodes.resize(NODE_COUNT);

        size_t nodeId = 0;

        // Linear
        auto &interLinearNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam interLinearParam;
        interLinearParam.hasBias = true;
        interLinearParam.transposeA = false;
        interLinearParam.transposeB = true;
        CREATE_OPERATION(interLinearParam, &interLinearNode.operation);
        interLinearNode.inTensorIds = { IN_HIDDENSTATES, IN_INTERLINEAR_WEIGHT, IN_INTERLINEAR_BIAS };
        interLinearNode.outTensorIds = { INTERMEDIATE_INTERLINEAR_OUT };

        // Gelu
        auto &interActNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::AclNNGeluParam interActParam;
        interActParam.geluApproximate = param.geluApproximate;
        interActNode.operation = new atb_speed::common::GeluOperation("interActNode", interActParam);
        interActNode.inTensorIds = { INTERMEDIATE_INTERLINEAR_OUT };
        interActNode.outTensorIds = { INTERMEDIATE_ACT_OUT };

        // Linear
        auto &outLinearNode = opGraph.nodes.at(nodeId++);
        atb::infer::LinearParam outLinearParam;
        outLinearParam.hasBias = true;
        outLinearParam.transposeA = false;
        outLinearParam.transposeB = true;
        CREATE_OPERATION(outLinearParam, &outLinearNode.operation);
        outLinearNode.inTensorIds = { INTERMEDIATE_ACT_OUT, IN_OUTLINEAR_WEIGHT, IN_OUTLINEAR_BIAS };
        outLinearNode.outTensorIds = { INTERMEDIATE_OUTLINEAR_OUT };

        // Add
        auto &outAddNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ELEWISE_ADD;
        CREATE_OPERATION(addParam, &outAddNode.operation);
        outAddNode.inTensorIds = { INTERMEDIATE_OUTLINEAR_OUT, IN_HIDDENSTATES };
        outAddNode.outTensorIds = { INTERMEDAITE_ADD_OUT };

        // Layer Norm
        auto &outNormNode = opGraph.nodes.at(nodeId++);
        atb_speed::common::AclNNLayerNormParam outNormParam;
        outNormParam.layerNormEps = param.layerNormEps;
        outNormParam.layerNormImplMode = param.layerNormImplMode;
        outNormNode.operation = new atb_speed::common::LayerNormOperation("outNormNode", outNormParam);
        outNormNode.inTensorIds = { INTERMEDAITE_ADD_OUT, IN_NORM_WEIGHT, IN_NORM_BIAS };
        outNormNode.outTensorIds = { OUT_FEEDFORWARD_RESULT };

        CREATE_OPERATION(opGraph, operation);
        return atb::NO_ERROR;
    }

}  // namespace atb_speed::berts
