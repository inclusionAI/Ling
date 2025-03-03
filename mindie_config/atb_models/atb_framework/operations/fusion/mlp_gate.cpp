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
#include "atb_speed/log.h"
#include "parallel_layer.h"
#include "mlp_gate.h"

namespace atb_speed {
namespace common {
template <class T> atb::Status AddmatmulUpNode(atb::Node &matmulUpNode, const MlpGateParam &param, T &config)
{
    atb::infer::LinearParam matmulUpParam = { false, param.transposeB, param.isBias };
    CREATE_OPERATION(matmulUpParam, &matmulUpNode.operation);
    if (param.isBias) {
        matmulUpNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_UP_ID, config.IN_BIAS_UP_ID };
    } else {
        matmulUpNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_UP_ID };
    }
    matmulUpNode.outTensorIds = { config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID };
    return atb::NO_ERROR;
}


template <class T>
atb::Status AddsplitNode(atb::Node &splitNode, atb::Node &matmulGateNode, const MlpGateParam &param, T &config,
    atb::GraphParam &opGraph)
{
    if (param.isPack) {
        atb::infer::SplitParam splitParam;
        splitParam.splitDim = -1; // 2: split最后一维
        splitParam.splitNum = 2;  // 2: 进行二等分
        CREATE_OPERATION(splitParam, &splitNode.operation);
        splitNode.inTensorIds = { config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID };
        splitNode.outTensorIds = { config.INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, config.INTERMEDIATE_SPLIT_OUT_ND_ID };
        opGraph.nodes.push_back(splitNode);
    } else {
        atb::infer::LinearParam matmulGateParam = { false, param.transposeB, param.isBias };
        CREATE_OPERATION(matmulGateParam, &matmulGateNode.operation);
        if (param.isBias) {
            matmulGateNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_GATE_ID,
                config.IN_BIAS_GATE_ID };
        } else {
            matmulGateNode.inTensorIds = { config.IN_HIDDENSTATES_ID, config.IN_WEIGHT_GATE_ID };
        }
        matmulGateNode.outTensorIds = { config.INTERMEDIATE_MATMUL_GATE_OUT_ND_ID };
        opGraph.nodes.push_back(matmulGateNode);
    }
    return atb::NO_ERROR;
}

template <class T> atb::Status AddactNode(atb::Node &actNode, const MlpGateParam &param, T &config)
{
    atb::infer::ActivationParam actParam;
    actParam.activationType = param.activationType;
    CREATE_OPERATION(actParam, &actNode.operation);
    actNode.inTensorIds = { config.INTERMEDIATE_MATMUL_GATE_OUT_ND_ID };
    actNode.outTensorIds = { config.INTERMEDIATE_ACTIVATION_OUT_ID };
    return atb::NO_ERROR;
}

template <class T> atb::Status AddmulNode(atb::Node &mulNode, const MlpGateParam &param, T &config)
{
    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CREATE_OPERATION(mulParam, &mulNode.operation);
    if (param.isPack) {
        mulNode.inTensorIds = { config.INTERMEDIATE_ACTIVATION_OUT_ID, config.INTERMEDIATE_SPLIT_OUT_ND_ID };
    } else {
        mulNode.inTensorIds = { config.INTERMEDIATE_ACTIVATION_OUT_ID, config.INTERMEDIATE_MATMUL_UP_OUT_ND_ID };
    }
    mulNode.outTensorIds = { config.INTERMEDIATE_MUL_OUT_ID };
    return atb::NO_ERROR;
}

template <class T> atb::Status AddmatmulDownNode(atb::Node &matmulDownNode, const MlpGateParam &param, T &config)
{
    atb_speed::common::ParallelParam linearParallelParam = { param.rank,       param.rankSize, 0,
                                                             nullptr,          param.isBias,   false,
                                                             param.transposeB, param.backend,  param.isBF16 };

    atb_speed::common::RowParallelLinear(linearParallelParam, &matmulDownNode.operation);
    if (param.isBias) {
        matmulDownNode.inTensorIds = { config.INTERMEDIATE_MUL_OUT_ID, config.IN_WEIGHT_DOWN_ID,
            config.IN_BIAS_DOWN_ID };
    } else {
        matmulDownNode.inTensorIds = { config.INTERMEDIATE_MUL_OUT_ID, config.IN_WEIGHT_DOWN_ID };
    }
    matmulDownNode.outTensorIds = { config.OUT_RESULT_ID };
    return atb::NO_ERROR;
}

template <class T> atb::Status MlpGateLayerBase(const MlpGateParam &param, atb::Operation **operation, T config)
{
    atb::GraphParam opGraph;
    opGraph.name = "MlpGateLayerBase";
    opGraph.inTensorNum = static_cast<unsigned>(config.inTensorNum);
    opGraph.outTensorNum = static_cast<unsigned>(config.outTensorNum);
    opGraph.internalTensorNum = static_cast<unsigned>(config.interTensorNum);
    atb::Node matmulUpNode;
    atb::Node splitNode;
    atb::Node matmulGateNode;
    atb::Node actNode;
    atb::Node mulNode;
    atb::Node matmulDownNode;
    CHECK_OPERATION_STATUS_RETURN(AddmatmulUpNode(matmulUpNode, param, config));
    opGraph.nodes.push_back(matmulUpNode);
    CHECK_OPERATION_STATUS_RETURN(AddsplitNode(splitNode, matmulGateNode, param, config, opGraph));
    CHECK_OPERATION_STATUS_RETURN(AddactNode(actNode, param, config));
    opGraph.nodes.push_back(actNode);
    CHECK_OPERATION_STATUS_RETURN(AddmulNode(mulNode, param, config));
    opGraph.nodes.push_back(mulNode);
    CHECK_OPERATION_STATUS_RETURN(AddmatmulDownNode(matmulDownNode, param, config));
    opGraph.nodes.push_back(matmulDownNode);
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}


atb::Status MlpGateLayer(const MlpGateParam &param, atb::Operation **operation)
{
    if (param.isBias && param.isPack) {
        return MlpGateLayerBase(param, operation, MlpGateWithPackAndBias(5, 1, 5, 5)); // 5:in 1:out 5:inter 5:node
    } else if (param.isBias) {
        return MlpGateLayerBase(param, operation, MlpGateWithBias(7, 1, 4, 5)); // 7:in 1:out 4:inter 5:node
    } else if (param.isPack) {
        return MlpGateLayerBase(param, operation, MlpGateWithPack(3, 1, 5, 5)); // 3:in 1:out 5:inter 5:node
    } else {
        return MlpGateLayerBase(param, operation, MlpGate(4, 1, 4, 5)); // 4:in 1:out 4:inter 5:node
    }
}
} // namespace common
} // namespace atb_speed
