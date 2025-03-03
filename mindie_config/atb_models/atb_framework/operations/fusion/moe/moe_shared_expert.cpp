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
#include <memory>
#include "atb_speed/log.h"
#include "moe_shared_expert.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetSharedExpertInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> sharedExpertInTensorCandidates = {
        {"default", {
            "in_hiddenstatus",
            "in_mlp_gate_up_weight", "in_mlp_gate_up_bias", "in_mlp_gate_up_descale",
            "in_mlp_gate_up_offset", "in_mlp_gate_up_scale", "in_mlp_gate_up_compress_idx",
            "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
            "in_mlp_down_scale", "in_mlp_down_compress_idx",
            "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
            "in_shared_expert_gate_offset", "in_shared_expert_gate_scale","in_shared_expert_gate_compress_idx"
            }
        }
    };
    return sharedExpertInTensorCandidates;
};

std::map<std::string, std::vector<std::string>> GetSharedExpertIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> sharedExpertIntermediateTensorCandidates = {
        {"default",
            {"intermediate_matmul_gate_up_out", "intermediate_hiddenstatus"}
        },
        {"not_support_swiglu",
            {"intermediate_matmul_gate_out", "intermediate_swish_out", "intermediate_matmul_up_out"}
        },
        {"has_shared_expert_gate",
            {
            "intermediate_shared_expert_out", "intermediate_shared_expert_gate_logits",
             "intermediate_shared_expert_weight"
            }
        }
    };
    return sharedExpertIntermediateTensorCandidates;
};

std::map<std::string, uint32_t> ConstructTensorMap(const SharedExpertParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto sharedExpertInTensorCandidates = GetSharedExpertInTensorCandidates();
    auto sharedExpertIntermediateTensorCandidates = GetSharedExpertIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(sharedExpertInTensorCandidates, "default", inTensorList);
    AddTensorToList(sharedExpertIntermediateTensorCandidates, "default", intermediateTensorList);
    // 如果不支持SwiGLU
    if (!param.supportSwiGLU) {
        AddTensorToList(sharedExpertIntermediateTensorCandidates,
                        "not_support_swiglu", intermediateTensorList);
    }
    // 如果支持shared expert gate
    if (param.hasSharedExpertGate) {
        AddTensorToList(sharedExpertIntermediateTensorCandidates,
                        "has_shared_expert_gate", intermediateTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

// Expert Ops
atb::Status CreateLinear(const SharedExpertParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node linearNode;
    atb_speed::common::FusionLinearParam linearParam;
    linearParam.hasBias = false;
    linearParam.isBF16 = param.isBF16;
    linearParam.transposeType = param.mlpLinearTransposeType.at(SHARED_MOE_GATE_LINEAR_INDEX);
    linearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.mlpLinearQuantType[SHARED_MOE_GATE_LINEAR_INDEX], false);
    linearParam.quantGroupSize = 0;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(linearParam, &linearNode.operation));
    linearNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_mlp_gate_up_weight"),
        GetTensorIdx(tensorMap, "in_mlp_gate_up_scale"),
        GetTensorIdx(tensorMap, "in_mlp_gate_up_offset"),
        GetTensorIdx(tensorMap, "in_mlp_gate_up_descale"),
        GetTensorIdx(tensorMap, "in_mlp_gate_up_bias"),
        GetTensorIdx(tensorMap, "in_mlp_gate_up_compress_idx"),
    };
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    opGraph.nodes.push_back(linearNode);
    ATB_SPEED_LOG_DEBUG("Gate up projection calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSplit(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node splitNode;
    atb::infer::SplitParam splitParam = {1, 2, {}};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    splitNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_out"),
                            GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    opGraph.nodes.push_back(splitNode);
    ATB_SPEED_LOG_DEBUG("Split calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivation(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node swishNode;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(activationParam, &swishNode.operation);
    swishNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_out")};
    swishNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    opGraph.nodes.push_back(swishNode);
    ATB_SPEED_LOG_DEBUG("Swish calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMul1(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node mulNode;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                           GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_hiddenstatus")};
    opGraph.nodes.push_back(mulNode);
    ATB_SPEED_LOG_DEBUG("ElewiseMul1 success");
    return atb::NO_ERROR;
}

// SwiGlu = Split + Activation + ElewiseMatmul
atb::Status CreateSwiGLU(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node swishNode;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNode.operation));
    swishNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swishNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_hiddenstatus")};
    opGraph.nodes.push_back(swishNode);
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateLinearDown(const SharedExpertParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node linearDownNode;
    atb_speed::common::FusionLinearParam linearDownParam;
    linearDownParam.hasBias = false;
    linearDownParam.isBF16 = param.isBF16;
    linearDownParam.transposeType = param.mlpLinearTransposeType.at(SHARED_MOE_DOWN_LINEAR_INDEX);
    linearDownParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.mlpLinearQuantType[SHARED_MOE_DOWN_LINEAR_INDEX], false);
    linearDownParam.quantGroupSize = 0;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(linearDownParam, &linearDownNode.operation));
    linearDownNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_mlp_down_weight"),
        GetTensorIdx(tensorMap, "in_mlp_down_scale"),
        GetTensorIdx(tensorMap, "in_mlp_down_offset"),
        GetTensorIdx(tensorMap, "in_mlp_down_descale"),
        GetTensorIdx(tensorMap, "in_mlp_down_bias"),
        GetTensorIdx(tensorMap, "in_mlp_down_compress_idx"),
    };
    if (param.hasSharedExpertGate) {
        linearDownNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shared_expert_out")};
    } else {
        linearDownNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    }
    opGraph.nodes.push_back(linearDownNode);
    ATB_SPEED_LOG_DEBUG("Projection down success");
    return atb::NO_ERROR;
}

// Expert Gate Ops

atb::Status CreateSharedExpertGate(const SharedExpertParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sharedexpertgateNode;
    atb_speed::common::FusionLinearParam linearParam;
    linearParam.hasBias = false;
    linearParam.hasBias = false;
    linearParam.isBF16 = param.isBF16;
    linearParam.quantType = GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.mlpLinearQuantType[SHARED_MOE_SHAREGATE_LINEAR_INDEX], false);
    linearParam.quantGroupSize = 0;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(linearParam, &sharedexpertgateNode.operation));
    sharedexpertgateNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_hiddenstatus"),
        GetTensorIdx(tensorMap, "in_shared_expert_gate_weight"),
        GetTensorIdx(tensorMap, "in_shared_expert_gate_scale"),
        GetTensorIdx(tensorMap, "in_shared_expert_gate_offset"),
        GetTensorIdx(tensorMap, "in_shared_expert_gate_descale"),
        GetTensorIdx(tensorMap, "in_shared_expert_gate_bias"),
        GetTensorIdx(tensorMap, "in_shared_expert_gate_compress_idx"),
    };
    sharedexpertgateNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shared_expert_gate_logits")};
    opGraph.nodes.push_back(sharedexpertgateNode);
    ATB_SPEED_LOG_DEBUG("Shared Expert Gate calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationSigmoid(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sigmoidNode;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SIGMOID;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &sigmoidNode.operation));
    sigmoidNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shared_expert_gate_logits")};
    sigmoidNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_shared_expert_weight")};
    opGraph.nodes.push_back(sigmoidNode);
    ATB_SPEED_LOG_DEBUG("Activation Sigmoid calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMul2(atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node sigmoidMulNode;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &sigmoidMulNode.operation));
    sigmoidMulNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_shared_expert_out"),
                                  GetTensorIdx(tensorMap, "intermediate_shared_expert_weight")};
    sigmoidMulNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
    opGraph.nodes.push_back(sigmoidMulNode);
    ATB_SPEED_LOG_DEBUG("ElewiseMul2 calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationBlock(const SharedExpertParam &param,
    atb::GraphParam &opGraph, std::map<std::string, uint32_t> &tensorMap)
{
    if (param.supportSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(CreateSwiGLU(opGraph, tensorMap));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSplit(opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(CreateActivation(opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul1(opGraph, tensorMap));
    }

    ATB_SPEED_LOG_DEBUG("ActivationBlock calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSharedExpertOperation(const SharedExpertParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.name = "SharedExpert";

    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    CHECK_OPERATION_STATUS_RETURN(CreateLinear(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(param, opGraph, tensorMap));
    CHECK_OPERATION_STATUS_RETURN(CreateLinearDown(param, opGraph, tensorMap));

    if (param.hasSharedExpertGate) {
        CHECK_OPERATION_STATUS_RETURN(CreateSharedExpertGate(param, opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationSigmoid(opGraph, tensorMap));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul2(opGraph, tensorMap));
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(GetTensorIdx(tensorMap, "in_hiddenstatus"));
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed