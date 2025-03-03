/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#include "moe_mlp.h"
#include <atb/atb_infer.h>
#include <memory>
#include "operations/fusion/moe/integrated_gmm.h"
#include "operations/aclnn/ops/finalize_routing_operation.h"
#include "operations/aclnn/ops/moe_init_routing_operation.h"
#include "operations/aclnn/ops/moe_compute_expert_tokens_operation.h"
#include "operations/aclnn/ops/moetoken_unpermute_operation.h"

namespace atb_speed {
namespace common {

static const uint64_t NUM2 = 2;
static const uint64_t NUM3 = 3;
static const uint64_t NUM4 = 4;

std::map<std::string, std::vector<std::string>> GetMoeMlpInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array", "in_selected_experts",
            "in_expert_weight"}
        },
    };
    return moeMlpInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetMoeMlpInterTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInterTensorCandidates = {
        {"default", {
            "intermediate_idx", "intermediate_weight_idx", "intermediate_dummy_zero",
            "intermediate_dummy_one", "intermediate_rev_idx", "intermediate_group_list",
            "intermediate_sorted_hiddenstates", "intermediate_rev_sorted_hiddenstates",
            "intermediate_matmul_gate_up_out", "intermediate_swish_out",
            "intermediate_mlp_out", "intermediate_mlp_out_weighted",
            "intermediate_sorted_weight"}
        },
        {"enableFusedRouting", {
            "intermediate_idx", "intermediate_group_list", "intermediate_group_list_int64",
            "intermediate_sorted_hiddenstates", "intermediate_matmul_gate_up_out", "intermediate_swish_out",
            "intermediate_mlp_out"}
        },
        {"disable_swiglu", {
            "intermediate_matmul_gate_out", "intermediate_matmul_up_out",
            "intermediate_swish_out_internal"}
        }
    };
    return moeMlpInterTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const MoeMlpParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto moeMlpInTensorCandidates = GetMoeMlpInTensorCandidates();
    auto moeMlpInterTensorCandidates = GetMoeMlpInterTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {};
    std::vector<std::string> outTensorList = {"out_moe_mlp_result"};

    AddTensorToList(moeMlpInTensorCandidates, "default", inTensorList);
    if (param.enableFusedRouting) {
        AddTensorToList(moeMlpInterTensorCandidates, "enableFusedRouting", interTensorList);
    } else {
        AddTensorToList(moeMlpInterTensorCandidates, "default", interTensorList);
    }
    if (!param.supportSwiGLU) {
        AddTensorToList(moeMlpInterTensorCandidates, "disable_swiglu", interTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}

// Step 1: hidden state permutation
atb::Status CreateInitRouting(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &initRoutingNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeInitRoutingParam initRoutingParam;
    initRoutingParam.topkNum = param.topk;
    initRoutingParam.expertNum = param.numOfExperts;
    initRoutingNode.operation = new atb_speed::common::MoeInitRoutingOperation("MoeInitRoutingOperation",
                                                                               initRoutingParam);
    initRoutingNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                              GetTensorIdx(tensorMap, "in_selected_experts")};
    initRoutingNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"),
                               GetTensorIdx(tensorMap, "intermediate_idx"),
                               GetTensorIdx(tensorMap, "intermediate_group_list")};
    ATB_SPEED_LOG_DEBUG("InitRouting calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateCast(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &castNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_INT64;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(castParam, &castNode.operation));
    castNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list")};
    castNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_group_list_int64")};
    ATB_SPEED_LOG_DEBUG("Cast calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGating(
    std::map<std::string, uint32_t> &tensorMap, const MoeMlpParam &param, std::shared_ptr<int64_t> batchDimPtr,
    atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatingNode = opGraph.nodes.at(nodeId++);
    CHECK_PARAM_NE(param.topk, 0);
    CHECK_PARAM_NE(param.numOfExperts, 0);
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = param.topk;
    gatingParam.cumSumNum = param.numOfExperts;
    gatingParam.cumSumInt64 = true;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &gatingNode.operation));
    gatingNode.inTensorIds = {
        GetTensorIdx(tensorMap, "in_selected_experts"), GetTensorIdx(tensorMap, "in_expert_array")};
    gatingNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_idx"), GetTensorIdx(tensorMap, "intermediate_group_list"),
        GetTensorIdx(tensorMap, "intermediate_weight_idx")};
    gatingNode.inTensorReshapeFuncs.resize(gatingNode.inTensorIds.size());
    gatingNode.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("Gating calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGather0(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatherNode0 = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode0.operation));
    gatherNode0.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"), GetTensorIdx(tensorMap, "intermediate_idx")};
    gatherNode0.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates")};
    ATB_SPEED_LOG_DEBUG("Gather0 calculation success");
    return atb::NO_ERROR;
}

// Step 2: grouped matmul calculation & activation
atb::Status CreateGmm(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param, size_t &nodeId)
{
    atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::IntegratedGmmParam gmmParam;
    gmmParam.hasBias = param.hasBias;
    gmmParam.isUp = true;
    gmmParam.moeLinearQuantType = param.moeLinearQuantType;
    gmmParam.packQuantType = param.packQuantType;
    gmmParam.transposeB = param.gateUpTransposeB;
    gmmParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateIntegratedGmmOperation(gmmParam, &gmmNode.operation));
    gmmNode.inTensorIds = {};
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_sorted_hiddenstates"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_bias_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"));
    gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"));
    if (param.enableFusedRouting) {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    gmmNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivation(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNode.operation));
    swishNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    swishNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSplit(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::infer::SplitParam splitParam = {1, 2};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(splitParam, &splitNode.operation));
    splitNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_up_out")};
    splitNode.outTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_matmul_gate_out"), GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    ATB_SPEED_LOG_DEBUG("Split calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateActivationO(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &swishNodeO = opGraph.nodes.at(nodeId++);
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(activationParam, &swishNodeO.operation));
    swishNodeO.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_matmul_gate_out")};
    swishNodeO.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out_internal")};
    ATB_SPEED_LOG_DEBUG("Activation calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMul(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &mulNode.operation));
    mulNode.inTensorIds = {
        GetTensorIdx(tensorMap, "intermediate_swish_out_internal"),
        GetTensorIdx(tensorMap, "intermediate_matmul_up_out")};
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out")};
    ATB_SPEED_LOG_DEBUG("ElewiseMul0 calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGmm1(std::map<std::string, uint32_t> &tensorMap,
    atb::GraphParam &opGraph, const MoeMlpParam &param, size_t &nodeId)
{
    atb::Node &gmmDownNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::IntegratedGmmParam gmmParam;
    gmmParam.hasBias = param.hasBias;
    gmmParam.isUp = false;
    gmmParam.moeLinearQuantType = param.moeLinearQuantType;
    gmmParam.packQuantType = param.packQuantType;
    gmmParam.transposeB = param.downTransposeB;
    gmmParam.outDataType = param.isBF16 ? ACL_BF16 : ACL_FLOAT16;
    CHECK_OPERATION_STATUS_RETURN(CreateIntegratedGmmOperation(gmmParam, &gmmDownNode.operation));
    gmmDownNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out")};
    gmmDownNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_swish_out"),
                               GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
                               GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert")};
    if (param.enableFusedRouting) {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list_int64"));
    } else {
        gmmDownNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_group_list"));
    }
    ATB_SPEED_LOG_DEBUG("GMM calculation success");
    return atb::NO_ERROR;
}

// Step 3: hidden state reduction
atb::Status CreateMoeTokenUnpermute(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph,  size_t &nodeId)
{
    atb::Node &unpermuteNode = opGraph.nodes.at(nodeId++);
    unpermuteNode.operation = new atb_speed::common::MoeTokenUnpermuteOperation("MoeTokenUnpermuteNode");
    unpermuteNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    unpermuteNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                GetTensorIdx(tensorMap, "intermediate_idx"),
                                GetTensorIdx(tensorMap, "in_expert_weight")};
    ATB_SPEED_LOG_DEBUG("UnpermuteNode calculation success");
    return atb::NO_ERROR;
}

// Op5 - Gather1
atb::Status CreateGather1(std::map<std::string, uint32_t> &tensorMap,
    std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatherNode1 = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode1.operation));
    gatherNode1.inTensorIds = {GetTensorIdx(tensorMap, "in_expert_weight"),
                               GetTensorIdx(tensorMap, "intermediate_weight_idx")};
    gatherNode1.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sorted_weight")};
    gatherNode1.inTensorReshapeFuncs.resize(gatherNode1.inTensorIds.size());
    gatherNode1.inTensorReshapeFuncs[0] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    ATB_SPEED_LOG_DEBUG("Gather1 calculation success");
    return atb::NO_ERROR;
}

// Op6 - ElewiseMul1
atb::Status CreateElewiseMul1(std::map<std::string, uint32_t> &tensorMap,
    std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &weightMulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &weightMulNode.operation));
    weightMulNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out"),
                                 GetTensorIdx(tensorMap, "intermediate_sorted_weight")};
    weightMulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out_weighted")};
    weightMulNode.inTensorReshapeFuncs.resize(weightMulNode.inTensorIds.size());
    weightMulNode.inTensorReshapeFuncs[1] = [batchDimPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    ATB_SPEED_LOG_DEBUG("ElewiseMul1 calculation success");
    return atb::NO_ERROR;
}

// Op7 - Argsort
atb::Status CreateArgsort(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &argsortNode = opGraph.nodes.at(nodeId++);
    atb::infer::GatingParam gatingParam;
    gatingParam.topkExpertNum = 1;
    gatingParam.cumSumNum = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatingParam, &argsortNode.operation));
    argsortNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_weight_idx"),
                               GetTensorIdx(tensorMap, "in_expert_array")};
    argsortNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_dummy_zero"),
                                GetTensorIdx(tensorMap, "intermediate_dummy_one"),
                                GetTensorIdx(tensorMap, "intermediate_rev_idx")};
    ATB_SPEED_LOG_DEBUG("Argsort calculation success");
    return atb::NO_ERROR;
}

// Op8 - Gather2
atb::Status CreateGather2(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &gatherNode2 = opGraph.nodes.at(nodeId++);
    atb::infer::GatherParam gatherParam;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(gatherParam, &gatherNode2.operation));
    gatherNode2.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_mlp_out_weighted"),
                               GetTensorIdx(tensorMap, "intermediate_rev_idx")};
    gatherNode2.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_rev_sorted_hiddenstates")};
    ATB_SPEED_LOG_DEBUG("Cather2 calculation success");
    return atb::NO_ERROR;
}

// Op9 - Reduction
atb::Status CreateReduction(std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param, std::shared_ptr<int64_t> batchDimPtr, atb::GraphParam &opGraph, size_t &nodeId)
{
    CHECK_PARAM_NE(param.topk, 0);
    atb::Node &reduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &reduceNode.operation));
    reduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_rev_sorted_hiddenstates")};
    reduceNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_mlp_result")};
    reduceNode.inTensorReshapeFuncs.resize(reduceNode.inTensorIds.size());
    reduceNode.inTensorReshapeFuncs[0] = [batchDimPtr, param](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // 3:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0] / param.topk;
        newShape.dims[1] = param.topk;
        newShape.dims[2] = oldShape.dims[1]; // 2:the third dimension of the new shape
    };
    ATB_SPEED_LOG_DEBUG("Reduction calculation success");
    return atb::NO_ERROR;
}

int64_t CalculateNodeSize(const MoeMlpParam &param)
{
    int64_t nodeCount = 10;
    if (!param.supportSwiGLU) {
        nodeCount += NUM2;
    }
    if (param.enableFusedRouting) {
        nodeCount -= NUM4;
    }
    return nodeCount;
}

atb::Status CreateActivationBlock(std::map<std::string, uint32_t> &tensorMap,
    const MoeMlpParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    if (param.supportSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(CreateActivation(tensorMap, opGraph, nodeId));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSplit(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationO(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul(tensorMap, opGraph, nodeId));
    }

    ATB_SPEED_LOG_DEBUG("ActivationBlock calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateMoeMlpOperation(const MoeMlpParam &param, atb::Operation **operation)
{
    std::shared_ptr<int64_t> batchDimPtr = std::make_shared<int64_t>(0);
    atb::GraphParam opGraph;
    opGraph.name = "MoeMlp";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum" << opGraph.internalTensorNum);

    uint64_t nodeCount = CalculateNodeSize(param);
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    if (param.enableFusedRouting) {
        CHECK_OPERATION_STATUS_RETURN(CreateInitRouting(tensorMap, param, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateCast(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateMoeTokenUnpermute(tensorMap, opGraph, nodeId));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateGating(tensorMap, param, batchDimPtr, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGather0(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm(tensorMap, opGraph, param, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateActivationBlock(tensorMap, param, nodeId, opGraph));
        CHECK_OPERATION_STATUS_RETURN(CreateGmm1(tensorMap, opGraph, param, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGather1(tensorMap, batchDimPtr, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMul1(tensorMap, batchDimPtr, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateArgsort(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGather2(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateReduction(tensorMap, param, batchDimPtr, opGraph, nodeId));
    }

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };
    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
}
} // namespace atb_speed