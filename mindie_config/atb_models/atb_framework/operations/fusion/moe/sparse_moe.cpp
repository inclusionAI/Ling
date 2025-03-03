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
#include "sparse_moe.h"
#include <atb/atb_infer.h>
#include <memory>
#include "moe_mlp.h"
#include "device_limited_routing.h"
#include "operations/aclnn/ops/moe_topk_softmax_operation.h"
#include "operations/aclnn/ops/vector_norm_operation.h"
#include "operations/aclnn/ops/std_operation.h"

namespace atb_speed {
namespace common {

const uint64_t NODE_SIZE_INCR_NORMALIZATION  = 2;

std::map<std::string, std::vector<std::string>> GetSparseMoeInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> moeMlpInTensorCandidates = {
        {"default", {
            "in_hiddenstates", "in_gate_weight", "in_gate_bias", "in_gate_descale", "in_gate_offset",
            "in_gate_scale", "in_gate_compress_idx", "in_mlp_gateup_weight_expert", "in_mlp_gateup_bias_expert",
            "in_mlp_gateup_descale_expert", "in_mlp_gateup_offset_expert", "in_mlp_gateup_scale_expert",
            "in_mlp_gateup_compress_idx_expert", "in_mlp_down_weight_expert",
            "in_mlp_down_bias_expert", "in_mlp_down_descale_expert", "in_mlp_down_offset_expert",
            "in_mlp_down_scale_expert", "in_mlp_down_compress_idx_expert", "in_expert_array",
            "in_expert_group", "in_one_hot", "in_zero_hot"}
        },
    };
    return moeMlpInTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const SparseMoeParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &interTensorNum)
{
    auto moeMlpInTensorCandidates = GetSparseMoeInTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> interTensorList = {
        "intermediate_router_logits", "intermediate_router_weights", "intermediate_router_weights_topk",
        "intermediate_selected_experts"};
    std::vector<std::string> outTensorList = {"out_moe_rout"};
 
    AddTensorToList(moeMlpInTensorCandidates, "default", inTensorList);
    if (param.useStdNorm) {
        interTensorList.push_back("intermediate_router_logits_std");
    }
    if (param.processLogits == "normalization" || param.processLogits == "norm") {
        interTensorList.push_back("intermediate_router_weights_topk_reduced");
        interTensorList.push_back("intermediate_router_weights_topk_sumed");
    } else if (param.processLogits == "scaling") {
        interTensorList.push_back("intermediate_router_weights_topk_reduced");
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    interTensorNum = interTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, interTensorList);
}


atb::Status CreateSparseMoemoeGate(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    FusionLinearParam moeGateParam;
    moeGateParam.transposeType = common::TRANSPOSE;
    moeGateParam.hasBias = param.rounterHasBias;
    moeGateParam.isBF16 = param.isBF16;
    moeGateParam.quantType = atb_speed::common::GetLinearQuantType(
        param.denseQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.denseQuantType,
        param.moeLinearQuantType[SparseMoeIdx::ROUTER_IDX], false);
    moeGateParam.quantGroupSize = 0;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(moeGateParam, &linearNode.operation));
    linearNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                              GetTensorIdx(tensorMap, "in_gate_weight"),
                              GetTensorIdx(tensorMap, "in_gate_scale"),
                              GetTensorIdx(tensorMap, "in_gate_offset"),
                              GetTensorIdx(tensorMap, "in_gate_descale"),
                              GetTensorIdx(tensorMap, "in_gate_bias"),
                              GetTensorIdx(tensorMap, "in_gate_compress_idx")};
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    ATB_SPEED_LOG_DEBUG("Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoeStd(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &stdNode = opGraph.nodes.at(nodeId++);
    stdNode.operation = new atb_speed::common::StdOperation("SparseMoeStdNode");
    stdNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    stdNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits_std")};
    ATB_SPEED_LOG_DEBUG("Router logits std calculation success");
    return atb::NO_ERROR;
}


atb::Status CreateSparseMoeNorm(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &normNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam normParam;
    normParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(normParam, &normNode.operation));
    normNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits"),
        GetTensorIdx(tensorMap, "intermediate_router_logits_std")};
    normNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    ATB_SPEED_LOG_DEBUG("Router weights norm calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoesoftMax(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &softMaxNode = opGraph.nodes.at(nodeId++);
    atb::infer::SoftmaxParam softMaxParam;
    softMaxParam.axes = param.axes;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(softMaxParam, &softMaxNode.operation));
    softMaxNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    softMaxNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    ATB_SPEED_LOG_DEBUG("Router weights calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoetopK(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &topKNode = opGraph.nodes.at(nodeId++);
    atb::infer::SortParam topKParam;
    topKParam.num = param.num;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(topKParam, &topKNode.operation));
    topKNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    topKNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk"),
                             GetTensorIdx(tensorMap, "intermediate_selected_experts")};
    ATB_SPEED_LOG_DEBUG("Expert selection success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoereduce(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &reduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::ReduceParam reduceParam;
    reduceParam.reduceType = atb::infer::ReduceParam::ReduceType::REDUCE_SUM;
    reduceParam.axis = {1};
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(reduceParam, &reduceNode.operation));
    reduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk")};
    reduceNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_sumed")};
    ATB_SPEED_LOG_DEBUG("Reduce sum calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseMoedivide(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &divideNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam divideParam;
    divideParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(divideParam, &divideNode.operation));
    divideNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk"),
                              GetTensorIdx(tensorMap, "intermediate_router_weights_topk_sumed")};
    divideNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced")};
    divideNode.inTensorReshapeFuncs.resize(divideNode.inTensorIds.size());
    divideNode.inTensorReshapeFuncs[1] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // 2:number of dimensions of the new shape
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = 1;
    };
    ATB_SPEED_LOG_DEBUG("Router weights calculated success");
    return atb::NO_ERROR;
}

atb::Status CreateElewiseMuls(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.mulsParam.varAttr = param.routedScalingFactor;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(elewiseParam, &mulNode.operation));
    mulNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk")};
    mulNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced")};
    ATB_SPEED_LOG_DEBUG("ElewiseMuls calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateDeviceLimitedRouting(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &deviceLimitedNode = opGraph.nodes.at(nodeId++);
    atb_speed::deviceLimitedRouting::DeviceLimitedRoutingParam deviceLimitedRoutingParam;
    deviceLimitedRoutingParam.numOfExperts = param.numOfExperts;
    deviceLimitedRoutingParam.numOfGroups = param.numOfGroups;
    deviceLimitedRoutingParam.topkGroups = param.topkGroups;
    atb_speed::deviceLimitedRouting::CreateDeviceLimitedRoutingOperation(deviceLimitedRoutingParam,
                                                                         &deviceLimitedNode.operation);
    deviceLimitedNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights"),
                                     GetTensorIdx(tensorMap, "in_expert_group"),
                                     GetTensorIdx(tensorMap, "in_one_hot"),
                                     GetTensorIdx(tensorMap, "in_zero_hot")};
    deviceLimitedNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    ATB_SPEED_LOG_DEBUG("Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateGroupOperation(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &deviceLimitedNode = opGraph.nodes.at(nodeId++);
    atb::infer::GroupTopkParam groupedParam;
    groupedParam.groupNum = param.numOfGroups;
    groupedParam.k = param.topkGroups[0];
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(groupedParam, &deviceLimitedNode.operation));
    deviceLimitedNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights"),
                                     GetTensorIdx(tensorMap, "in_expert_group")};
    deviceLimitedNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights")};
    ATB_SPEED_LOG_DEBUG("Fusion Router logits calculation success");
    return atb::NO_ERROR;
}

atb::Status CreateSparseTopkSoftMax(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &topKNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeTopkSoftmaxParam moeTopkSoftmaxParam;
    moeTopkSoftmaxParam.topkNum = int64_t(param.num.at(0));
    topKNode.operation = new atb_speed::common::MoeTopkSoftmaxOperation("MoeTopkSoftmaxOperation", moeTopkSoftmaxParam);
    topKNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_logits")};
    topKNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk"),
                             GetTensorIdx(tensorMap, "intermediate_selected_experts"),
                             GetTensorIdx(tensorMap, "intermediate_router_weights")};
    ATB_SPEED_LOG_DEBUG("Expert selection success");
    return atb::NO_ERROR;
}

atb::Status CreateVectorNorm(std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId)
{
    atb::Node &vectorNormNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::AclNNVectorNormParam aclNNVectorNormParam;
    vectorNormNode.operation = new atb_speed::common::VectorNormOperation("vectorNormOperation", aclNNVectorNormParam);
    vectorNormNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk")};
    vectorNormNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_router_weights_topk_sumed")};
    ATB_SPEED_LOG_DEBUG("execute vector norm success");
    return atb::NO_ERROR;
}

atb::Status GetoutTensorIdx(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, atb::GraphParam &opGraph, size_t &nodeId)
{
    auto &expertNode = opGraph.nodes.at(nodeId++);
    atb_speed::common::MoeMlpParam mlpExpertParam;
    mlpExpertParam.transpose = param.transpose;
    mlpExpertParam.topk = param.num.at(0);
    mlpExpertParam.numOfExperts = param.numOfExperts;
    mlpExpertParam.supportSwiGLU = param.supportSwiGLU;
    mlpExpertParam.moeLinearQuantType = param.moeLinearQuantType;
    mlpExpertParam.packQuantType = param.packQuantType;
    mlpExpertParam.denseQuantType = param.denseQuantType;
    mlpExpertParam.isBF16 = param.isBF16;
    mlpExpertParam.gateUpTransposeB = param.gateUpTransposeB;
    mlpExpertParam.downTransposeB = param.downTransposeB;
    mlpExpertParam.enableFusedRouting = param.enableFusedRouting;
    atb_speed::common::CreateMoeMlpOperation(mlpExpertParam, &expertNode.operation);
    expertNode.outTensorIds = {GetTensorIdx(tensorMap, "out_moe_rout")};
    expertNode.inTensorIds = {GetTensorIdx(tensorMap, "in_hiddenstates"),
                              GetTensorIdx(tensorMap, "in_mlp_gateup_weight_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_gateup_bias_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_gateup_descale_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_gateup_offset_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_gateup_scale_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_gateup_compress_idx_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_down_weight_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_down_bias_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_down_descale_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_down_offset_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_down_scale_expert"),
                              GetTensorIdx(tensorMap, "in_mlp_down_compress_idx_expert"),
                              GetTensorIdx(tensorMap, "in_expert_array"),
                              GetTensorIdx(tensorMap, "intermediate_selected_experts")};
    if (param.processLogits != "none") {
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_router_weights_topk_reduced"));
    } else {
        expertNode.inTensorIds.push_back(GetTensorIdx(tensorMap, "intermediate_router_weights_topk"));
    }
    ATB_SPEED_LOG_DEBUG("Expert Group calculation success");
    return atb::NO_ERROR;
}

int64_t CalculateNodeSize(const SparseMoeParam &param)
{
    uint64_t nodeSize = 4;
    if (param.routingMethod == "deviceLimited") {
        nodeSize += uint64_t(1);
    } else if (param.routingMethod == "integratedSoftmaxTopK") {
        nodeSize -= uint64_t(1);
    }
    if (param.useStdNorm) {
        nodeSize += uint64_t(NODE_SIZE_INCR_NORMALIZATION);
    }
    if (param.processLogits == "normalization" || param.processLogits == "norm") {
        nodeSize += uint64_t(2); // 2:number of nodes added
    } else if (param.processLogits == "scaling") {
        nodeSize += uint64_t(1);
    }
    return nodeSize;
}

atb::Status RoutingBlock(
    std::map<std::string, uint32_t> &tensorMap,
    const SparseMoeParam &param, size_t &nodeId, atb::GraphParam &opGraph)
{
    if (param.routingMethod == "deviceLimited") {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoesoftMax(tensorMap, param, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateGroupOperation(tensorMap, param, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopK(tensorMap, param, opGraph, nodeId));
    } else if (param.routingMethod == "integratedSoftmaxTopK") {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseTopkSoftMax(tensorMap, param, opGraph, nodeId));
    } else {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoesoftMax(tensorMap, param, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoetopK(tensorMap, param, opGraph, nodeId));
    }
    ATB_SPEED_LOG_DEBUG("Routing Block success");
    return atb::NO_ERROR;
}


atb::Status CreateSparseMoeOperation(const SparseMoeParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "SparseMoe";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("opGraph.internalTensorNum" << opGraph.internalTensorNum);

    uint64_t nodeCount = CalculateNodeSize(param);
    opGraph.nodes.resize(nodeCount);
    size_t nodeId = 0;
    CHECK_OPERATION_STATUS_RETURN(CreateSparseMoemoeGate(tensorMap, param, opGraph, nodeId));
    if (param.useStdNorm) {
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoeStd(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoeNorm(tensorMap, opGraph, nodeId));
    }
    CHECK_OPERATION_STATUS_RETURN(RoutingBlock(tensorMap, param, nodeId, opGraph));

    if (param.processLogits == "normalization") {
        // In_tensor[0]: router_weights: Batch * Seq; 2
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoereduce(tensorMap, opGraph, nodeId));
        // In_tensor[0]: router_weights: Batch * Seq; 2
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoedivide(tensorMap, opGraph, nodeId));
    } else if (param.processLogits == "scaling") {
        CHECK_OPERATION_STATUS_RETURN(CreateElewiseMuls(tensorMap, param, opGraph, nodeId));
    } else if (param.processLogits == "norm") {
        CHECK_OPERATION_STATUS_RETURN(CreateVectorNorm(tensorMap, opGraph, nodeId));
        CHECK_OPERATION_STATUS_RETURN(CreateSparseMoedivide(tensorMap, opGraph, nodeId));
    }

    CHECK_OPERATION_STATUS_RETURN(GetoutTensorIdx(tensorMap, param, opGraph, nodeId));

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    return atb::CreateOperation(opGraph, operation);
}
}
} // namespace atb_speed