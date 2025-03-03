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

#include "models/deepseek/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseek {

void DeepseekLayerParam::PrintParam()
{
    MoeLayerParam::PrintParam();
    std::stringstream ss;
    ss << "Deepseek Layer Param: " << "hasSharedExpert: " << this->hasSharedExpert
       << ", hasSharedExpertGate: " << this->hasSharedExpertGate
       << ", numOfGroups: " << this->numOfGroups
       << ", numOfSharedExperts: " << this->numOfSharedExperts
       << ", firstKDenseReplace: " << this->firstKDenseReplace
       << ", topkGroups: " << this->topkGroups;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

DeepseekDecoderLayer::DeepseekDecoderLayer(
    const DeepseekLayerParam &param) : atb_speed::moe::MoeDecoderLayer<atb::infer::RmsNormParam>(
        static_cast<atb_speed::moe::MoeLayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
    this->inTensorCandidates["shared_expert_weight"] = {
        "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert", "in_mlp_gateup_descale_shared_expert",
        "in_mlp_gateup_offset_shared_expert", "in_mlp_gateup_scale_shared_expert",
        "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
        "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    };
    if (param.layerId < this->param.firstKDenseReplace && param.tensorParallelInfo.worldSize <= 1) {
        this->internalTensorCandidates["default_moe"] = {"norm_out"};
    } else {
        this->internalTensorCandidates["default_moe"] = {"norm_out", "shared_expert_out"};
    }

    if (this->param.hasSharedExpert && param.layerId >= this->param.firstKDenseReplace) {
        this->internalTensorCandidates["default_moe"].push_back("moe_out");
    }
}

void DeepseekDecoderLayer::ConstructInTensorMap()
{
    this->inTensorList.clear();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "shared_expert_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "moe_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default_moe", this->inTensorList);
    if (this->param.enableSpeculate || this->param.enableSplitFuse || this->param.enablePrefixCache) {
        atb_speed::common::AddTensorToList(this->inTensorCandidates, "q_len", this->inTensorList);
    }
}

atb::Status DeepseekDecoderLayer::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttention());
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
    CHECK_OPERATION_STATUS_RETURN(this->AddSelfNorm());
    if (param.layerId < param.firstKDenseReplace) {
        atb_speed::common::SharedExpertParam mlpExpertParam = SetSharedExpertParam();
        CHECK_OPERATION_STATUS_RETURN(this->AddMlpExpert(mlpExpertParam));
    } else {
        if (param.hasSharedExpert) {
            atb_speed::common::SharedExpertParam sharedExpertParam = SetSharedExpertParam();
            sharedExpertParam.hasSharedExpertGate = this->param.hasSharedExpertGate;
            CHECK_OPERATION_STATUS_RETURN(this->AddMlpExpert(sharedExpertParam));
        }
        CHECK_OPERATION_STATUS_RETURN(this->AddMoe());
        if (param.hasSharedExpert) {
            CHECK_OPERATION_STATUS_RETURN(this->AddExpertAdd());
        }
    };
    if (param.tensorParallelInfo.worldSize > 1) {
        CHECK_OPERATION_STATUS_RETURN(this->AddMoeAllReduce());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddMlpResidualAdd());
    ATB_SPEED_LOG_DEBUG("DeepSeek Layer Add Op to Graph success");
    return atb::NO_ERROR;
}

atb_speed::common::SharedExpertParam DeepseekDecoderLayer::SetSharedExpertParam()
{
    atb_speed::common::SharedExpertParam sharedExpertParam;
    sharedExpertParam.transposeGateup = this->param.transpose;
    sharedExpertParam.transposeDown = this->param.transpose;
    sharedExpertParam.hasSharedExpertGate = false;
    sharedExpertParam.mlpLinearQuantType = this->param.mlpLinearQuantType;
    sharedExpertParam.mlpLinearTransposeType = this->param.mlpLinearTransposeType;
    sharedExpertParam.packQuantType = this->param.packQuantType.at(1);
    sharedExpertParam.isBF16 = this->param.isBF16;
    sharedExpertParam.supportSwiGLU = this->param.enableSwiGLU;
    ATB_SPEED_LOG_DEBUG("set shared expert success");
    return sharedExpertParam;
}

atb::Status DeepseekDecoderLayer::AddMlpExpert(const atb_speed::common::SharedExpertParam &mlpExpertParam)
{
    atb::Node mlpExpertNode;
    CHECK_OPERATION_STATUS_RETURN(
        atb_speed::common::CreateSharedExpertOperation(mlpExpertParam, &mlpExpertNode.operation));
    mlpExpertNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {
        "norm_out", "in_mlp_gateup_weight_shared_expert", "in_mlp_gateup_bias_shared_expert",
        "in_mlp_gateup_descale_shared_expert", "in_mlp_gateup_offset_shared_expert",
        "in_mlp_gateup_scale_shared_expert", "in_mlp_gateup_compress_idx_shared_expert",
        "in_mlp_down_weight_shared_expert", "in_mlp_down_bias_shared_expert", "in_mlp_down_descale_shared_expert",
        "in_mlp_down_offset_shared_expert", "in_mlp_down_scale_shared_expert", "in_mlp_down_compress_idx_shared_expert",
        "in_shared_expert_gate_weight", "in_shared_expert_gate_bias", "in_shared_expert_gate_descale",
        "in_shared_expert_gate_offset", "in_shared_expert_gate_scale", "in_shared_expert_gate_compress_idx"
    });
    if (param.layerId < this->param.firstKDenseReplace) {
        if (param.tensorParallelInfo.worldSize > 1) {
            mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"shared_expert_out"});
        } else {
            mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
        }
    } else {
        mlpExpertNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"shared_expert_out"});
    }
    
    this->graph.nodes.push_back(mlpExpertNode);
    ATB_SPEED_LOG_DEBUG("mlp expert calculation success");
    return atb::NO_ERROR;
}

atb::Status DeepseekDecoderLayer::AddMoe()
{
    atb::Node moeNode;
    atb_speed::common::SparseMoeParam sparseMoeParam;
    this->SetSparseMoeParam(sparseMoeParam);
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::CreateSparseMoeOperation(sparseMoeParam, &moeNode.operation));
    std::vector<std::string> sparseMoeInTensorNames = {
        "norm_out", "block_sparse_moe_gate_weight", "block_sparse_moe_gate_bias", "block_sparse_moe_gate_descale",
        "block_sparse_moe_gate_offset", "block_sparse_moe_gate_scale", "block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight", "in_mlp_gateup_bias", "in_mlp_gateup_descale", "in_mlp_gateup_offset",
        "in_mlp_gateup_scale", "in_mlp_gateup_compress_idx", "in_mlp_down_weight", "in_mlp_down_bias",
        "in_mlp_down_descale", "in_mlp_down_offset", "in_mlp_down_scale", "in_mlp_down_compress_idx", "expert_array",
        "expert_group", "one_hot", "zero_hot"
    };
    moeNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, sparseMoeInTensorNames);
    if (this->param.hasSharedExpert) {
        moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
    } else {
        if (param.tensorParallelInfo.worldSize > 1) {
            moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
        } else {
            moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
        }
    }

    this->graph.nodes.push_back(moeNode);
    ATB_SPEED_LOG_DEBUG("Moe calculation success");

    return atb::NO_ERROR;
}

atb::Status DeepseekDecoderLayer::AddMoeAllReduce()
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = this->param.tensorParallelInfo.rank;
    allReduceParam.rankSize = this->param.tensorParallelInfo.worldSize;
    allReduceParam.backend = this->param.tensorParallelInfo.backend;
    allReduceParam.rankTableFile = this->param.tensorParallelInfo.rankTableFile;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allReduceParam, &moeAllReduceNode.operation));
    if (this->param.hasSharedExpert && param.layerId >= this->param.firstKDenseReplace) {
        moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
    } else {
        moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"shared_expert_out"});
    }
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("create all reduce");
    return atb::NO_ERROR;
}

atb::Status DeepseekDecoderLayer::AddExpertAdd()
{
    atb::Node expertAddNode;
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &expertAddNode.operation));
    expertAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out", "shared_expert_out"});
    if (param.tensorParallelInfo.worldSize > 1) {
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
    } else {
        expertAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    }

    this->graph.nodes.push_back(expertAddNode);
    ATB_SPEED_LOG_DEBUG("create add operation");
    return atb::NO_ERROR;
}

} // namespace deepseek
} // namespace atb_speed