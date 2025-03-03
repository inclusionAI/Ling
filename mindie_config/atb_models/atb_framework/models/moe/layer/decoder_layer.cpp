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

#include "operations/fusion/linear/linear.h"
#include "operations/fusion/norm/norm_linear.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {

void MoeLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << "Moe Layer Param: " << "enableTopKSoftmax: " << this->enableTopKSoftmax
       << ", transpose: " << this->transpose
       << ", numOfExperts: " << this->numOfExperts
       << ", expertParallelDegree: " << this->expertParallelDegree;
    ATB_SPEED_LOG_DEBUG(ss.str());
}

template <typename NormType>
MoeDecoderLayer<NormType>::MoeDecoderLayer(
    const MoeLayerParam &param) : atb_speed::base::DecoderLayer<NormType>(
        static_cast<atb_speed::base::LayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
    this->inTensorCandidates["moe_weight"] = {
        "block_sparse_moe_gate_weight", "block_sparse_moe_gate_bias", "block_sparse_moe_gate_descale",
        "block_sparse_moe_gate_offset", "block_sparse_moe_gate_scale", "block_sparse_moe_gate_compress_idx",
        "in_mlp_gateup_weight", "in_mlp_gateup_bias", "in_mlp_gateup_descale", "in_mlp_gateup_offset",
        "in_mlp_gateup_scale", "in_mlp_gateup_compress_idx",
        "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
        "in_mlp_down_scale", "in_mlp_down_compress_idx"
    };
    this->inTensorCandidates["default_moe"] = {
        "expert_array", "expert_group", "one_hot", "zero_hot"
    };
    if (param.tensorParallelInfo.worldSize > 1) {
        this->internalTensorCandidates["default_moe"] = {
            "norm_out", "moe_out"};
    } else {
        this->internalTensorCandidates["default_moe"] = {"norm_out"};
    }
}

template <typename NormType>
void MoeDecoderLayer<NormType>::ConstructInTensorMap()
{
    this->inTensorList.clear();
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "input_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "attn_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "post_attn_norm_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "moe_weight", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default", this->inTensorList);
    atb_speed::common::AddTensorToList(this->inTensorCandidates, "default_moe", this->inTensorList);
}

template <typename NormType>
void MoeDecoderLayer<NormType>::ConstructInternalTensorMap()
{
    atb_speed::base::DecoderLayer<NormType>::ConstructInternalTensorMap();
    atb_speed::common::AddTensorToList(this->internalTensorCandidates, "default_moe", this->intermediateTensorList);
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddOperationToGraph()
{
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttention());
    CHECK_OPERATION_STATUS_RETURN(this->AddFusionAttentionResidualAdd());
    CHECK_OPERATION_STATUS_RETURN(this->AddSelfNorm());
    CHECK_OPERATION_STATUS_RETURN(this->AddMoe());
    if (param.tensorParallelInfo.worldSize > 1) {
        CHECK_OPERATION_STATUS_RETURN(this->AddMoeAllReduce());
    }
    CHECK_OPERATION_STATUS_RETURN(this->AddMlpResidualAdd());
    ATB_SPEED_LOG_DEBUG("Add Op to Graph success");
    return atb::NO_ERROR;
}

template<typename NormType>
void MoeDecoderLayer<NormType>::SetSelfNormParam(atb_speed::common::NormLinearParam<NormType> &selfNormParam)
{
    atb_speed::common::MlpParam<NormType> mlpParam;
    atb_speed::base::DecoderLayer<NormType>::SetMlpParam(mlpParam);
    selfNormParam.isAntiOutlier = atb_speed::common::CheckAntiOutlier(mlpParam.packQuantType);
    selfNormParam.normHasBias = this->param.normHasBias;
    selfNormParam.enableAddNorm = mlpParam.enableAddNorm;
    selfNormParam.normParamType = mlpParam.normParamType;
    selfNormParam.normQuantParamType = mlpParam.normQuantParamType;
}

template<typename NormType>
std::map<std::string, uint32_t> MoeDecoderLayer<NormType>::ConstructNormTensorMap() const
{
    std::vector<std::string> targetNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias", "in_scale", "in_offset",
        "intermediate_norm", "out_add", "in_residual_input"
    };
    std::vector<std::string> originNames = {
        "in_hidden_states", "in_post_attn_norm_weight", "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
        "in_post_attn_norm_new_bias", "in_mlp_gateup_scale", "in_mlp_gateup_offset", "norm_out", "in_hidden_states",
        "intermediate_attn_out"
    };
    std::map<std::string, uint32_t> normTensorMap;
    for (size_t i = 0; i < targetNames.size(); ++i) {
        normTensorMap[targetNames.at(i)] = atb_speed::common::GetTensorIdx(this->tensorMap, originNames.at(i));
    }
    return normTensorMap;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddSelfNorm()
{
    atb_speed::common::NormLinearParam<NormType> selfNormParam;
    SetSelfNormParam(selfNormParam);
    std::map<std::string, uint32_t> selfNormTensorMap = ConstructNormTensorMap();
    CHECK_OPERATION_STATUS_RETURN(atb_speed::common::InsertNorm(this->graph, selfNormParam, selfNormTensorMap));
    return atb::NO_ERROR;
}

template <typename NormType>
void MoeDecoderLayer<NormType>::SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam)
{
    sparseMoeParam.transpose = this->param.transpose;
    sparseMoeParam.numOfExperts = this->param.numOfExperts;
    sparseMoeParam.num = this->param.numOfSelectedExperts;
    sparseMoeParam.expertParallelDegree = this->param.expertParallelDegree;
    sparseMoeParam.processLogits = this->param.processLogits;
    sparseMoeParam.supportSwiGLU = this->param.enableSwiGLU;
    sparseMoeParam.routingMethod = this->param.routingMethod;
    sparseMoeParam.moeLinearQuantType = this->param.moeLinearQuantType;
    sparseMoeParam.packQuantType = this->param.packQuantType.at(1);
    sparseMoeParam.isBF16 = this->param.isBF16;
    sparseMoeParam.enableFusedRouting = this->param.enableFusedRouting;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddMoe()
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
    if (param.tensorParallelInfo.worldSize > 1) {
        moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
    } else {
        moeNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    }
    this->graph.nodes.push_back(moeNode);
    ATB_SPEED_LOG_DEBUG("Moe calculation success");

    return atb::NO_ERROR;
}

template <typename NormType>
atb::Status MoeDecoderLayer<NormType>::AddMoeAllReduce()
{
    atb::Node moeAllReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = this->param.tensorParallelInfo.rank;
    allReduceParam.rankSize = this->param.tensorParallelInfo.worldSize;
    allReduceParam.backend = this->param.tensorParallelInfo.backend;
    allReduceParam.rankTableFile = this->param.tensorParallelInfo.rankTableFile;
    CHECK_OPERATION_STATUS_RETURN(CreateOperation(allReduceParam, &moeAllReduceNode.operation));
    moeAllReduceNode.inTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"moe_out"});
    moeAllReduceNode.outTensorIds = atb_speed::common::GetTensorIdxList(this->tensorMap, {"intermediate_mlp_out"});
    this->graph.nodes.push_back(moeAllReduceNode);
    ATB_SPEED_LOG_DEBUG("create all reduce");

    return atb::NO_ERROR;
}

template class MoeDecoderLayer<atb::infer::RmsNormParam>;
template class MoeDecoderLayer<atb::infer::LayerNormParam>;

} // namespace moe
} // namespace atb_speed