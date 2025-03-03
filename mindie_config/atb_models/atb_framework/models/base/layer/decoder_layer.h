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
#ifndef ATB_SPEED_MODELS_BASE_DECODER_LAYER_H
#define ATB_SPEED_MODELS_BASE_DECODER_LAYER_H

#include <vector>
#include "nlohmann/json.hpp"

#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include "models/base/param/layer_param.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"

namespace atb_speed {
namespace base {

template <typename NormType>
class DecoderLayer {
public:
    explicit DecoderLayer(const LayerParam &param);
    virtual ~DecoderLayer() {};

    virtual int64_t BuildGraph(atb::Operation **operation);

protected:
    virtual void ConstructInTensorMap();
    virtual void ConstructInternalTensorMap();
    virtual void SetFusionAttentionParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    virtual void SetFusionAttentionNormParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    virtual void SetFusionAttentionLinearParam(atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    virtual void SetFusionAttentionATBSelfAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    virtual void SetFusionAttentionATBPagedAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    virtual void SetFusionAttentionAclNNIncreAttentionParam(
        atb_speed::common::FusionAttentionParam<NormType> &fusionAttentionParam);
    virtual void SetMlpParam(atb_speed::common::MlpParam<NormType> &mlpParam);
    virtual void SetMlpNormParam(atb_speed::common::MlpParam<NormType> &mlpParam);
    atb::Status CreateFusionAttentionOperation(atb::Operation **op);
    atb::Status CreateMlpOperation(atb::Operation **op);
    virtual std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor();
    virtual std::map<unsigned int, std::vector<std::string>> GetMlpIntensor();
    virtual atb::Status AddOperationToGraph();
    virtual atb::Status AddFusionAttention();
    virtual atb::Status AddFusionAttentionResidualAdd();
    virtual atb::Status AddMlp();
    virtual atb::Status AddMlpResidualAdd();

    std::map<std::string, std::vector<std::string>> inTensorCandidates = {};
    std::map<std::string, std::vector<std::string>> internalTensorCandidates = {};
    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};
    std::map<std::string, uint32_t> tensorMap = {};

    LayerParam param;
    atb::GraphParam graph;

private:
    const std::vector<std::string> attnWeight = {
        // Pack:
        // MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        // GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
        // No pack:
        // (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        "in_qkv_weight_0", "in_qkv_bias_0", "in_qkv_descale_0", "in_qkv_offset_0", "in_qkv_scale_0",
        "in_qkv_compress_idx_0",
        // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", "in_qkv_offset_1", "in_qkv_scale_1",
        "in_qkv_compress_idx_1",
        // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
        "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", "in_qkv_offset_2", "in_qkv_scale_2",
        "in_qkv_compress_idx_2",
        // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
        "in_qkv_dense_weight", "in_qkv_dense_bias", "in_qkv_dense_descale", "in_qkv_dense_offset",
        "in_qkv_dense_scale", "in_qkv_dense_compress_idx"};

    const std::vector<std::string> mlpWeight = {
        // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
        // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
        "in_mlp_weight_0", "in_mlp_bias_0", "in_mlp_descale_0", "in_mlp_offset_0", "in_mlp_scale_0",
        "in_mlp_compress_idx_0",
        // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
        "in_mlp_weight_1", "in_mlp_bias_1", "in_mlp_descale_1", "in_mlp_offset_1", "in_mlp_scale_1",
        "in_mlp_compress_idx_1",
        // shape: [hiddenSize, intermediateSizePerRank]
        "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
        "in_mlp_down_scale", "in_mlp_down_compress_idx"};
};

}  // namespace base
}  // namespace atb_speed
#endif
