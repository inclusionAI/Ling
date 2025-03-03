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
#include "models/baichuan2/13b/model/paged_attention_quant_model.h"

#include <vector>

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "atb_speed/log.h"
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_13b {
REGISTER_MODEL(baichuan2_13b, PagedAttentionQuantModel);

void BaichuanModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("enableAlibiMaskFree")) {
        this->enableAlibiMaskFree = paramJson["enableAlibiMaskFree"].get<bool>();
    }
}

void BaichuanModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("BaichuanModelParam: enableAlibiMaskFree: " << this->enableAlibiMaskFree);
}

PagedAttentionQuantModel::PagedAttentionQuantModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["alibi_mask_compress"] = {"in_slopes"};
}

void PagedAttentionQuantModel::ConstructInTensorMap()
{
    atb_speed::base::DecoderModel::ConstructInTensorMap();
    atb_speed::common::AssignTensorIdx(this->inTensorCandidates, "alibi_mask_compress", this->inTensorMap);
}

atb::Status PagedAttentionQuantModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    BaichuanLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    PagedAttentionQuantLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

void PagedAttentionQuantModel::SetLayerParam(BaichuanLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.enableAlibiMaskFree = this->param.enableAlibiMaskFree;
}

void PagedAttentionQuantModel::SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerNodeInput(layerNode, layerId);
    layerNode.inTensors.at(layerNode.inTensors.size() - 1) =
        &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "in_slopes"));
}

void PagedAttentionQuantModel::SetFinalNormParam(atb::infer::RmsNormParam &normParam)
{
    atb_speed::base::DecoderModel::SetFinalNormParam(normParam);
    normParam.normParam.precisionMode = atb::infer::RmsNormParam::HIGH_PERFORMANCE_MODE;
}

atb::Status PagedAttentionQuantModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_SPEED_LOG_DEBUG("Baichuan BindParamHostTensor nodeId = " << nodeId);

    if (nodeId != 0) {
        return atb::NO_ERROR;
    }

    uint32_t tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "token_offset");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->tokenOffset.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "seq_len");
    if (!this->param.isPrefill && this->param.enableCompressHead) {
        tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "in_ra_seqlens");
    }
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->seqLen.data();
    }

    tensorIdx = atb_speed::common::GetTensorIdx(this->inTensorMap, "q_len");
    if (tensorIdx != UINT32_MAX) {
        graph_.inTensors.at(tensorIdx).hostData = this->qLen.data();
    }

    ATB_SPEED_LOG_DEBUG("Baichuan BindParamHostTensor end");
    return atb::NO_ERROR;
}

} // namespace baichuan2_13b
} // namespace atb_speed
