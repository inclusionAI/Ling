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
#include "paged_attention_quant_model.h"

#include <vector>

#include "atb/atb_infer.h"
#include "nlohmann/json.hpp"

#include "models/baichuan2/7b/layer/paged_attention_quant_layer.h"

namespace atb_speed {
namespace baichuan2_7b {
REGISTER_MODEL(baichuan2_7b, PagedAttentionQuantModel);

PagedAttentionQuantModel::PagedAttentionQuantModel(const std::string &param) : atb_speed::base::DecoderModel(param) {}

atb::Status PagedAttentionQuantModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    PagedAttentionQuantLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace baichuan2_7b
} // namespace atb_speed
