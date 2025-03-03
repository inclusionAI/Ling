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
#include "models/starcoder2/model/decoder_model.h"
#include "models/starcoder2/layer/decoder_layer.h"
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"

#include <atb/types.h>

namespace atb_speed {
namespace starcoder2 {

DecoderModel::DecoderModel(const std::string &param) : atb_speed::base::DecoderModel(param)
{
    this->weightCountFinalNorm = 2; // 2 is weightCountFinalNorm
}

atb::Status DecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    atb_speed::base::LayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    DecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));
    return atb::NO_ERROR;
}

} // namespace starcoder2
} // namespace atb_speed