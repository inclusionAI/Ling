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
#ifndef ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_LAYER_H
#define ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include <map>
#include <string>

#include "models/base/layer/decoder_layer.h"
#include "models/base/param/layer_param.h"

namespace atb_speed {
namespace baichuan2_13b {

class BaichuanLayerParam : public atb_speed::base::LayerParam {
public:
    void PrintParam() override;

    // 是否开启alibi mask free
    bool enableAlibiMaskFree = false;
};

class PagedAttentionQuantLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit PagedAttentionQuantLayer(const BaichuanLayerParam &param);

protected:
    void ConstructInTensorMap() override;
    std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor() override;
    void SetFusionAttentionNormParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    void SetFusionAttentionATBSelfAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    void SetMlpNormParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam) override;

    BaichuanLayerParam param;
};

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
