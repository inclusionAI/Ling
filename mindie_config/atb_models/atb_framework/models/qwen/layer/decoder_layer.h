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
#ifndef ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H
#define ATB_SPEED_MODELS_QWEN_DECODER_LAYER_H

#include "models/base/layer/decoder_layer.h"
#include "models/base/param/layer_param.h"

namespace atb_speed {
namespace qwen {

class QwenLayerParam : public atb_speed::base::LayerParam {
public:
    void PrintParam() override;

    bool enableLogN = false;
    bool isEmbedding = false;
    bool enableQScale = false;
};

class QwenDecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit QwenDecoderLayer(const QwenLayerParam &param);
    ~QwenDecoderLayer() override{};

protected:
    void ConstructInTensorMap() override;
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor() override;

    QwenLayerParam param;
};

} // namespace qwen
} // namespace atb_speed
#endif
