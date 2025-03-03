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
#ifndef ATB_SPEED_MODELS_GEMMA_DECODER_MODEL_H
#define ATB_SPEED_MODELS_GEMMA_DECODER_MODEL_H

#include "atb_speed/base/model.h"
#include "models/base/param/model_param.h"
#include "atb_speed/utils/model_factory.h"
#include "models/gemma/layer/decoder_layer.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace gemma {
class GemmaModelParam : public atb_speed::base::ModelParam {
public:
    uint32_t hiddenSize = 0;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class GemmaDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit GemmaDecoderModel(const std::string &param);
protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLayerParam(GemmaLayerParam &layerParam, uint32_t layerId);
    atb::Status AddOperationToGraph() override;
    GemmaModelParam param;
private:
    atb::Status AddMuls();
};
REGISTER_MODEL(gemma, GemmaDecoderModel);
}  // namespace gemma
}  // namespace atb_speed
#endif
