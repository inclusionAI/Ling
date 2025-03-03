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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_DECODER_MODEL_H
#define ATB_SPEED_MODELS_DEEPSEEK_DECODER_MODEL_H

#include <atb_speed/utils/model_factory.h>
#include <models/moe/model/decoder_model.h>
#include "models/deepseek/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseek {

class DeepseekModelParam : public atb_speed::moe::MoeModelParam {
public:
    void PrintParam() override;

    std::string processLogits = "normalization";
    // shared experts
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = true;
    int numOfSharedExperts = 2;
    int firstKDenseReplace = 1;
    // group limited routing
    int numOfGroups = 8;
    atb::SVector<int32_t> topkGroups = {};

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class DecoderModel : public atb_speed::moe::MoeDecoderModel {
public:
    explicit DecoderModel(const std::string &param);

protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLayerParam(DeepseekLayerParam &layerParam, uint32_t layerId);
    void ConstructInTensorMap() override;
    void SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId) override;

    DeepseekModelParam param;
};

REGISTER_MODEL(deepseek, DecoderModel);

}  // namespace deepseek
}  // namespace atb_speed
#endif
