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
#ifndef ATB_SPEED_MODELS_DEEPSEEK_DECODER_LAYER_H
#define ATB_SPEED_MODELS_DEEPSEEK_DECODER_LAYER_H

#include "operations/fusion/moe/moe_shared_expert.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace deepseek {

class DeepseekLayerParam : public atb_speed::moe::MoeLayerParam {
public:
    void PrintParam() override;

    bool hasSharedExpert = true;
    bool hasSharedExpertGate = true;
    int numOfGroups = 8;
    int numOfSharedExperts = 2;
    int firstKDenseReplace = 1;
    int layerId = 0;
    atb::SVector<int32_t> topkGroups = {3};
};

class DeepseekDecoderLayer : public atb_speed::moe::MoeDecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit DeepseekDecoderLayer(const DeepseekLayerParam &param);
    ~DeepseekDecoderLayer() override {};

protected:
    void ConstructInTensorMap() override;
    atb::Status AddOperationToGraph() override;
    atb_speed::common::SharedExpertParam SetSharedExpertParam();
    atb::Status AddMlpExpert(const atb_speed::common::SharedExpertParam &mlpExpertParam);
    atb::Status AddMoe() override;
    atb::Status AddMoeAllReduce() override;
    atb::Status AddExpertAdd();

    DeepseekLayerParam param;
};


}  // namespace deepseek
}  // namespace atb_speed
#endif