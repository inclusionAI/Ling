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
#ifndef ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_MODEL_H
#define ATB_SPEED_MODELS_BAICHUAN2_13B_PA_QUANT_MODEL_H

#include "atb_speed/base/model.h"
#include "atb_speed/utils/model_factory.h"
#include "models/baichuan2/13b/layer/paged_attention_quant_layer.h"
#include "models/base/model/decoder_model.h"
#include "models/base/param/model_param.h"

namespace atb_speed {
namespace baichuan2_13b {

class BaichuanModelParam : public atb_speed::base::ModelParam {
public:
    void PrintParam() override;

    // 是否开启alibi mask free
    bool enableAlibiMaskFree = false;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class PagedAttentionQuantModel : public atb_speed::base::DecoderModel {
public:
    explicit PagedAttentionQuantModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLayerParam(BaichuanLayerParam &layerParam, uint32_t layerId);
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;
    void SetFinalNormParam(atb::infer::RmsNormParam &normParam) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;

    BaichuanModelParam param;
};

} // namespace baichuan2_13b
} // namespace atb_speed
#endif
