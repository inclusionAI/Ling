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
#include "models/gemma/layer/decoder_layer.h"

namespace atb_speed {
namespace gemma {
GemmaDecoderLayer::GemmaDecoderLayer(
    const GemmaLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
}
void GemmaDecoderLayer::SetMlpParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetMlpParam(mlpParam);
    if (!this->param.enableSwiGLU) {
        mlpParam.activationParam.geluMode = atb::infer::ActivationParam::NONE_MODE;
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    }
}
} // namespace gemma
} // namespace atb_speed