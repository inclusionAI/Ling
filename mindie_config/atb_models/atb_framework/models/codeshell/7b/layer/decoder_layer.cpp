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
#include "codeshell/7b/layer/decoder_layer.h"

namespace atb_speed {
namespace codeshell_7b {

DecoderLayer::DecoderLayer(
    const atb_speed::base::LayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::LayerNormParam>(
        static_cast<atb_speed::base::LayerParam>(param))
{
    this->param = param;
    this->param.CheckParam();
}

void DecoderLayer::SetMlpParam(atb_speed::common::MlpParam<atb::infer::LayerNormParam> &mlpParam)
{
    atb_speed::base::DecoderLayer<atb::infer::LayerNormParam>::SetMlpParam(mlpParam);
    mlpParam.mlpPackType = atb_speed::common::GetMlpPackType(param.packQuantType.at(1), true);
    if (!this->param.enableSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_GELU;
    }
}
} // namespace codeshell_7b
} // namespace atb_speed