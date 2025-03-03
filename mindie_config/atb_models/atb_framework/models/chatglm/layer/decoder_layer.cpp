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
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/norm/norm_linear.h"
#include "models/chatglm/layer/decoder_layer.h"

namespace atb_speed {
namespace chatglm {

ChatglmDecoderLayer::ChatglmDecoderLayer(
    const ChatglmLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.CheckParam();
};

void ChatglmDecoderLayer::SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionParam(fusionAttentionParam);
    // // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::HALF_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = this->param.hiddenSizePerAttentionHead / 2; // 2: half rotary
}

void ChatglmDecoderLayer::SetFusionAttentionNormParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionNormParam(fusionAttentionParam);
    fusionAttentionParam.enableNormQuantOp = false;
}

} // namespace chatglm
} // namespace atb_speed