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
#include "models/llama/layer/decoder_layer.h"

namespace atb_speed {
namespace llama {

void LlamaLayerParam::PrintParam()
{
    LayerParam::PrintParam();
    std::stringstream ss;
    ss << "Llama Layer Param:splitWithStride: " << this->splitWithStride;
    ATB_SPEED_LOG_INFO(ss.str());
}

LlamaDecoderLayer::LlamaDecoderLayer(
    const LlamaLayerParam &param) : atb_speed::base::DecoderLayer<atb::infer::RmsNormParam>(param)
{
    this->param = param;
    this->param.CheckParam();
};

void LlamaDecoderLayer::SetFusionAttentionLinearParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam)
{
    DecoderLayer<atb::infer::RmsNormParam>::SetFusionAttentionLinearParam(fusionAttentionParam);
    fusionAttentionParam.splitWithStride = this->param.splitWithStride;
}

} // namespace llama
} // namespace atb_speed