/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except inccompliance with the License.
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
#ifndef ATB_SPEED_MODELS_CHATGLM_DECODER_LAYER_H
#define ATB_SPEED_MODELS_CHATGLM_DECODER_LAYER_H

#include "atb/atb_infer.h"

#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"

namespace atb_speed {
namespace chatglm {

class ChatglmLayerParam : public atb_speed::base::LayerParam {
};

class ChatglmDecoderLayer : public atb_speed::base::DecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit ChatglmDecoderLayer(const ChatglmLayerParam &param);
    ~ChatglmDecoderLayer() override {};

protected:
    void SetFusionAttentionParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;
    void SetFusionAttentionNormParam(
        atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam) override;

    ChatglmLayerParam param;
};

}  // namespace chatglm
}  // namespace atb_speed
#endif