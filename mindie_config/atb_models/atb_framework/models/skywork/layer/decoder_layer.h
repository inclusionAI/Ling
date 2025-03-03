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
#ifndef ATB_SPEED_MODELS_SKYWORK_DECODER_LAYER_H
#define ATB_SPEED_MODELS_SKYWORK_DECODER_LAYER_H

#include "atb/atb_infer.h"
#include "operations/fusion/moe/sparse_moe.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace skywork {

class DecoderLayer : public atb_speed::moe::MoeDecoderLayer<atb::infer::RmsNormParam> {
public:
    explicit DecoderLayer(const atb_speed::moe::MoeLayerParam &param);
    ~DecoderLayer() override {};

protected:
    void SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam) override;
    atb_speed::moe::MoeLayerParam param;
};
}  // namespace skywork
}  // namespace atb_speed
#endif
