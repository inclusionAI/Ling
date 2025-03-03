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
#ifndef ATB_SPEED_MODELS_DBRX_DECODER_LAYER_H
#define ATB_SPEED_MODELS_DBRX_DECODER_LAYER_H

#include "atb/atb_infer.h"

#include "operations/fusion/linear/linear_parallel.h"
#include "models/base/param/layer_param.h"
#include "models/base/layer/decoder_layer.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace dbrx {

class DbrxDecoderLayer : public atb_speed::moe::MoeDecoderLayer<atb::infer::LayerNormParam> {
public:
    explicit DbrxDecoderLayer(const atb_speed::moe::MoeLayerParam &param);
    ~DbrxDecoderLayer() override {};
protected:
    void SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam) override;
    atb_speed::moe::MoeLayerParam param;
};

}  // namespace dbrx
}  // namespace atb_speed
#endif