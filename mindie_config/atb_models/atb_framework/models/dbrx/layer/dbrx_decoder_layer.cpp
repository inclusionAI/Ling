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
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"
#include "models/dbrx/layer/dbrx_decoder_layer.h"

namespace atb_speed {
namespace dbrx {

DbrxDecoderLayer::DbrxDecoderLayer(
    const atb_speed::moe::MoeLayerParam &param) : atb_speed::moe::MoeDecoderLayer<atb::infer::LayerNormParam>(
        static_cast<atb_speed::moe::MoeLayerParam>(param))
{};

void DbrxDecoderLayer::SetSparseMoeParam(atb_speed::common::SparseMoeParam &sparseMoeParam)
{
    MoeDecoderLayer<atb::infer::LayerNormParam>::SetSparseMoeParam(sparseMoeParam);
    sparseMoeParam.processLogits = "norm";
    sparseMoeParam.gateUpTransposeB = true;
}

} // namespace dbrx
} // namespace atb_speed