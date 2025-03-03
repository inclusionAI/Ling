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
#ifndef ATB_SPEED_MODELS_GPTNEOX_20B_PA_LAYER_H
#define ATB_SPEED_MODELS_GPTNEOX_20B_PA_LAYER_H

#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"

namespace atb_speed {
namespace gptneox_20b {
struct PALayerParam {
    float layerNormEps = 0;
    int headNum = 0;
    int dk = 0;
    float rotaryPct = 0.0;
    float qScale = 1.0;
    float qkScale = 1.0;
    bool transposedWeight = true;
    std::string model = "gptneox_20b";
    bool isPrefill = false;
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
};

enum LayerPATensorId : int {
    IN_HIDDENSTATES = 0,
    IN_INPUTLAYERNORMWEIGTH, // weights
    IN_INPUTLAYERNORMBIAS,
    IN_POSTATTNLAYERNORMWEIGHT,
    IN_POSTATTNLAYERNORMBIAS,
    IN_QKVWEIGHT,
    IN_QKVBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARBIAS,
    IN_FFNLINEARWEIGHT,
    IN_FFNLINEARBIAS,
    IN_FFNOUTLINEARWEIGHT,
    IN_FFNOUTLINEARBIAS,
    IN_POSITIONIDS, // inputs
    IN_COSEMBED,
    IN_SINEMBED,
    IN_ATTENTIONMASK,
    IN_CACHEK, // kvcache
    IN_CACHEV,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_INPUT_LENGTHS,

    OUT_GPTNEOXLAYEROUT,

    INTERMEDIATE_INPUTLAYERNORMOUT,
    INTERMEDIATE_MIXEDQKVLINEAROUT,
    INTERMEDIATE_QUERYEMBED,
    INTERMEDIATE_KEYEMBED,
    INTERMEDIATE_VALUE,
    INTERMEDIATE_QUERYEMBED_SCALED,
    INTERMEDIATE_SELFATTNOUT,
    INTERMEDIATE_SELFATTNLINEAROUT,
    INTERMEDIATE_POSTATTNLAYERNORMOUT,
    INTERMEDIATE_FFNLINEAROUT,
    INTERMEDIATE_FFNACTOUT,
    INTERMEDIATE_FFNOUTLINEAROUT,
    INTERMEDIATE_ATTNMLPADDOUT,
    INTERMEDIATE_ATTNMLP_ALLREDUCEOUT
};

struct PositionEmbeddingPAParam {
    int32_t headNum = 0;
    int32_t dk = 0;
    float rotaryPct = 0.25;
};

atb::Status PositionEmbeddingPAOperation(const PositionEmbeddingPAParam &param, atb::Operation **operation);

atb::Status PALayer(const PALayerParam &param, atb::Operation **operation);

atb::Operation *CreatePALayer(const nlohmann::json &paramJson);
} // namespace gptneox_20b
} // namespace atb_speed

#endif // ATB_SPEED_MODELS_GPTNEOX_20B_PA_LAYER_H
