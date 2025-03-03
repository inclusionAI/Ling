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

#ifndef ATB_MODELS_BERT_ENCODER_LAYER_H
#define ATB_MODELS_BERT_ENCODER_LAYER_H


#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/str_split.h"


namespace atb_speed::bert {

    struct EncoderLayerParam {
        int dk = 0;
        int64_t geluApproximate = -1;
        int headNum = 0;
        float layerNormEps = 0;
        int64_t layerNormImplMode = 0;
        std::string model = "bert";
        int rank = 0;
        int rankSize = 1;
    };

    atb::Status EncoderLayer(const EncoderLayerParam &param, atb::Operation **operation);

}  // namespace atb_speed::bert

#endif  // ATB_MODELS_BERT_ENCODER_LAYER_H
