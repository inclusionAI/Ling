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

#ifndef ATB_SPEED_MODELS_BERT_MLP_OPERATION_H
#define ATB_SPEED_MODELS_BERT_MLP_OPERATION_H


#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"


namespace atb_speed::bert {

    struct MlpParam {
        int64_t geluApproximate = -1;
        float layerNormEps = 0;
        int64_t layerNormImplMode = 0;
        int rank = 0;
        int rankSize = 1;
    };

    atb::Status Mlp(const MlpParam &param, atb::Operation **operation);

}  // namespace atb_speed::bert

#endif  // ATB_SPEED_MODELS_BERT_MLP_OPERATION_H
