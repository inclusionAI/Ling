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

#ifndef ATB_SPEED_LAYER_PARALLEL_LAYER_V2_H
#define ATB_SPEED_LAYER_PARALLEL_LAYER_V2_H
#include <atb/atb_infer.h>
#include "atb_speed/utils/operation_util.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace common {

struct QuantParam {
    atb::infer::QuantType quantType;
    atb::infer::ElewiseParam::ElewiseType elewiseType;
    float inputScale = 1.0f;
    int inputOffset = 0;
    int tilingN = 0;
    int tilingK = 0;
    bool isQuantOp = false;
};

struct CommParam {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    std::string backend = "hccl";
};

struct ParallelParamV2 {
    bool isBias = false;
    bool transposeA = false;
    bool transposeB = true;
    bool isQuant = false;
    bool isSparse = false;
    bool isAllGatherTranspose = false;
    bool isBF16 = false;
    CommParam commParam;
    QuantParam quantParam;
};

atb::Status RowParallelLinearV2(const ParallelParamV2 &param, atb::Operation **operation);
atb::Status ColumnParallelLinearV2(const ParallelParamV2 &param, atb::Operation **operation);
atb::Status VocabParallelEmbeddingV2(const atb::Operation **operation);
} // namespace common
} // namespace atb_speed

#endif
