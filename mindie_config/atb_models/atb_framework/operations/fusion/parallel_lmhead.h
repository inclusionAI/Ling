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
#ifndef ATB_SPEED_LAYERS_PARALLEL_LMHEAD_LAYER_H
#define ATB_SPEED_LAYERS_PARALLEL_LMHEAD_LAYER_H
#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>

#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "common_op_base.h"

namespace atb_speed {
namespace common {
struct ParallelLmHeadParam {
    int rank = 0;
    int rankSize = 1;
    std::string backend = "hccl";
    bool unpadInputs = false;
    bool gatherAhead = false;
    bool transposeA = false;
    bool transposeB = true;
    bool isBF16 = false;
};

class ParallelLmHeadConfig : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum ParallelLmHeadId : unsigned int {
        IN_HIDDENSTATES_ID = 0,
        IN_WEIGHT_ID,
        OUT_LOGITS_ID,
        INTERMEDIATE_ALLGATHER_OUT_ID,
        IN_LMHEAD_INDICES_ID,
        INTERMIDATE_GATHER_OUT_ID,
    };
};

class ParallelLmHeadGatherAheadConfig : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum ParallelLmHeadGatherAheadId : unsigned int {
        IN_HIDDENSTATES_ID = 0,
        IN_WEIGHT_ID,
        IN_LMHEAD_INDICES_ID,
        OUT_LOGITS_ID,
        INTERMIDATE_GATHER_OUT_ID,
        INTERMEDIATE_ALLGATHER_OUT_ID,
    };
};

atb::Status ParallelLmHead(const ParallelLmHeadParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif