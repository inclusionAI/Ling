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

#ifndef ATB_SPEED_LAYERS_PARALLEL_LAYER_H
#define ATB_SPEED_LAYERS_PARALLEL_LAYER_H
#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "nlohmann/json.hpp"
#include "common_op_base.h"

namespace atb_speed {
namespace common {
struct ParallelParam {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    bool isBias = false;
    bool transposeA = false;
    bool transposeB = true;
    std::string backend = "hccl";
    bool isBF16 = false;
};

class LinearWithBiasAndParallel : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearWithBiasAndParallelId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        IN_BIAS,
        OUT_LINEAROUT,
        INTERMIDATE_MATMULOUT,
        INTERMIDATE_ALLREDUCEOUT,
    };
};

class LinearWithParallel : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearWithParallelId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        INTERMIDATE_ALLREDUCEOUT,
        INTERMIDATE_MATMULOUT,
        IN_BIAS,
        OUT_LINEAROUT,
    };
};

class LinearWithBias : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearWithBiasId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        IN_BIAS,
        OUT_LINEAROUT,
        INTERMIDATE_MATMULOUT,
        INTERMIDATE_ALLREDUCEOUT,
    };
};

class LinearOnly : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum LinearOnlyId : unsigned int {
        IN_INPUT = 0,
        IN_WEIGHT,
        INTERMIDATE_MATMULOUT,
        IN_BIAS,
        OUT_LINEAROUT,
        INTERMIDATE_ALLREDUCEOUT,
    };
};

atb::Status RowParallelLinear(const ParallelParam &param, atb::Operation **operation);
atb::Status ColumnParallelLinear(const ParallelParam &param, atb::Operation **operation);
atb::Status VocabParallelEmbedding(const atb::Operation **operation);
} // namespace common
} // namespace atb_speed

#endif
