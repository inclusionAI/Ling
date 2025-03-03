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

#ifndef ASCEND_SPEED_INFERENCE_COMMON_LINEAR_PARALLEL_H
#define ASCEND_SPEED_INFERENCE_COMMON_LINEAR_PARALLEL_H

#include <atb/atb_infer.h>
#include "acl/acl.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/linear/linear.h"

namespace atb_speed {
namespace common {

enum LinearParallelType : uint32_t {
    UNDEFINED = 0,
    ROW_PARALLEL,     // all reduce
    COLUMN_PARALLEL,  // all gather
};

struct TensorParallelInfo {
    int rank = 0;
    int worldSize = 1;
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::string commDomain = "";
    atb::infer::AllReduceParam::QuantType quantType = atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED;
    aclDataType outDataType = ACL_DT_UNDEFINED;
};

struct LinearParallelParam {
    atb_speed::common::FusionLinearParam fusionLinearParam;
    int parallelType = UNDEFINED;
    bool biasAfterSync = false;
    bool unpadInputs = false;  // all reduce时不会使用到此参数
    bool supportLcoc = false;
    bool useImMask = false;
    TensorParallelInfo tensorParallelInfo;
};

atb::Status LinearParallel(const LinearParallelParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed

#endif
