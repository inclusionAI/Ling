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

#ifndef ATB_SPEED_LAYERS_MLP_GATE_H
#define ATB_SPEED_LAYERS_MLP_GATE_H

#include <atb/atb_infer.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "nlohmann/json.hpp"

#include "common_op_base.h"

namespace atb_speed {
namespace common {
class MlpGateWithBias : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum MlpGateWithBiasId : int {
        IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
        IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_GATE_ID,                  // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
        IN_BIAS_UP_ID,                      //
        IN_BIAS_GATE_ID,                    //
        IN_BIAS_DOWN_ID,                    //
        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
        INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_SPLIT_OUT_ND_ID,
    };
};

class MlpGate : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum MlpGateId : int {
        IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
        IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_GATE_ID,                  // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
        INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, ffnHiddenSize], half
        IN_BIAS_UP_ID,                      // no need
        IN_BIAS_GATE_ID,                    // no need
        IN_BIAS_DOWN_ID,                    // no need
        INTERMEDIATE_SPLIT_OUT_ND_ID,
    };
};

class MlpGateWithPackAndBias : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum MlpGateWithPackId : int {
        IN_HIDDENSTATES_ID = 0, // [batch, seqLen, hiddenSize], half
        IN_WEIGHT_UP_ID,        // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_DOWN_ID,      // [ffnHiddenSize, hiddenSize], half
        IN_BIAS_UP_ID,
        IN_BIAS_DOWN_ID,
        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
        INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_SPLIT_OUT_ND_ID,       // [batch, seqLen, ffnHiddenSize], half
        IN_WEIGHT_GATE_ID,                  // no need
        IN_BIAS_GATE_ID,                    // no need
    };
};

class MlpGateWithPack : public CommonOpBase {
public:
    using CommonOpBase::CommonOpBase;

    enum MlpGateWithPackId : int {
        IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
        IN_WEIGHT_UP_ID,                    // [hiddenSize, ffnHiddenSize], half
        IN_WEIGHT_DOWN_ID,                  // [ffnHiddenSize, hiddenSize], half
        OUT_RESULT_ID,                      // [batch, seqLen, hiddenSize], half
        INTERMEDIATE_MATMUL_GATE_OUT_ND_ID, // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MATMUL_UP_OUT_ND_ID,   // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_ACTIVATION_OUT_ID,     // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, ffnHiddenSize], half
        INTERMEDIATE_SPLIT_OUT_ND_ID,       // [batch, seqLen, ffnHiddenSize], half
        IN_WEIGHT_GATE_ID,                  // no need
        IN_BIAS_UP_ID,                      // no need
        IN_BIAS_GATE_ID,                    // no need
        IN_BIAS_DOWN_ID,                    // no need
    };
};

struct MlpGateParam {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    bool transposeB = true;
    bool isBias = false;
    bool isPack = false;
    std::string backend = "hccl";
    bool isBF16 = false;
};

atb::Status MlpGateLayer(const MlpGateParam &param, atb::Operation **operation);
} // namespace common
} // namespace atb_speed
#endif
