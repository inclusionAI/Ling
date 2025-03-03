/*
* Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
*
*  * Licensed under the Apache License, Version 2.0 (the "License");
*  * you may not use this file except in compliance with the License.
*  * You may obtain a copy of the License at
*  *
*  * http://www.apache.org/licenses/LICENSE-2.0
*  *
*  * Unless required by applicable law or agreed to in writing, software
*  * distributed under the License is distributed on an "AS IS" BASIS,
*  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  * See the License for the specific language governing permissions and
*  * limitations under the License.
*  */
#ifndef ATB_SPEED_MODELS_SPARSE_MOE_OPERATION_H
#define ATB_SPEED_MODELS_SPARSE_MOE_OPERATION_H
#include <atb/atb_infer.h>
#include <atb/svector.h>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/fusion/utils.h"
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"

namespace atb_speed {
namespace common {

enum SparseMoeIdx : int {
    ROUTER_IDX = 0,
    MOE_MLP_GATE_IDX,
    MOE_MLP_UP_IDX,
    MOE_MLP_DOWN_IDX
};

struct SparseMoeParam {
    atb::SVector<int64_t> axes = {1};
    atb::SVector<int32_t> num = {6}; // num of selected experts
    atb::SVector<int32_t> topkGroups = {3}; // number of groups/device selected
    std::vector<int> moeLinearQuantType = {};
    int numOfGroups = 8; // number of groups in total
    int numOfExperts = 64; // num of experts in total
    int expertParallelDegree = 1;
    float routedScalingFactor = 1.0;
    bool transpose = true;
    bool supportSwiGLU = true;
    bool isBF16 = false;
    std::string routingMethod = "softMaxTopK";
    std::string processLogits = "none";
    bool gateUpTransposeB = false;
    bool downTransposeB = false;
    bool enableFusedRouting = false;
    bool rounterHasBias = false;
    int packQuantType = atb_speed::common::PackQuantType::ALL_FP;
    int denseQuantType = atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    bool useStdNorm = false;
};
atb::Status CreateSparseMoeOperation(const SparseMoeParam &param, atb::Operation **operation);
atb::Status CreateSparseMoemoeGate(
    const SparseMoeParam &param, atb::Node &linearNode, atb::GraphParam opGraph);
atb::Status CreateSparseMoesoftMax(
    const SparseMoeParam &param, atb::Node &softMaxNode, atb::GraphParam opGraph);
atb::Status CreateSparsMoetopK(
    const SparseMoeParam &param, atb::Node &topKNode, atb::GraphParam opGraph);
atb::Status CreateSparseMoereduce(atb::Node &reduceNode, atb::GraphParam opGraph);
atb::Status CreateSparseMoedivide(
    std::shared_ptr<int64_t> batchDimPtr, atb::Node &divideNode, atb::GraphParam opGraph);
atb::Status CreateSparseMoeStd(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId);

atb::Status CreateSparseMoeNorm(
    std::map<std::string, uint32_t> &tensorMap, atb::GraphParam &opGraph, size_t &nodeId);

}
} // namespace atb_speed
#endif