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

#ifndef ATB_SPEED_MODELS_COMMON_UITLS_H
#define ATB_SPEED_MODELS_COMMON_UITLS_H

#include <vector>
#include <map>
#include <atb/atb_infer.h>
#include "atb_speed/log.h"


namespace atb_speed {
namespace common {

enum PackQuantType : unsigned int {
    PACK_QUANT_UNDEFINED = 0,
    ALL_FP = 1,
    ALL_W8A8 = 2,
    ALL_W8A8_ANTI = 3,
    MIX_W8A8 = 4,
    MIX_W8A8_ANTI = 5,
    ALL_W8A16 = 6,
    ALL_W8A8SC = 7,
    MIX_W8A8SC = 8,
    ALL_W8A8SC_ANTI = 9,
    MIX_W8A8SC_ANTI = 10,
    ALL_W4A16 = 11,
    ALL_W8A16_ANTI = 12,
    ALL_W4A16_ANTI = 13,
    MIX_W4A16 = 14,
    MIX_W4A16_ANTI = 15,
    MIX_W8A16 = 16,
    MIX_W8A16_ANTI = 17,
    ALL_W8A8_DYNAMIC = 18,
    ALL_W8A8_DYNAMIC_ANTI = 19,
    MIX_W8A8_DYNAMIC = 20,
    MIX_W8A8_DYNAMIC_ANTI = 21
};

enum OpBackend: unsigned int {
    ATB = 0,
    ACLNN = 1,
};

enum LinearQuantType : unsigned int {
    NO_QUANT = 0,
    LINEAR_W8A8_DEQUANT,  // QUANT在RMS_NORM中执行，DEQUANT在此operaion中执行
    LINEAR_W8A8_QUANT,    // QUANT和DEQUANT操作都在此Operation中执行
    W4A16,
    W8A16,
    LINEAR_W8A8_SC_DEQUANT,
    LINEAR_W8A8_SC_QUANT,
    LINEAR_W8A8_DYNAMIC_DEQUANT,
    LINEAR_W8A8_DYNAMIC_QUANT,
};

enum LinearType : int {
    INVALID = -1,
    FP = 0,
    INT = 1,
};

enum TransposeType : int {
    TRANSPOSE_INVALID = -1,
    NOT_TRANSPOSE = 0,
    TRANSPOSE = 1,
};

void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, uint32_t &tensorIdx, std::map<std::string, uint32_t> &tensorMap);
void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, std::map<std::string, uint32_t> &tensorMap);
template <typename T>
void AddTensorToList(
    const std::map<std::string, T> &tensorCandidates,
    std::string targetKey, T &tensorList);
std::map<std::string, uint32_t> GetTensorMap(
    std::vector<std::string> &inTensorList, std::vector<std::string> &outTensorList,
    std::vector<std::string> &intermediateTensorList);
uint32_t GetTensorIdx(const std::map<std::string, uint32_t> &tensorMap, std::string tensorName);
atb::SVector<uint32_t> GetTensorIdxList(const std::map<std::string, uint32_t> &tensorMap,
    std::vector<std::string>tensorNames);

bool CheckAntiOutlier(const int &packQuantType);
bool CheckPack(const int &packQuantType);

atb::Status CheckParamVectorSize(const std::vector<int> &vector, size_t threshold);

PackQuantType ConvertQuantTypeToPackType(std::string quantType);
} // namespace common
} // namespace atb_speed
#endif