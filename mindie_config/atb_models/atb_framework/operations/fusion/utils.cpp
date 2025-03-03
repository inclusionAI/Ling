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

#include "operations/fusion/utils.h"

namespace atb_speed {
namespace common {

void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, uint32_t &tensorIdx, std::map<std::string, uint32_t> &tensorMap)
{
    if (tensorCandidates.find(targetKey) == tensorCandidates.end()) {
        ATB_SPEED_LOG_WARN("targetKey: " << targetKey << " not found in tensorCandidates");
        return;
    }

    for (std::string tensor : tensorCandidates.at(targetKey)) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }
}

void AssignTensorIdx(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, std::map<std::string, uint32_t> &tensorMap)
{
    if (tensorCandidates.find(targetKey) == tensorCandidates.end()) {
        ATB_SPEED_LOG_WARN("targetKey: " << targetKey << " not found in tensorCandidates");
        return;
    }

    uint32_t startTensorIdx = tensorMap.size();
    for (std::string tensor : tensorCandidates.at(targetKey)) {
        tensorMap[tensor] = startTensorIdx;
        startTensorIdx++;
    }
}

template <typename T>
void AddTensorToList(
    const std::map<std::string, T> &tensorCandidates,
    std::string targetKey, T &tensorList)
{
    if (tensorCandidates.find(targetKey) == tensorCandidates.end()) {
        ATB_SPEED_LOG_WARN("targetKey: " << targetKey << " not found in tensorCandidates");
        return;
    }

    for (const auto& item : tensorCandidates.at(targetKey)) {
        tensorList.push_back(item);
    }
}

std::map<std::string, uint32_t> GetTensorMap(
    std::vector<std::string> &inTensorList, std::vector<std::string> &outTensorList,
    std::vector<std::string> &intermediateTensorList)
{
    std::map<std::string, uint32_t> tensorMap = {};
    uint32_t tensorIdx = 0;

    // 添加inTensor
    for (const auto &tensor : inTensorList) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }

    // 添加outTensor
    for (const auto &tensor : outTensorList) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }

    // 添加intermediateTensor
    for (const auto &tensor : intermediateTensorList) {
        tensorMap[tensor] = tensorIdx;
        tensorIdx++;
    }

    std::stringstream ss;
    for (auto tensor = tensorMap.cbegin(); tensor != tensorMap.cend(); ++tensor) {
        ss << "tensor name: " << tensor->first << ", tensor id: " << tensor->second << std::endl;
    }
    ATB_SPEED_LOG_DEBUG("tensor map" << ss.str());

    return tensorMap;
}

uint32_t GetTensorIdx(const std::map<std::string, uint32_t> &tensorMap, std::string tensorName)
{
    if (tensorMap.find(tensorName) == tensorMap.end()) {
        ATB_SPEED_LOG_DEBUG("Cannot find " << tensorName << " in tensor Map");
        return UINT32_MAX;
    }
    return tensorMap.at(tensorName);
}

atb::SVector<uint32_t> GetTensorIdxList(const std::map<std::string, uint32_t> &tensorMap,
    std::vector<std::string>tensorNames)
{
    atb::SVector<uint32_t> tensorIdxList = {};
    for (std::string tensorName : tensorNames) {
        tensorIdxList.push_back(GetTensorIdx(tensorMap, tensorName));
    }
    return tensorIdxList;
}

bool CheckAntiOutlier(const int &packQuantType)
{
    bool isAntiOutlier = packQuantType == atb_speed::common::MIX_W8A8_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A8_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A8SC_ANTI || \
        packQuantType == atb_speed::common::MIX_W8A8SC_ANTI || \
        packQuantType == atb_speed::common::ALL_W8A16_ANTI || \
        packQuantType == atb_speed::common::MIX_W8A16_ANTI || \
        packQuantType == atb_speed::common::ALL_W4A16_ANTI || \
        packQuantType == atb_speed::common::MIX_W4A16_ANTI;
    return isAntiOutlier;
}

bool CheckPack(const int &packQuantType)
{
    bool isPack = packQuantType != atb_speed::common::MIX_W8A8 && \
        packQuantType != atb_speed::common::MIX_W8A8_ANTI && \
        packQuantType != atb_speed::common::MIX_W8A8SC && \
        packQuantType != atb_speed::common::MIX_W8A8SC_ANTI && \
        packQuantType != atb_speed::common::MIX_W8A16 && \
        packQuantType != atb_speed::common::MIX_W8A16_ANTI && \
        packQuantType != atb_speed::common::MIX_W4A16 && \
        packQuantType != atb_speed::common::MIX_W4A16_ANTI;
    return isPack;
}

atb::Status CheckParamVectorSize(const std::vector<int> &vector, size_t threshold)
{
    if (vector.size() < threshold) {
        return atb::ERROR_INVALID_PARAM;
    }
    return atb::NO_ERROR;
}

PackQuantType ConvertQuantTypeToPackType(std::string quantType)
{
    const std::unordered_map<std::string, atb_speed::common::PackQuantType> quantTypeToPackType = {
        {"float", atb_speed::common::PackQuantType::ALL_FP},
        {"w8a8", atb_speed::common::PackQuantType::ALL_W8A8},
        {"w8a8s", atb_speed::common::PackQuantType::ALL_W8A8},
        {"w8a8sc", atb_speed::common::PackQuantType::ALL_W8A8SC},
        {"w8a8_dynamic", atb_speed::common::PackQuantType::ALL_W8A8_DYNAMIC},
        {"w8a16", atb_speed::common::PackQuantType::ALL_W8A16},
        {"w4a16", atb_speed::common::PackQuantType::ALL_W4A16},
        {"", atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED},
    };

    auto it = quantTypeToPackType.find(quantType);
    if (it == quantTypeToPackType.end()) {
        return atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED;
    }

    return it->second;
}

template void AddTensorToList(
    const std::map<std::string, std::vector<std::string>> &tensorCandidates,
    std::string targetKey, std::vector<std::string> &tensorList);
template void AddTensorToList(
    const std::map<std::string, atb::SVector<std::string>> &tensorCandidates,
    std::string targetKey, atb::SVector<std::string> &tensorList);
} // namespace common
} // namespace atb_speed