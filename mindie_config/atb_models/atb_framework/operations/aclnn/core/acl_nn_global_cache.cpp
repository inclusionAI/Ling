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
#include <sstream>
#include "atb_speed/log.h"
#include "atb_speed/utils/operation_util.h"
#include "operations/aclnn/utils/utils.h"
#include "acl_nn_global_cache.h"

namespace atb_speed {
namespace common {

AclNNGlobalCache::AclNNGlobalCache()
{
    const char *envStr = std::getenv("ATB_ACLNN_CACHE_GLOABL_COUNT");
    uint64_t globalCacheCountMax = DEFAULT_ACLNN_GLOBAL_CACHE_SIZE;
    if (envStr != nullptr) {
        globalCacheCountMax = static_cast<uint64_t>(strtol(envStr, nullptr, DECIMAL));
    }
    envStr = std::getenv("MINDIE_ACLNN_CACHE_GLOBAL_COUNT");
    if (envStr != nullptr) {
        globalCacheCountMax = static_cast<uint64_t>(strtol(envStr, nullptr, DECIMAL));
    }

    this->globalCacheCountMax_ = globalCacheCountMax;
    if (this->globalCacheCountMax_ >= 100) {  // 100: threshold
        std::stringstream ss;
        ss << "The size of AclNN operations' global cache should be less than 100." << std::endl;
        throw std::runtime_error(ss.str());
    }
}

std::shared_ptr<AclNNOpCache> AclNNGlobalCache::GetGlobalCache(std::string opName, atb::VariantPack variantPack)
{
    // 获取Op对应的Global Cache列表
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it = \
        this->aclnnGlobalCache_.find(opName);
    if (it == this->aclnnGlobalCache_.end()) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] not found in AclNNGlobalCache");
        return nullptr;
    }
    std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;

    // 在Global Cache列表中基于variantPack找到匹配的Cache
    for (size_t i = 0; i < opGlobalCacheList.size(); i++) {
        if (opGlobalCacheList[i] == nullptr) {
            ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Global Cache index " << i << " is nullptr");
            continue;
        }
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Global Cache index " << i << " call IsVariankPackEqual");
        if (opGlobalCacheList[i]->executorRepeatable && \
            IsVariankPackEqual(opGlobalCacheList[i]->aclnnVariantPack, variantPack)) {
            // Global Cache命中
            return opGlobalCacheList[i];
        }
    }

    return nullptr;
}

atb::Status AclNNGlobalCache::UpdateGlobalCache(std::string opName, std::shared_ptr<AclNNOpCache> cache)
{
    // 若Local Cache中Executor不可复用，不更新Global Cache
    if (!cache->executorRepeatable) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] not repeatable, do not update global cache");
        return atb::NO_ERROR;
    }
    
    // 获取Op对应的Global Cache列表
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it = \
        this->aclnnGlobalCache_.find(opName);
    if (it == this->aclnnGlobalCache_.end()) {
        // 不存在opName对应的Cache列表
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] not found in AclNNGlobalCache, add one");
        this->aclnnGlobalCache_[opName] = {cache};
        return atb::NO_ERROR;
    }
    std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;

    // Cache未已满
    if (opGlobalCacheList.size() < this->globalCacheCountMax_) {
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << opName << "] global cache is not full, add one");
        opGlobalCacheList.push_back(cache);
        return atb::NO_ERROR;
    }

    // Cache已满
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name["
                  << opName << "] global cache is full, update index " << nextUpdateIndex_);
    opGlobalCacheList[nextUpdateIndex_] = cache;
    CHECK_PARAM_NE(globalCacheCountMax_, 0);
    nextUpdateIndex_ = (nextUpdateIndex_ + 1) % globalCacheCountMax_;
    return atb::NO_ERROR;
}

std::string AclNNGlobalCache::PrintGlobalCache()
{
    std::stringstream ss;
    ss << "Plugin Op Cache: Global Cache Summary ";
    std::map<std::string, std::vector<std::shared_ptr<AclNNOpCache>>>::iterator it;
    for (it = this->aclnnGlobalCache_.begin(); it != this->aclnnGlobalCache_.end(); it++) {
        ss << "Op name[" << it->first << "] ";
        std::vector<std::shared_ptr<AclNNOpCache>> &opGlobalCacheList = it->second;
        for (size_t i = 0; i < opGlobalCacheList.size(); i++) {
            ss << "Cache Addr[" << opGlobalCacheList[i].get() << "] ";
        }
    }
    return ss.str();
}

} // namespace common
} // namespace atb_speed
