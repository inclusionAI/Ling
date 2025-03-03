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
#include "atb_speed/log.h"
#include "atb_speed/utils/timer.h"
#include "atb_speed/utils/statistic.h"
#include "operations/aclnn/utils/utils.h"
#include "atb_speed/utils/singleton.h"
#include "executor_manager.h"
#include "acl_nn_global_cache.h"
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {

AclNNOperation::AclNNOperation(const std::string &opName) : opName_(opName)
{
    this->aclnnOpCache_ = std::make_shared<AclNNOpCache>();
}

AclNNOperation::~AclNNOperation()
{
    ATB_SPEED_LOG_DEBUG("AclNNOperation deconstructor");
    this->DestroyOperation();
}

std::string AclNNOperation::GetName() const { return this->opName_; }

void AclNNOperation::DestroyOperation() const
{
    this->aclnnOpCache_->Destroy();
}

atb::Status AclNNOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " setup start");

    // 1. 检查Context是否为空
    if (context == nullptr) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " setup context is null");
        return atb::ERROR_INVALID_PARAM;
    }

    // 2. 获取Executor和Workspace
    int ret = UpdateAclNNOpCache(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call UpdateAclNNOpCache, error:" << ret);
        this->aclnnOpCache_->Destroy();
        return ret;
    }

    // 3. 更新传入的workspaceSize
    workspaceSize = this->aclnnOpCache_->workspaceSize;

    ATB_SPEED_LOG_DEBUG("setup end");
    ATB_SPEED_LOG_DEBUG(GetSingleton<AclNNGlobalCache>().PrintGlobalCache());
    ATB_SPEED_LOG_DEBUG(GetSingleton<ExecutorManager>().PrintExecutorCount());
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::UpdateAclNNOpCache(const atb::VariantPack &variantPack)
{
    // 此方法会准备好Execute时所需的Executor和workspace
    // 前提条件：GlobalCache中的executor要保证LocalCache里面一定也要有引用；仅对LocalCache进行释放

    // 1. 查看Local Cache中Executor是否可以复用
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Local Cache call IsVariankPackEqual");
    if (this->aclnnOpCache_->executorRepeatable && \
        IsVariankPackEqual(this->aclnnOpCache_->aclnnVariantPack, variantPack)) {
        // Local Cache命中
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" <<
            (this) << "] Cache addr[" << this->aclnnOpCache_.get() << "] Executor addr[" <<
            this->aclnnOpCache_->aclExecutor << "] Local Cache Hit");
        return atb::NO_ERROR;
    }

    // 2. 查看Global Cache中Executor是否可以复用
    std::shared_ptr<AclNNOpCache> globalCache = \
        GetSingleton<AclNNGlobalCache>().GetGlobalCache(this->opName_, variantPack);
    if (globalCache != nullptr) {
        // Global Cache命中
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" << (this) << "] Cache addr["
            << globalCache.get() << "] Executor addr[" << globalCache->aclExecutor << "] Global Cache Hit");
        // 2.1 释放旧的Local Cache
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: destroy local cache before switching to global cache");
        this->aclnnOpCache_->Destroy();
        // 2.2 更新Local Cache
        this->aclnnOpCache_ = globalCache;
        // 2.3 更新ExecutorManager
        int count = GetSingleton<ExecutorManager>().IncreaseReference(this->aclnnOpCache_->aclExecutor);
        ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Executor addr[" <<
            this->aclnnOpCache_->aclExecutor << "] count update to " << count);
        return atb::NO_ERROR;
    }

    // 3. Local Cache和Global Cache都未命中
    // 3.1 释放Local Cache
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: destroy local cache before create a new one");
    this->aclnnOpCache_->Destroy();
    // 3.2 根据variantPack，更新aclnnOpCache_，获取WorkSpace和Executor
    this->aclnnOpCache_ = std::make_shared<AclNNOpCache>();
    int ret = CreateAclNNOpCache(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call CreateAclNNOpCache fail, error:" << ret);
        return ret;
    }
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] Op addr[" <<
        (this) << "] Cache addr[" << this->aclnnOpCache_.get() << "] Executor addr[" <<
        this->aclnnOpCache_->aclExecutor << "] create Local Cache");
    // 3.3 更新ExecutorManager，新增Executor，count为1
    int count = GetSingleton<ExecutorManager>().IncreaseReference(this->aclnnOpCache_->aclExecutor);
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: Op name[" << this->opName_ << "] increase Executor addr[" <<
        this->aclnnOpCache_->aclExecutor << "] count update to " << count);

    // 3.4 更新Global Cache（旧的Global Cache直接替换指针就行）
    GetSingleton<AclNNGlobalCache>().UpdateGlobalCache(this->opName_, this->aclnnOpCache_);

    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNOpCache(const atb::VariantPack &variantPack)
{
    atb::Status ret = CreateAclNNVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call CreateAclNNVariantPack fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    ret = SetAclNNWorkspaceExecutor();
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call SetAclNNWorkspaceExecutor fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    // 若此时Local Cache为空
    if (this->aclnnOpCache_ == nullptr) {
        ATB_SPEED_LOG_ERROR("Plugin Op Cache: Op name[" << this->opName_ << "] cache is nullptr after " <<
            "initialization, please check.");
        return atb::ERROR_INTERNAL_ERROR;
    }

    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: create Executor addr[" << this->aclnnOpCache_->aclExecutor << "]");

    // 设置Local Cache中的aclExecutor为可复用，设置成功返回0，否则返回其他值
    ret = aclSetAclOpExecutorRepeatable(this->aclnnOpCache_->aclExecutor);
    if (ret != 0) {
        // 设置算子可复用失败，标记Local Cache中executor不可复用
        ATB_SPEED_LOG_WARN(this->opName_ << " call aclSetAclOpExecutorRepeatable fail: " << ret);
        this->aclnnOpCache_->executorRepeatable = false;
    } else {
        // 设置算子可复用成功，标记Local Cache中executor可复用
        this->aclnnOpCache_->executorRepeatable = true;
    }
    
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                                    atb::Context *context)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " execute start");
    if (!context) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, context param is null. Enable log: "
            << "export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = context->GetExecuteStream();
    if (!stream) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " execute fail, execute stream in context is null. "
            << "Enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl, ATB_MODELS_EXECUTION_FAILURE);
        return atb::ERROR_INVALID_PARAM;
    }

    // 更新数据传入的地址
    int ret = this->aclnnOpCache_->UpdateAclNNVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call UpdateAclNNVariantPack fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    ATB_SPEED_LOG_DEBUG("Input workspaceSize " << workspaceSize << " localCache workspaceSize " <<
        this->aclnnOpCache_->workspaceSize);
    ret = ExecuteAclNNOp(workspace, stream);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " call ExecuteAclNNOp fail, error:" << ret);
        return atb::ERROR_CANN_ERROR;
    }

    ATB_SPEED_LOG_DEBUG(this->opName_ << " execute end");

    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG(this->opName_ << " CreateAclNNVariantPack start");
    atb::Status ret = 0;
    ret = CreateAclNNInTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNInTensorVariantPack fail");
        return ret;
    }

    ret = CreateAclNNOutTensorVariantPack(variantPack);
    if (ret != 0) {
        ATB_SPEED_LOG_ERROR(this->opName_ << " AclNNTensor CreateAclNNOutTensorVariantPack fail");
        return ret;
    }
    
    ATB_SPEED_LOG_DEBUG(opName_ << " CreateAclNNVariantPack end");
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        aclnnVariantPack.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i), i);
        if (aclnnVariantPack.aclInTensors[i]->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

atb::Status AclNNOperation::CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclOutTensors.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < aclnnVariantPack.aclOutTensors.size(); ++i) {
        aclnnVariantPack.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i), i);
        if (aclnnVariantPack.aclOutTensors[i]->tensor == nullptr) {
            return atb::ERROR_INTERNAL_ERROR;
        }
    }
    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed
