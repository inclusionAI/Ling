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
#include "atb_speed/utils/singleton.h"
#include "executor_manager.h"
#include "acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

void AclNNOpCache::Destroy()
{
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: AclNNOpCache addr [" << (this) << "]destroy");
    if (this->aclExecutor == nullptr) { return; }

    // ExecutorManager中的引用减1
    int count = GetSingleton<ExecutorManager>().DecreaseReference(this->aclExecutor);
    if (count != 0) {
        // 如果executor的引用不为0，则不删除executor及其对应的aclTensor
        return;
    }

    // 如果aclExecutor存在且引用为0，则destroy
    int ret = -1;
    ATB_SPEED_LOG_DEBUG("Plugin Op Cache: destroy Executor addr[" << this->aclExecutor << "]");
    if (this->executorRepeatable) {
        // 如果executor可复用，进行destroy；否则不destroy，避免对aclExecutor的重复释放
        ret = aclDestroyAclOpExecutor(this->aclExecutor);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("Plugin Op Cache: destroy Executor failed.");
        }
    }
    this->aclExecutor = nullptr;

    // 清空aclTensor
    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensors.size(); ++i) {
        if (this->aclnnVariantPack.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclDestroyTensor(this->aclnnVariantPack.aclInTensors[i]->tensor);
            if (ret != 0) {
                ATB_SPEED_LOG_ERROR("Plugin Op Cache: destroy aclInTensors " << i << " failed.");
            }
        }
    }
    this->aclnnVariantPack.aclInTensors.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensors.size(); ++i) {
        if (this->aclnnVariantPack.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclDestroyTensor(this->aclnnVariantPack.aclOutTensors[i]->tensor);
            if (ret != 0) {
                ATB_SPEED_LOG_ERROR("Plugin Op Cache: destroy aclOutTensors " << i << " failed.");
            }
        }
    }
    this->aclnnVariantPack.aclOutTensors.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensorList.size(); ++i) {
        ret = aclDestroyTensorList(this->aclnnVariantPack.aclInTensorList[i]);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("Plugin Op Cache: destroy aclInTensorList " << i << " failed.");
        }
    }
    this->aclnnVariantPack.aclInTensorList.clear();

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensorList.size(); ++i) {
        ret = aclDestroyTensorList(this->aclnnVariantPack.aclOutTensorList[i]);
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("Plugin Op Cache: destroy aclOutTensorList " << i << " failed.");
        }
    }
    this->aclnnVariantPack.aclOutTensorList.clear();
}

atb::Status AclNNOpCache::UpdateAclNNVariantPack(const atb::VariantPack &variantPack)
{
    ATB_SPEED_LOG_DEBUG("call UpdateAclNNVariantPack ");
    for (size_t i = 0; i < this->aclnnVariantPack.aclInTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack.aclInTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack.aclInTensors[i]->atbTensor = variantPack.inTensors.at(i);
        if (this->aclnnVariantPack.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetInputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclInTensors[i]->tensor,
                this->aclnnVariantPack.aclInTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicInputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclInTensors[i]->tensorListidx,
                this->aclnnVariantPack.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclInTensorList[this->aclnnVariantPack.aclInTensors[i]->tensorListidx],
                this->aclnnVariantPack.aclInTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("inTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret);
            return atb::ERROR_CANN_ERROR;
        }
    }

    for (size_t i = 0; i < this->aclnnVariantPack.aclOutTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack.aclOutTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack.aclOutTensors[i]->atbTensor = variantPack.outTensors.at(i);
        if (this->aclnnVariantPack.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetOutputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclOutTensors[i]->tensor,
                this->aclnnVariantPack.aclOutTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicOutputTensorAddr(this->aclExecutor,
                this->aclnnVariantPack.aclOutTensors[i]->tensorListidx,
                this->aclnnVariantPack.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack.aclOutTensorList[this->aclnnVariantPack.aclOutTensors[i]->tensorListidx],
                this->aclnnVariantPack.aclOutTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_SPEED_LOG_ERROR("outTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret);
            return atb::ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

} // namespace common
} // namespace atb_speed