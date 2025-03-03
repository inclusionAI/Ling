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
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_LOCAL_CACHE_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_LOCAL_CACHE_H
#include "acl_nn_tensor.h"

namespace atb_speed {
namespace common {

/// Information about input and output tensors of an AclNN operation.
struct AclNNVariantPack {
    /// A container stores an AclNN operation's in tensor in order.
    /// Each `AclNNTensor` object contains one `aclTensor`.
    atb::SVector<std::shared_ptr<AclNNTensor>> aclInTensors;
    /// A container stores an AclNN operation's out tensor in order.
    /// Each `AclNNTensor` object contains one `aclTensor`.
    atb::SVector<std::shared_ptr<AclNNTensor>> aclOutTensors;
    /// A container stores an AclNN operation's input `aclTensorList` in order.
    /// Each `aclTensorList` object may contain multiple `aclTensor`.
    atb::SVector<aclTensorList *> aclInTensorList;
    /// A container stores an AclNN operation's output `aclTensorList` in order.
    /// Each `aclTensorList` object may contain multiple `aclTensor`.
    atb::SVector<aclTensorList *> aclOutTensorList;
};

/// AclNNOpCache stores information of an operation that can be reused between operations.
struct AclNNOpCache {
    /// Information about input and output tensors of an AclNN operation.
    AclNNVariantPack aclnnVariantPack;
    /// AclNN operation's executor, which contains the operator computation process.
    aclOpExecutor *aclExecutor = nullptr;
    /// An indicator shows whether the `aclOpExecutor` is repeatable.
    bool executorRepeatable = false;
    /// Size of the workspace to be allocated on the device.
    uint64_t workspaceSize;
    /// Update the device memory address in `aclTensor` objects when the device memory changes.
    ///
    /// \param variantPack Information about input and output tensors of an AclNN operation.
    /// \return A status code that indicates whether the update operation was successful.
    atb::Status UpdateAclNNVariantPack(const atb::VariantPack &variantPack);
    /// Destroy resources allocated in `AclNNOpCache`.
    ///
    /// Destory `aclOpExecutor` if it's repeatable and has no reference.
    /// Destory `aclTensor` and `aclTensorList` if `aclOpExecutor` is destroyed.
    void Destroy();
};

} // namespace common
} // namespace atb_speed
#endif