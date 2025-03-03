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
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_TENSOR_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_TENSOR_H
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>

namespace atb_speed {
namespace common {

/// A class contains tensor information.
///
/// AclNN operations and ATB operations organize tensor in different format.
/// This class stores the information necessary for easy conversion and tensor usage.
class AclNNTensor {
public:
    /// An enumerator to indicate that the `tensorListidx` is invalid.
    static const int64_t notInTensorList = -1;

    /// Tensor passed through the ATB framework.
    atb::Tensor atbTensor;
    /// The stride of each dimension in the tensor's view shape. Used when creating `aclTensor`.
    atb::SVector<int64_t> strides = {};
    /// Tensor passed into the AclNN operation.
    aclTensor *tensor = nullptr;
    /// The index of the tensor in the tensor list. Used when `aclTensor` is passed into `aclTensorList`.
    int tensorListidx = notInTensorList;
    /// The index of the tensor in `aclOpExecutor`'s parameter list.
    int tensorIdx = -1;
    /// An indicator that shows whether the tensor's device data needs to be updated in the execution.
    bool needUpdateTensorDataPtr = false;
};

} // namespace common
} // namespace atb_speed
#endif