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
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#include "acl_nn_operation_cache.h"

namespace atb_speed {
namespace common {

/// An class that inherited from `atb::Operation` class. An `atb::Operation` class defines a series of
/// interfaces required for operation preparation and execution.
class AclNNOperation : public atb::Operation {
public:
    /// Class constructor.
    ///
    /// Initialize an `AclNNOpCache` pointer for the object's local cache (`aclnnOpCache_`) and set `opName`.
    /// \param opName The name of the AclNN operation.
    explicit AclNNOperation(const std::string &opName);
    ~AclNNOperation() override;
    /// Return the AclNN operation's name.
    /// \return The object's `opName`.
    std::string GetName() const override;
    /// Preparations before operation execution.
    ///
    /// This function calls `UpdateAclNNOpCache` to update `aclnnOpCache_`
    /// and calculate the memory space that needs to be allocated during the operation execution process.
    /// \param variantPack Operation's input and output tensor info.
    /// \param workspaceSize The size of the work space.
    /// \param context The context in which operation's preparation is performed.
    /// \return A status code that indicates whether the setup process was successful.
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
    /// Operation execution process.
    ///
    /// Call `GetExecuteStream` from `context`. Call `UpdateAclNNVariantPack` to update tensor's device data.
    /// Execute the operation.
    /// \param variantPack Operation's input and output tensor info.
    /// \param workspace A pointer the memory address allocated by the operation.
    /// \param workspaceSize The size of the work space.
    /// \param context The context in which operation's preparation is performed.
    /// \return A status code that indicates whether the execute process was successful.
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                        atb::Context *context) override;
    /// Release all occupied resources, particularly those stored in `aclnnOpCache_`.
    void DestroyOperation() const;

protected:
    /// Create the operation's local cache (`aclnnOpCache_`).
    ///
    /// Create the operation's input tensor and output tensor by calling `CreateAclNNVariantPack`.
    /// Call `SetAclNNWorkspaceExecutor` to get work space size and `aclOpExecutor`.
    /// Call `aclSetAclOpExecutorRepeatable` to make `aclOpExecutor` reusable.
    /// \param variantPack Operation's input and output tensor info passed from ATB framework.
    /// \return A status code that indicates whether `aclnnOpCache_` was successfully created.
    atb::Status CreateAclNNOpCache(const atb::VariantPack &variantPack);
    /// Verify if the local cache or global cache is hit. If neither is hit, create a new instance
    /// by calling `CreateAclNNOpCache`, then update both the `ExecutorManager` and `AclNNGlobalCache`.
    /// \param variantPack Operation's input and output tensor info.
    /// \return A status code that indicates whether `aclnnOpCache_` was successfully updated.
    atb::Status UpdateAclNNOpCache(const atb::VariantPack &variantPack);
    /// Prepare the operation's input tensors and output tensors.
    ///
    /// This function calls `CreateAclNNInTensorVariantPack` and `CreateAclNNOutTensorVariantPack`.
    /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
    /// \return A status code that indicates whether variantPack was created successfully.
    virtual atb::Status CreateAclNNVariantPack(const atb::VariantPack &variantPack);
    /// Prepare the operation's input tensors.
    ///
    /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
    /// \return A status code that indicates whether variantPack was created successfully.
    virtual atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack);
    /// Prepare the operation's output tensors.
    ///
    /// \param variantPack An `atb::VariantPack` object containing tensor info passed through ATB framework.
    /// \return A status code that indicates whether variantPack was created successfully.
    virtual atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack);
    /// Call AclNN operation's first phase API to get work space size and `aclOpExecutor`.
    ///
    /// \return The return value of AclNN's first phase API.
    virtual int SetAclNNWorkspaceExecutor() = 0;
    /// Call AclNN operation's second phase API to execute the operation.
    ///
    /// \return The return value of AclNN's second phase API.
    virtual int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) = 0;

    /// An `AclNNOpCache` object that can be reused within the current operation object.
    std::shared_ptr<AclNNOpCache> aclnnOpCache_ = nullptr;
    /// A human identifiable name for the operation's name.
    std::string opName_;
};
} // namespace common
} // namespace atb_speed
#endif