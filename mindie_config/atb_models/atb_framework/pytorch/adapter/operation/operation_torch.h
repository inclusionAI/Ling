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
#ifndef ATB_SPEED_OPERATION_TORCH_H
#define ATB_SPEED_OPERATION_TORCH_H
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/custom_class.h>
#include "atb/operation.h"
#include "atb_speed/base/hosttensor_binder.h"

namespace atb_speed {
class OperationTorch : public torch::CustomClassHolder {
public:
    using Task = std::function<int()>;
    using RunTaskFunc = std::function<void(const std::string &taskName, Task task)>;
    explicit OperationTorch(std::string opName);
    ~OperationTorch() override;
    void SetName(std::string name);
    void SetParam(std::string param);
    std::vector<torch::Tensor> ExecuteWithParam(std::vector<torch::Tensor> atInTensors, std::string varaintPackParam);
    void ExecuteOutWithParam(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                             std::string varaintPackParam);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors);
    void ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors);
    c10::intrusive_ptr<OperationTorch> Clone() const { return c10::make_intrusive<OperationTorch>(opName_); }

private:
    void CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors);
    void ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                        const std::string &varaintPackParam = "");
    std::vector<torch::Tensor> ExecuteImpl(std::vector<torch::Tensor> &atInTensors);
    void BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                          atb::VariantPack &variantPack);
    void RunTask(std::string taskName, std::function<int()> task) const;
    atb::Status ExecutePlan();
    void Clear();
    void ExecutePlanASync();

private:
    std::string opName_;
    uint64_t opId_ = 0;
    std::string nodeId_ = "0";
    std::string name_;
    std::string param_;
    std::unique_ptr<atb::Operation> operation_;
    uint64_t executeCount_ = 0;
    bool isTaskQueueEnable_ = false;
    std::unique_ptr<atb_speed::HostTensorBinder> hostTensorBinder_;
    std::shared_ptr<atb::Context> context_;
    atb::VariantPack variantPack_;
    uint64_t workspaceSize_ = 0;
    void *workspace_ = nullptr;
    RunTaskFunc runTaskFunc_ = nullptr;
};
}
#endif