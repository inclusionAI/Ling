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
#include "operation_torch.h"
#include <acl/acl.h>
#include <torch/torch.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <atb_speed/utils/singleton.h>
#include <atb_speed/base/context_factory.h>
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/log.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"
#include "operation_creator.h"

namespace atb_speed {
static uint64_t GetNewOpId()
{
    static uint64_t opId = 0;
    uint64_t newOpId = opId++;
    return newOpId;
}

OperationTorch::OperationTorch(std::string opName) : opName_(opName), name_(opName)
{
    opId_ = GetNewOpId();
    nodeId_ = std::to_string(opId_);
    const char *taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
    const char *blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
    
    isTaskQueueEnable_ = !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
        (blockingEnv != nullptr && std::string(blockingEnv) == "1"));
    ATB_SPEED_LOG_DEBUG("OperationTorch::OperationTorch, TASK_QUEUE_ENABLE:" << isTaskQueueEnable_ << ", opName:" <<
        opName << ", opId:" << opId_);
    context_ = atb_speed::ContextFactory::GetAtbContext(Utils::GetCurrentStream());
}

OperationTorch::~OperationTorch()
{
    context_.reset();
    atb_speed::ContextFactory::FreeAtbContext();
}

void OperationTorch::SetName(std::string name) { name_ = name; }

void OperationTorch::SetParam(std::string param)
{
    ATB_SPEED_LOG_DEBUG(name_ << " set param start, param:" << param);
    param_ = param;

    if (isTaskQueueEnable_) {
        runTaskFunc_ = std::bind(&OperationTorch::RunTask, this, std::placeholders::_1, std::placeholders::_2);
    }

    atb::Operation *operation = CreateOperation(opName_, param_);
    if (operation == nullptr) {
        ATB_SPEED_LOG_ERROR(name_ << " create operation fail, opName:" << opName_ << ", param:" << param_);
        return;
    }

    operation_.reset(operation);
    ATB_SPEED_LOG_DEBUG(name_ << " set param end");
}

std::vector<torch::Tensor> OperationTorch::ExecuteImpl(std::vector<torch::Tensor> &atInTensors)
{
    if (!operation_) {
        std::stringstream ss;
        ss << name_ << " execute fail, operation is null. Enable log: "
            << "export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1 to find the first error. "
            << "For more details, see the MindIE official document." << std::endl;
        ATB_SPEED_LOG_ERROR(ss.str(), ATB_MODELS_EXECUTION_FAILURE);
        throw std::runtime_error(ss.str());
    }
    Utils::ContiguousAtTensor(atInTensors);

    std::vector<torch::Tensor> atOutTensors;
    CreateAtOutTensors(atInTensors, atOutTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    return atOutTensors;
}

std::vector<torch::Tensor> OperationTorch::ExecuteWithParam(std::vector<torch::Tensor> atInTensors,
    std::string varaintPackParam)
{
    ATB_SPEED_LOG_DEBUG(name_ << " execute start");
    if (!operation_) {
        SetParam(varaintPackParam);
    }

    std::vector<torch::Tensor> atOutTensors = ExecuteImpl(atInTensors);

    ExecuteOutImpl(atInTensors, atOutTensors, varaintPackParam);
    return atOutTensors;
}

void OperationTorch::ExecuteOutWithParam(std::vector<torch::Tensor> atInTensors,
    std::vector<torch::Tensor> atOutTensors, std::string varaintPackParam)
{
    ATB_SPEED_LOG_DEBUG(name_ << " execute out start");
    if (!operation_) {
        SetParam(varaintPackParam);
    }

    if (!operation_) {
        ATB_SPEED_LOG_ERROR(name_ << " execute out fail, operation is null");
        return;
    }

    Utils::ContiguousAtTensor(atInTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors, varaintPackParam);
}

std::vector<torch::Tensor> OperationTorch::Execute(std::vector<torch::Tensor> atInTensors)
{
    ATB_SPEED_LOG_DEBUG(name_ << " execute start");

    std::vector<torch::Tensor> atOutTensors = ExecuteImpl(atInTensors);

    ExecuteOutImpl(atInTensors, atOutTensors);
    return atOutTensors;
}

void OperationTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors)
{
    ATB_SPEED_LOG_DEBUG(name_ << " execute out start");
    if (!operation_) {
        ATB_SPEED_LOG_ERROR(name_ << " execute out fail, operation is null");
        return;
    }

    Utils::ContiguousAtTensor(atInTensors);
    Utils::ContiguousAtTensor(atOutTensors);
    ExecuteOutImpl(atInTensors, atOutTensors);
}

void OperationTorch::ExecuteOutImpl(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                                    const std::string &varaintPackParam)
{
    Clear();
    ATB_SPEED_LOG_DEBUG(name_ << " execute impl execCount:" << executeCount_);
    if (hostTensorBinder_) {
        nlohmann::json paramJson;
        try {
            paramJson = nlohmann::json::parse(varaintPackParam);
        } catch (const std::exception &e) {
            ATB_SPEED_LOG_ERROR(name_ << " parse json fail, error:" << e.what());
            return;
        }
        hostTensorBinder_->ParseParam(paramJson);
    }

    BuildVariantPack(atInTensors, atOutTensors, variantPack_);

    if (hostTensorBinder_) {
        hostTensorBinder_->BindTensor(variantPack_);
    }

    atb::Status st = operation_->Setup(variantPack_, workspaceSize_, context_.get());
    if (st != 0) {
        ATB_SPEED_LOG_ERROR(name_ << " setup fail, not call execute, error code: " << st);
        return;
    }

    ATB_SPEED_LOG_DEBUG(name_ << " get plan workspace size:" << workspaceSize_);

    if (workspaceSize_ > 0) {
        workspace_ = atb_speed::GetSingleton<atb_speed::Workspace>().GetWorkspaceBuffer(workspaceSize_);
    }

    if (runTaskFunc_) {
        ExecutePlanASync();
    } else {
        ExecutePlan();
    }
}

atb::Status OperationTorch::ExecutePlan()
{
    atb::Status st = operation_->Execute(variantPack_, (uint8_t*)workspace_, workspaceSize_, context_.get());
    executeCount_++;
    return st;
}

void OperationTorch::ExecutePlanASync()
{
    if (runTaskFunc_) {
        runTaskFunc_(name_, [=]() {
            return ExecutePlan();
        });
    }
}

void OperationTorch::Clear()
{
    variantPack_.inTensors.clear();
    variantPack_.outTensors.clear();
    workspaceSize_ = 0;
    workspace_ = nullptr;
}

void OperationTorch::CreateAtOutTensors(const std::vector<torch::Tensor> &atInTensors,
                                        std::vector<torch::Tensor> &atOutTensors)
{
    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.resize(operation_->GetOutputNum());
    atb::SVector<atb::TensorDesc> inTensorDescs;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        auto &atInTensor = atInTensors.at(i);
        atb::Tensor inTensor = Utils::AtTensor2Tensor(atInTensor);
        if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
            if (inTensor.desc.format == ACL_FORMAT_NCHW) {
                inTensor.desc.format = ACL_FORMAT_ND;
            }
        }
        inTensorDescs.push_back(inTensor.desc);
        ATB_SPEED_LOG_DEBUG(name_ <<" infer shape inTensors[" << i
                      <<"]:" << atb_speed::TensorUtil::TensorToString(inTensor));
    }
    atb::Status st = operation_->InferShape(inTensorDescs, outTensorDescs);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR(name_ << " infer shape fail, error code: " << st);
    }
    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(name_ <<" infer shape outTensorDescs[" << i
                      <<"]:" << atb_speed::TensorUtil::TensorDescToString(outTensorDescs.at(i)));
        at::Tensor newTensor = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
        atOutTensors.at(i) = newTensor;
    }
}

void OperationTorch::BuildVariantPack(std::vector<torch::Tensor> &atInTensors, std::vector<torch::Tensor> &atOutTensors,
                                      atb::VariantPack &variantPack)
{
    variantPack.inTensors.resize(atInTensors.size());
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(name_ <<" execute start, atInTensors[" << i << "].options:" << atInTensors.at(i).options()
                      <<", data:" << atInTensors.at(i).data_ptr()
                      <<", storage_offset:" << atInTensors.at(i).storage_offset()
                      <<", format:" << Utils::GetTensorNpuFormat(atInTensors.at(i)));
        if (atb_speed::GetSingleton<atb_speed::Config>().IsTorchTensorFormatCast()) {
            atInTensors.at(i) = Utils::NpuFormatCast(atInTensors.at(i));
        }
        variantPack.inTensors.at(i) = Utils::AtTensor2Tensor(atInTensors.at(i));
        if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND() &&
            variantPack.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack.inTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }

    variantPack.outTensors.resize(atOutTensors.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(name_ <<"execute start, atOutTensors[" << i << "].options:" << atOutTensors.at(i).options()
                      <<", data:" << atOutTensors.at(i).data_ptr()
                      <<", storage_offset:" << atOutTensors.at(i).storage_offset()
                      <<", format:" << Utils::GetTensorNpuFormat(atOutTensors.at(i)));
        variantPack.outTensors.at(i) = Utils::AtTensor2Tensor(atOutTensors.at(i));
        if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND() &&
            variantPack.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
            variantPack.outTensors.at(i).desc.format = ACL_FORMAT_ND;
        }
    }
}

void OperationTorch::RunTask(std::string taskName, std::function<int()> task) const
{
#ifdef TORCH_SETCUSTOMHANDLER
    at_npu::native::OpCommand cmd;
    cmd.Name(taskName);
    cmd.SetCustomHandler(task);
    cmd.Run();
#else
    ATB_SPEED_LOG_ERROR(modelName_ << "torch_npu is low, can't support SetCustomHandler");
#endif
}

TORCH_LIBRARY(OperationTorch, m)
{
    m.class_<OperationTorch>("OperationTorch")
        .def(torch::init<std::string>())
        .def("set_name", &OperationTorch::SetName)
        .def("set_param", &OperationTorch::SetParam)
        .def("execute", &OperationTorch::Execute)
        .def("execute_out", &OperationTorch::ExecuteOut)
        .def("execute_with_param", &OperationTorch::ExecuteWithParam)
        .def("execute_out_with_param", &OperationTorch::ExecuteOutWithParam);
    ;
}
}
