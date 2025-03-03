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

#include "model_torch.h"

#include <acl/acl.h>
#include <atb/utils.h>
#include <torch/torch.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "atb_speed/base/context_factory.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/config.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/tensor_util.h"
#include "atb_speed/utils/operation_util.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
void *ModelTorch::GetWorkSpace(const uint64_t bufferSize) const
{
    void *workspace = nullptr;
    if (bufferSize > 0) {
        workspace = atb_speed::GetSingleton<atb_speed::Workspace>().GetWorkspaceBuffer(bufferSize);
    }
    return workspace;
}

atb::Tensor ModelTorch::CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc)
{
    torch::Tensor newAtTensor = Utils::CreateAtTensorFromTensorDesc(tensorDesc);
    atInternalTensors_.push_back(newAtTensor);
    return Utils::AtTensor2Tensor(newAtTensor);
}

void ModelTorch::RunTask(std::string taskName, std::function<int()> task) const
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

static uint64_t GetNewModelId()
{
    static uint64_t modelId = 0;
    uint64_t newModelId = modelId++;
    return newModelId;
}

ModelTorch::ModelTorch(std::string modelName) : modelName_(modelName)
{
    modelId_ = GetNewModelId();
    context_ = atb_speed::ContextFactory::GetAtbContext(Utils::GetCurrentStream());
    ATB_SPEED_LOG_DEBUG("ModelTorch new modelName:" << modelName_ << ", modelId:" << modelId_);
}

ModelTorch::~ModelTorch()
{
    model_.reset();
    context_.reset();
    atb_speed::ContextFactory::FreeAtbContext();
};

int64_t ModelTorch::SetParam(std::string param)
{
    ATB_SPEED_LOG_DEBUG("ModelTorch set param start, modelName:" << modelName_ << ", param:" << param);

    model_ = atb_speed::ModelFactory::CreateInstance(modelName_, param);
    if (model_ != nullptr) {
        ATB_SPEED_LOG_DEBUG("Get model from the ModelFactory, " << modelName_
                        << ". If other models also want to be obtained from the ModelFactory, "
                        << "please register it and set `namespace` and `model class name`. "
                        << "Examples: REGISTER_MODEL(chatglm2_6b, ChatGlm2CommonModelFa). "
                        << "And then set `chatglm2_6b_ChatGlm2CommonModelFa` as input modelName_.");
    } else {
        ATB_SPEED_LOG_ERROR("Not support modelName: " << modelName_ << ", not found in ModelFactory.");
        throw std::runtime_error("Not support modelName, check ATB_LOG");
    }

    const char *taskQueueEnv = std::getenv("TASK_QUEUE_ENABLE");
    const char *blockingEnv = std::getenv("ASCEND_LAUNCH_BLOCKING");
    bool isTaskQueueEnable = !((taskQueueEnv != nullptr && std::string(taskQueueEnv) == "0") ||
        (blockingEnv != nullptr && std::string(blockingEnv) == "1"));
    auto getWorkspaceFunc = std::bind(&ModelTorch::GetWorkSpace, this, std::placeholders::_1);
    auto createInternalTensorFromDescFunc =
        std::bind(&ModelTorch::CreateInternalTensorFromDesc, this, std::placeholders::_1);
    auto runTaskFunc = std::bind(&ModelTorch::RunTask, this, std::placeholders::_1, std::placeholders::_2);
    int64_t atbStatus = 0;
    if (isTaskQueueEnable) {
        atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc, runTaskFunc);
    } else {
        atbStatus = model_->Init(getWorkspaceFunc, createInternalTensorFromDescFunc, nullptr);
    }
    ATB_SPEED_LOG_DEBUG("ModelTorch set param end");
    return atbStatus;
}

int64_t ModelTorch::SetWeight(std::vector<torch::Tensor> atWeightTensors)
{
    if (model_ == nullptr) {
        throw std::runtime_error("you should set model param first");
    }
    ATB_SPEED_LOG_DEBUG("ModelTorch set weight:" << atWeightTensors.size());
    for (size_t i = 0; i < atWeightTensors.size(); ++i) {
        const torch::Tensor &atTensor = atWeightTensors.at(i);
        ATB_SPEED_LOG_DEBUG("ModelTorch atWeightTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options());
    }
    std::vector<atb::Tensor> weigthTensors;
    AtTensor2Tensor(atWeightTensors, weigthTensors);
    return model_->SetWeight(weigthTensors);
}

int64_t ModelTorch::SetKVCache(std::vector<torch::Tensor> atKCacheTensors, std::vector<torch::Tensor> atVCacheTensors)
{
    if (model_ == nullptr) {
        throw std::runtime_error("you should set model param first");
    }
    ATB_SPEED_LOG_DEBUG("ModelTorch set k cache tensors:" << atKCacheTensors.size()
                  << ", v cache tensors:" << atVCacheTensors.size());
    for (size_t i = 0; i < atKCacheTensors.size(); ++i) {
        const torch::Tensor &atkTensor = atKCacheTensors.at(i);
        ATB_SPEED_LOG_DEBUG("ModelTorch atKCacheTensors[" << i << "]"
                      << " data:" << atkTensor.data_ptr() << ", storage_offset:" << atkTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atkTensor) << ", shape:" << atkTensor.sizes()
                      << ", options:" << atkTensor.options());
        const torch::Tensor &atvTensor = atVCacheTensors.at(i);
        ATB_SPEED_LOG_DEBUG("ModelTorch atVCacheTensors[" << i << "]"
                      << " data:" << atvTensor.data_ptr() << ", storage_offset:" << atvTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atvTensor) << ", shape:" << atvTensor.sizes()
                      << ", options:" << atvTensor.options());
    }

    std::vector<atb::Tensor> kCacheTensors;
    std::vector<atb::Tensor> vCacheTensors;
    AtTensor2Tensor(atKCacheTensors, kCacheTensors);
    AtTensor2Tensor(atVCacheTensors, vCacheTensors);

    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (auto &kCacheTensor : kCacheTensors) {
            if (kCacheTensor.desc.format == ACL_FORMAT_NCHW) {
                kCacheTensor.desc.format = ACL_FORMAT_ND;
            }
        }
        for (auto &vCacheTensor : vCacheTensors) {
            if (vCacheTensor.desc.format == ACL_FORMAT_NCHW) {
                vCacheTensor.desc.format = ACL_FORMAT_ND;
            }
        }
    }
    int64_t atbStatus = model_->SetKVCache(kCacheTensors, vCacheTensors);
    return atbStatus;
}

std::vector<torch::Tensor> ModelTorch::Execute(std::vector<torch::Tensor> atInTensors, std::string param)
{
    CHECK_THROW(model_ == nullptr, "you should set model param first");
    atInternalTensors_.clear();
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        const torch::Tensor &atTensor = atInTensors.at(i);
        ATB_SPEED_LOG_DEBUG("ModelTorch atInTensors[" << i << "]"
                      << " data:" << atTensor.data_ptr() << ", storage_offset:" << atTensor.storage_offset()
                      << ", format:" << Utils::GetTensorNpuFormat(atTensor) << ", shape:" << atTensor.sizes()
                      << ", options:" << atTensor.options());
    }

    std::vector<atb::Tensor> inTensors;
    AtTensor2Tensor(atInTensors, inTensors);
    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (auto &inTensor : inTensors) {
            if (inTensor.desc.format == ACL_FORMAT_NCHW) {
                inTensor.desc.format = ACL_FORMAT_ND;
            }
        }
    }
    std::vector<atb::TensorDesc> inTensorDescs(model_->GetInputNum());
    for (size_t i = 0; i < inTensors.size(); ++i) {
        inTensorDescs.at(i) = inTensors.at(i).desc;
    }
    std::vector<atb::TensorDesc> outTensorDescs(model_->GetOutputNum());
    atb::Status st = model_->InferShape(inTensorDescs, outTensorDescs);
    CHECK_THROW(st != 0, "infer shape fail, enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1");

    std::vector<torch::Tensor> atOutTensors(outTensorDescs.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        ATB_SPEED_LOG_DEBUG("ModelTorch outTensorDescs[" << i
                      << "]:" << atb_speed::TensorUtil::TensorDescToString(outTensorDescs.at(i)));
        atOutTensors.at(i) = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
            atb::Utils::GetTensorSize(outTensorDescs.at(i));
    }

    std::vector<atb::Tensor> outTensors;
    AtTensor2Tensor(atOutTensors, outTensors);
    if (atb_speed::GetSingleton<atb_speed::Config>().IsConvertNCHWToND()) {
        for (auto &outTensor : outTensors) {
            if (outTensor.desc.format == ACL_FORMAT_NCHW) {
                outTensor.desc.format = ACL_FORMAT_ND;
            }
        }
    }

    int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
    if (atbStatus != atb::NO_ERROR) {
        std::vector<torch::Tensor> atNullOutTensors(0);
        return atNullOutTensors;
    }
    return atOutTensors;
}

int64_t ModelTorch::ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
                               std::string param)
{
    atInternalTensors_.clear();
    std::vector<atb::Tensor> inTensors;
    AtTensor2Tensor(atInTensors, inTensors);

    std::vector<atb::Tensor> outTensors;
    AtTensor2Tensor(atOutTensors, outTensors);

    int64_t atbStatus = ExecuteOutImpl(inTensors, outTensors, param);
    return atbStatus;
}

int64_t ModelTorch::ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                                   const std::string &param)
{
    int64_t atbStatus = model_->Execute(context_.get(), inTensors, outTensors, param);
    executeCount_++;
    return atbStatus;
}

void ModelTorch::AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors) const
{
    for (auto &atTensor : atTensors) {
        Utils::ContiguousAtTensor(atTensor);
        atb::Tensor tensor = Utils::AtTensor2Tensor(atTensor);
        opsTensors.push_back(tensor);
    }
}

TORCH_LIBRARY(ModelTorch, m)
{
    m.class_<ModelTorch>("ModelTorch")
        .def(torch::init<std::string>())
        .def("set_param", &ModelTorch::SetParam)
        .def("set_weight", &ModelTorch::SetWeight)
        .def("set_kv_cache", &ModelTorch::SetKVCache)
        .def("execute", &ModelTorch::Execute)
        .def("execute_out", &ModelTorch::ExecuteOut);
}
}
