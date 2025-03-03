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
#include "operation.h"

#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

#include "operation_factory.h"
#include "atb_context_factory.h"
#include "utils.h"
#include "config.h"

namespace atb_torch {
Operation::Operation(const std::string &opName) : opName_(opName)
{
    if (opName_.empty()) {
        std::ostringstream ss;
        ss << "Operation_" << this;
        opName_ = ss.str();
    }
}

Operation::~Operation()
{
    if (atbOperation_ && releaseAtbOperation_) {
        atb::DestroyOperation(atbOperation_);
        atbOperation_ = nullptr;
    }

    atbContext_.reset();
    AtbContextFactory::Instance().FreeAtbContext();
}

std::string Operation::SetOpName(const std::string &opName)
{
    opName_ = opName;
    return opName_;
}

std::string Operation::GetOpName() const { return opName_; }

atb::Operation *Operation::GetAtbOperation() { return atbOperation_; }

void Operation::SetAtbOperation(atb::Operation *atbOperation) { atbOperation_ = atbOperation; }

void Operation::SetReleaseAtbOperation(bool release) { releaseAtbOperation_ = release; }

std::vector<std::string> Operation::GetInputNames()
{
    std::vector<std::string> inputNames;
    if (atbOperation_) {
        inputNames.resize(atbOperation_->GetInputNum());
    }
    for (size_t i = 0; i < inputNames.size(); ++i) {
        inputNames[i] = "in" + std::to_string(i);
    }
    return inputNames;
}

std::vector<std::string> Operation::GetOutputNames()
{
    std::vector<std::string> outputNames;
    if (atbOperation_) {
        outputNames.resize(atbOperation_->GetOutputNum());
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
        outputNames[i] = "out" + std::to_string(i);
    }
    return outputNames;
}

TorchTensorList Operation::InferShape(const TorchTensorList &atInTensors)
{
    TorchTensorList atOutTensors;
    if (!atbOperation_) {
        ATB_SPEED_LOG_ERROR(opName_ << " operation is null, infer shape fail");
        return atOutTensors;
    }

    atOutTensors = CreateOutTensors(atInTensors);
    return atOutTensors;
}

void Operation::PreInputTensor(const TorchTensorMap &preInAtTensorMap)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " PreInputTensor start");
    preInAtTensorMap_ = preInAtTensorMap;
    for (auto &it : preInAtTensorMap) {
        ATB_SPEED_LOG_DEBUG(opName_ << " preInAtTensorMap[" << it.first << "] " << Utils::AtTensor2String(it.second));
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " PreInputTensor end");
}

void Operation::PreOutputTensor(const TorchTensorMap &preOutAtTensorMap)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " PreOutputTensor start");
    preOutAtTensorMap_ = preOutAtTensorMap;
    for (auto &it : preOutAtTensorMap) {
        ATB_SPEED_LOG_DEBUG(opName_ << " preOutAtTensorMap[" << it.first << "] "
                            << Utils::AtTensor2String(it.second));
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " PreOutputTensor end");
}

void Operation::PreBindTensor(const TorchTensorMap &preBindAtTensorMap)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " PreBindTensor start");
    preBindAtTensorMap_ = preBindAtTensorMap;
    for (auto &it : preBindAtTensorMap) {
        ATB_SPEED_LOG_DEBUG(opName_ << " preBindAtTensorMap[" << it.first << "] "
                            << Utils::AtTensor2String(it.second));
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " PreBindTensor end");
}

void Operation::SetWeights(const TorchTensorMap &atWeightsMap)
{
    std::map<std::string, int> inNameMap;
    std::vector<std::string> inNames = GetInputNames();
    for (size_t i = 0; i < inNames.size(); ++i) {
        inNameMap[inNames[i]] = i;
    }
    for (auto it = atWeightsMap.begin(); it != atWeightsMap.end(); ++it) {
        CHECK_THROW(inNameMap.find(it->first) == inNameMap.end(),
            opName_ << " error weight key: " << it->first);
        atbWeights_[inNameMap[it->first]] = Utils::AtTensor2Tensor(it->second);
    }
}

TorchTensorMap Operation::Forward(const TorchTensorMap &atInTensorMap, const TorchTensorMap &atOutTensorMap,
                                  const TorchTensorMap &bindTensorMap)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " execute_name start");
    LogTensors(atInTensorMap, atOutTensorMap, bindTensorMap);

    std::map<std::string, int> inNameMap;
    std::map<std::string, int> outNameMap;
    std::vector<std::string> outNames;
    GetInOutNameMap(inNameMap, outNameMap, outNames);

    TorchTensorList atInTensors;
    ConvertTensorMapToTensorList(atInTensorMap, preInAtTensorMap_, inNameMap, atInTensors);

    TorchTensorList atOutTensors;
    ConvertTensorMapToTensorList(atOutTensorMap, preOutAtTensorMap_, outNameMap, atOutTensors);

    TorchTensorList bindTensors;
    ConvertTensorMapToTensorList(bindTensorMap, preBindAtTensorMap_, inNameMap, bindTensors);

    TorchTensorList atRetTensorList = ExecuteImpl(atInTensors, atOutTensors, bindTensors);

    TorchTensorMap atRetTensorMap;
    for (size_t i = 0; i < atRetTensorList.size(); ++i) {
        atRetTensorMap[outNames.at(i)] = atRetTensorList.at(i);
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " execute_name success");
    return atRetTensorMap;
}

TorchTensorList Operation::ExecuteImpl(const TorchTensorList &atInTensors, const TorchTensorList &atOutTensors,
                                       const TorchTensorList &bindTensors)
{
    CheckInput(atInTensors, atOutTensors, bindTensors);

    if (!atbContext_) {
        atbContext_ = AtbContextFactory::Instance().GetAtbContext(Utils::GetCurrentStream());
    }

    if (!atbContext_) {
        ATB_SPEED_LOG_ERROR(opName_ << " atb context is null, execute fail", ATB_MODELS_EXECUTION_FAILURE);
        throw std::runtime_error("atb context is null");
    }

    TorchTensorList atContiguousInTensors = ContiguousTensors(atInTensors);
    TorchTensorList atContiguousOutTensors =
        atOutTensors.empty() ? CreateOutTensors(atInTensors) : ContiguousTensors(atOutTensors);

    std::vector<atb::Tensor> atbInTensors;
    std::vector<atb::Tensor> atbOutTensors;
    ConvertAtTensorToAtbTensor(atContiguousInTensors, atContiguousOutTensors, bindTensors, atbInTensors, atbOutTensors);

    ExecuteAtbTensor(atbInTensors, atbOutTensors);
    return atContiguousOutTensors;
}

void Operation::ExecuteSync(atb::VariantPack &variantPack,
    uint8_t *workspace, uint64_t workspaceSize, atb::Context *atbContext)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " atb operation execute start");
    atb::Status st = atbOperation_->Execute(variantPack, workspace, workspaceSize, atbContext);
    if (st == 0) {
        ATB_SPEED_LOG_DEBUG(opName_ << " atb operation execute success");
    } else {
        ATB_SPEED_LOG_ERROR(opName_ << " atb operation execute fail, error:" << st, ATB_MODELS_EXECUTION_FAILURE);
        CHECK_THROW(st == atb::ERROR_OUT_OF_DEVICE_MEMORY, "Npu out of memory, OOM");
        throw std::runtime_error("atb operation execute fail");
    }
}

void Operation::ExecuteAsync(atb::VariantPack &variantPack,
    uint8_t *workspace, uint64_t workspaceSize, atb::Context *atbContext)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " push atb operation execute task to task queue");
    at_npu::native::OpCommand cmd;
    cmd.Name(opName_);
    cmd.SetCustomHandler([=]() {
        ATB_SPEED_LOG_DEBUG(opName_ << " atb operation execute start");
        atb::Status st = atbOperation_->Execute(variantPack, workspace, workspaceSize, atbContext);
        if (st == 0) {
            ATB_SPEED_LOG_DEBUG(opName_ << " atb operation execute success");
        } else {
            ATB_SPEED_LOG_ERROR(opName_ << " atb operation execute fail, error:" << st, ATB_MODELS_EXECUTION_FAILURE);
        }
        return st;
    });
    cmd.Run();
}

void Operation::ExecuteAtbTensor(const std::vector<atb::Tensor> &atbInTensors,
                                 const std::vector<atb::Tensor> &atbOutTensors)
{
    if (!atbOperation_) {
        ATB_SPEED_LOG_ERROR(opName_ << " atb operation is null, execute fail", ATB_MODELS_EXECUTION_FAILURE);
        throw std::runtime_error("atb operation is null");
    }

    atb::VariantPack variantPack;
    
    atb::SVector<atb::Tensor> ins;
    ins.resize(atbInTensors.size());
    for (size_t i = 0; i < ins.size(); i++) {
        ins.at(i) = atbInTensors.at(i);
    }
    atb::SVector<atb::Tensor> outs;
    outs.resize(atbOutTensors.size());
    for (size_t i = 0; i < outs.size(); i++) {
        outs.at(i) = atbOutTensors.at(i);
    }

    variantPack.inTensors = ins;
    variantPack.outTensors = outs;

    ATB_SPEED_LOG_DEBUG(opName_ << " atb operation setup start");
    uint64_t workspaceSize = 0;
    atb::Status st = atbOperation_->Setup(variantPack, workspaceSize, atbContext_.get());
    CHECK_THROW(st == atb::ERROR_OUT_OF_DEVICE_MEMORY, "Npu out of memory, OOM");
    if (st != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " atb operation setup fail, error:" << st);
        throw std::runtime_error("atb operation setup fail");
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " atb operation setup success, workspace size:" << workspaceSize);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        workspace = CreateWorkspace(workspaceSize);
    }

    atb::Context *atbContext = atbContext_.get();
    if (Config::Instance().IsTaskQueueEnable()) {
        ExecuteAsync(variantPack, (uint8_t *)workspace, workspaceSize, atbContext);
    } else {
        ExecuteSync(variantPack, (uint8_t *)workspace, workspaceSize, atbContext);
    }
}

TorchTensorList Operation::CreateOutTensors(const TorchTensorList &atInTensors)
{
    TorchTensorList atOutTensors;

    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.resize(atbOperation_->GetOutputNum());
    atb::SVector<atb::TensorDesc> inTensorDescs;
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        auto &atInTensor = atInTensors.at(i);
        atb::Tensor inTensor = Utils::AtTensor2Tensor(atInTensor);
        inTensorDescs.push_back(inTensor.desc);
        ATB_SPEED_LOG_DEBUG(opName_ << " infer shape inTensors[" << i << "]:" << Utils::TensorToString(inTensor));
    }
    atb::Status st = atbOperation_->InferShape(inTensorDescs, outTensorDescs);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR(opName_ << " infer shape fail, error code: " << st);
    }

    atOutTensors.resize(outTensorDescs.size());
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(opName_ << " infer shape outTensorDescs[" << i
                            << "]:" << Utils::TensorDescToString(outTensorDescs.at(i)));

        atOutTensors.at(i) = Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(i));
    }

    return atOutTensors;
}

void Operation::CheckInput(const TorchTensorList &atInTensors, const TorchTensorList &atOutTensors,
                           const TorchTensorList &bindTensors) const
{
    ATB_SPEED_LOG_DEBUG("in tensor num: " << atInTensors.size());
    ATB_SPEED_LOG_DEBUG("out tensor num: " << atOutTensors.size());
    ATB_SPEED_LOG_DEBUG("bind tensor num: " << bindTensors.size());
}

TorchTensorList Operation::ContiguousTensors(const TorchTensorList &atTensors) const
{
    TorchTensorList atContiguousInTensors;
    atContiguousInTensors.resize(atTensors.size());
    for (size_t i = 0; i < atTensors.size(); ++i) {
        atContiguousInTensors[i] = atTensors.at(i).is_contiguous() ? atTensors.at(i) : atTensors.at(i).contiguous();
    }
    return atContiguousInTensors;
}

void Operation::ConvertAtTensorToAtbTensor(const TorchTensorList &atInTensors, const TorchTensorList &atOutTensors,
                                           const TorchTensorList &bindTensors, std::vector<atb::Tensor> &atbInTensors,
                                           std::vector<atb::Tensor> &atbOutTensors)
{
    atbInTensors.resize(atInTensors.size());
    for (size_t i = 0; i < atInTensors.size(); ++i) {
        auto &inTensor = atbInTensors.at(i);
        if (atbWeights_.find(i) != atbWeights_.end()) {
            inTensor = atbWeights_[i];
        } else {
            inTensor = Utils::AtTensor2Tensor(atInTensors.at(i));
        }
        void *bindTensorData = nullptr;
        if (bindTensors.at(i).has_storage()) {
            if (atInTensors.at(i).scalar_type() != bindTensors.at(i).scalar_type()) {
                ATB_SPEED_LOG_ERROR(opName_ << " bindTensor's dtype should be the same as inTensor's dtype: "
                    << atInTensors.at(i).scalar_type()  << ", but get " << bindTensors.at(i).scalar_type();
                throw std::runtime_error("bindTensor's dtype is different from inTensor's dtype"));
            }
            bindTensorData = bindTensors.at(i).data_ptr();
        }
        if (bindTensorData != nullptr) {
            ATB_SPEED_LOG_DEBUG(opName_ << " bind atInTensors[" << i << "]" << Utils::TensorToString(inTensor));
            inTensor.hostData = bindTensorData;
        }
    }

    atbOutTensors.resize(atOutTensors.size());
    for (size_t i = 0; i < atOutTensors.size(); ++i) {
        auto &atbOutTensor = atbOutTensors.at(i);
        atbOutTensor = Utils::AtTensor2Tensor(atOutTensors.at(i));
    }
}

void Operation::LogTensors(const TorchTensorMap &atInTensorMap, const TorchTensorMap &atOutTensorMap,
                           const TorchTensorMap &bindTensorMap)
{
    for (auto &it : atInTensorMap) {
        ATB_SPEED_LOG_DEBUG(opName_ << " atInTensorMap[" << it.first << "] " << Utils::AtTensor2String(it.second));
    }

    for (auto &it : atOutTensorMap) {
        ATB_SPEED_LOG_DEBUG(opName_ << " atOutTensorMap[" << it.first << "] " << Utils::AtTensor2String(it.second));
    }

    for (auto &it : bindTensorMap) {
        ATB_SPEED_LOG_DEBUG(opName_ << " bindTensorMap[" << it.first << "] " << Utils::AtTensor2String(it.second));
    }
}

void Operation::GetInOutNameMap(std::map<std::string, int> &inNameMap, std::map<std::string, int> &outNameMap,
                                std::vector<std::string> &outNames)
{
    std::vector<std::string> inNames = GetInputNames();
    for (size_t i = 0; i < inNames.size(); ++i) {
        inNameMap[inNames[i]] = i;
    }

    outNames = GetOutputNames();
    for (size_t i = 0; i < outNames.size(); ++i) {
        outNameMap[outNames[i]] = i;
    }
}

void Operation::ConvertTensorMapToTensorList(const TorchTensorMap &tensorMap, const TorchTensorMap &preTensorMap,
                                             const std::map<std::string, int> &nameMap, TorchTensorList &tensorList)
{
    tensorList.resize(nameMap.size());

    for (auto &i : tensorMap) {
        auto j = nameMap.find(i.first);
        if (j == nameMap.end()) {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid tensor name:" + i.first);
            throw std::invalid_argument("invalid tensor name:" + i.first);
        }
        tensorList.at(j->second) = i.second;
    }

    for (auto &i : preTensorMap) {
        auto j = nameMap.find(i.first);
        if (j == nameMap.end()) {
            ATB_SPEED_LOG_ERROR(opName_ << " invalid tensor name:" + i.first);
            throw std::invalid_argument("invalid tensor name:" + i.first);
        }
        tensorList.at(j->second) = i.second;
    }
}

void *Operation::CreateWorkspace(size_t workspaceSize)
{
    if (Config::Instance().GetGlobalWorkspaceSize() > 0) {
        ATB_SPEED_LOG_DEBUG(opName_ << " use global workspace tensor");
        if (workspaceSize > Config::Instance().GetGlobalWorkspaceSize()) {
            if (aclrtSynchronizeDevice() != 0) {
                return nullptr;
            }
            ATB_SPEED_LOG_DEBUG(opName_ << " new global workspace tensor");
            Config::Instance().SetGlobalWorkspaceSize(workspaceSize);
        }

        return Config::Instance().GetGlobalWorkspaceTensor().data_ptr();
    }

    ATB_SPEED_LOG_DEBUG(opName_
                        << " use temp workspace tensor, unsafe_empty_workspace workspaceSize:" << workspaceSize);
    throw std::runtime_error("temp workspace tensor not implement");
    return nullptr;
}
} // namespace atb_torch
