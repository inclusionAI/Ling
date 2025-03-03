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
#ifndef ATB_TORCH_OPERATION_H
#define ATB_TORCH_OPERATION_H
#include <string>
#include <vector>
#include <unordered_map>
#include <torch/script.h>
#include <atb/atb_infer.h>

namespace atb_torch {
using TorchTensorList = std::vector<torch::Tensor>;
using TorchTensorMap = std::unordered_map<std::string, torch::Tensor>;

struct ExecuteResult {
    int status;
    std::string errorMsg;
};

class Operation {
public:
    explicit Operation(const std::string &opName);
    virtual ~Operation();
    std::string SetOpName(const std::string &opName);
    std::string GetOpName() const;
    virtual std::vector<std::string> GetInputNames();
    virtual std::vector<std::string> GetOutputNames();
    virtual TorchTensorList InferShape(const TorchTensorList &atInTensors);
    void PreInputTensor(const TorchTensorMap &preInAtTensorMap);
    void PreOutputTensor(const TorchTensorMap &preOutAtTensorMap);
    void PreBindTensor(const TorchTensorMap &preBindAtTensorMap);
    void SetWeights(const TorchTensorMap &atWeightsMap);
    TorchTensorMap Forward(const TorchTensorMap &atInTensorMap, const TorchTensorMap &atOutTensorMap,
                           const TorchTensorMap &bindTensorMap);

    atb::Operation *GetAtbOperation();
    void SetAtbOperation(atb::Operation *atbOperation);
    void SetReleaseAtbOperation(bool release);

protected:
    void *CreateWorkspace(size_t workspaceSize);
    virtual void ExecuteAtbTensor(const std::vector<atb::Tensor> &atbInTensors,
                                  const std::vector<atb::Tensor> &atbOutTensors);

protected:
    std::string opName_;
    atb::Operation *atbOperation_ = nullptr;
    bool releaseAtbOperation_ = true;
    std::shared_ptr<atb::Context> atbContext_;
    TorchTensorMap preInAtTensorMap_;
    TorchTensorMap preOutAtTensorMap_;
    TorchTensorMap preBindAtTensorMap_;
    std::map<int, atb::Tensor> atbWeights_;

private:
    TorchTensorList ExecuteImpl(const TorchTensorList &atInTensors, const TorchTensorList &atOutTensors,
                                const TorchTensorList &bindTensors);
    TorchTensorList CreateOutTensors(const TorchTensorList &atInTensors);
    void CheckInput(const TorchTensorList &atInTensors, const TorchTensorList &atOutTensors,
                    const TorchTensorList &bindTensors) const;
    TorchTensorList ContiguousTensors(const TorchTensorList &atTensors) const;
    void ConvertAtTensorToAtbTensor(const TorchTensorList &atInTensors, const TorchTensorList &atOutTensors,
                                    const TorchTensorList &bindTensors, std::vector<atb::Tensor> &atbInTensors,
                                    std::vector<atb::Tensor> &atbOutTensors);
    void LogTensors(const TorchTensorMap &atInTensorMap, const TorchTensorMap &atOutTensorMap,
                    const TorchTensorMap &bindTensorMap);
    void GetInOutNameMap(std::map<std::string, int> &inNameMap, std::map<std::string, int> &outNameMap,
                         std::vector<std::string> &outNames);
    void ConvertTensorMapToTensorList(const TorchTensorMap &tensorMap, const TorchTensorMap &preTensorMap,
                                      const std::map<std::string, int> &nameMap, TorchTensorList &tensorList);
    void ExecuteSync(atb::VariantPack &variantPack,
                    uint8_t *workspace, uint64_t workspaceSize, atb::Context *atbContext);
    void ExecuteAsync(atb::VariantPack &variantPack,
                    uint8_t *workspace, uint64_t workspaceSize, atb::Context *atbContext);
};

#define CHECK_THROW(condition, message) \
    do { \
        if (condition) { \
            std::stringstream ss; \
            ss << message << std::endl; \
            throw std::runtime_error(ss.str()); \
        } \
    } while (0)

} // namespace atb_torch

#endif