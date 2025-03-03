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
#ifndef ATB_TORCH_GRAPH_OPERATION_H
#define ATB_TORCH_GRAPH_OPERATION_H
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <atb/atb_infer.h>
#include "operation.h"

namespace atb_torch {
using ReshapeFunc = std::function<std::vector<int64_t>(const std::vector<int64_t> &orgShape)>;

enum class TensorType {
    TENSOR_TYPE_IN = 0,
    TENSOR_TYPE_OUT,
    TENSOR_TYPE_INTERNAL,
};

struct ExecuteNode {
    atb::Operation *atbOperation = nullptr;
    std::vector<atb::Tensor *> inTensors;
    std::vector<atb::Tensor *> outTensors;
    std::vector<atb::ReshapeFunc> inTensorReshapeFuncs;
    atb::SVector<TensorType> inTensorTypes;
    atb::SVector<TensorType> outTensorTypes;

    atb::VariantPack variantPack;
    void *workspace = nullptr;
    uint64_t workspaceSize = 0;
};

struct CachedTorchTensor {
    atb::Tensor* atbTensorPtr;
    torch::Tensor atTensor;
    bool used = false;
    size_t size = 0;
};

struct ExecuteGraph {
    std::vector<atb::Tensor> inTensors;
    std::vector<atb::Tensor> outTensors;
    std::vector<atb::Tensor> internalTensors;
    std::vector<ExecuteNode> nodes;
    std::map<uint64_t, std::set<atb::Tensor *>> maxNodeIdTensorMap;
    std::vector<CachedTorchTensor> cachedTorchTensors;
    void InitTensorMaxNodeMap();
};

class GraphOperation : public Operation {
public:
    explicit GraphOperation(const std::string &opName);
    ~GraphOperation() override;
    std::vector<std::string> GetInputNames() override;
    std::vector<std::string> GetOutputNames() override;
    const atb::GraphParam &GetGraphParam() const;
    int AddInputOutput(const std::vector<std::string> &inTensorNames, const std::vector<std::string> &outTensorNames);
    int AddOperation(Operation *operation, const std::vector<std::string> &inTensorNames,
                     const std::vector<std::string> &outTensorNames);
    int AddReshape(const std::string &orgTensorName, const std::string &newTensorName, const ReshapeFunc &reshapeFunc);
    void Build();
    void SetExecuteAsSingle(bool asSingle) noexcept;
    bool GetExecuteAsSingle() const noexcept;

protected:
    atb::GraphParam atbGraphParam_;
    uint32_t internalTensorNum = 0;
    std::vector<std::string> inTensorNames_;
    std::vector<std::string> outTensorNames_;
    std::map<std::string, uint32_t> inTensorIds_;
    std::map<std::string, uint32_t> outTensorIds_;
    std::map<std::string, uint32_t> internalTensorIds_;
    std::map<std::string, std::pair<uint32_t, atb::ReshapeFunc>> viewTensorIds_; // key: viewTensorName

    std::unique_ptr<ExecuteGraph> executeGraph_;
    std::vector<Operation*> subOperations_;
    bool executeAsSingle_ = true;

protected:
    void ExecuteAtbTensor(const std::vector<atb::Tensor> &inTensors,
                          const std::vector<atb::Tensor> &outTensors) override;

private:
    int32_t GetTensorId(const std::string &tensorName);
    void AtbGraphParam2ExecuteGraph(const atb::GraphParam &atbGraphParam, ExecuteGraph &executeGraph) const;
    void BuildFullTensorPtrs(std::vector<std::pair<atb::Tensor *, TensorType>> &fullTensorPtrs,
                             ExecuteGraph &executeGraph) const;
    void BuildSingleNodeVariantPack(size_t nodeId);
    void FreeGraphInternalTensor(size_t nodeId);
    void ExecuteSingleNode(size_t nodeId);
    at::Tensor MallocInternalTensor(size_t nodeId, size_t outTensorId, const atb::TensorDesc &tensorDesc);
    void FreeInternalTensor(size_t nodeId, const void *tensorDeviceData);
};

} // namespace atb_torch

#endif