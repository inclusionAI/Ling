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
#include <torch_npu/csrc/framework/OpCommand.h>
#include "operation_factory.h"
#include "config.h"
#include "utils.h"
#include "atb_speed/log.h"
#include "graph_operation.h"

namespace atb_torch {
constexpr int MAX_DIM_NUM = 8;

void ExecuteGraph::InitTensorMaxNodeMap()
{
    for (auto &internalTensor : internalTensors) {
        uint64_t maxNodeId = 0;

        for (size_t nodeId = 0; nodeId < nodes.size(); nodeId++) {
            auto &node = nodes[nodeId];
            for (auto tensor : node.inTensors) {
                if (tensor == &internalTensor) {
                    maxNodeId = std::max(maxNodeId, nodeId);
                }
            }
            for (auto tensor : node.outTensors) {
                if (tensor == &internalTensor) {
                    maxNodeId = std::max(maxNodeId, nodeId);
                }
            }
        }
        maxNodeIdTensorMap[maxNodeId].insert(&internalTensor);
    }
}

static void ReshapeFuncAdapter(const ReshapeFunc &reshapeFunc, const atb::Dims &oldShape, atb::Dims &newShape)
{
    std::vector<int64_t> orgShape(oldShape.dimNum);
    for (size_t i = 0; i < orgShape.size(); ++i) {
        orgShape[i] = oldShape.dims[i];
    }

    std::vector<int64_t> viewShape = reshapeFunc(orgShape);
    ATB_SPEED_LOG_DEBUG("reshape tensor, orgShape:" << orgShape << ", viewShape:" << viewShape);

    if (viewShape.size() > MAX_DIM_NUM) {
        ATB_SPEED_LOG_ERROR("invalid view shape, max dim num: " << MAX_DIM_NUM);
        return;
    }

    newShape.dimNum = viewShape.size();
    for (size_t i = 0; i < viewShape.size(); ++i) {
        newShape.dims[i] = viewShape[i];
    }
}

static std::string JoinInts(const atb::SVector<uint32_t> &ids)
{
    std::string ret;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i == 0) {
            ret.append(std::to_string(ids.at(i)));
        } else {
            ret.append(", " + std::to_string(ids.at(i)));
        }
    }
    return ret;
}

static std::string AtbGraphParam2Str(const atb::GraphParam &atbGraphParam)
{
    std::stringstream ss;
    ss << "inTensorNum:" << atbGraphParam.inTensorNum << ", outTensorNum:" << atbGraphParam.outTensorNum
       << ", internalTensorNum:" << atbGraphParam.internalTensorNum;
    for (size_t i = 0; i < atbGraphParam.nodes.size(); ++i) {
        ss << "\nnode[" << i << "]: operation:" << atbGraphParam.nodes.at(i).operation << ", inTensorIds:["
           << JoinInts(atbGraphParam.nodes.at(i).inTensorIds) << "], outTensorIds:["
           << JoinInts(atbGraphParam.nodes.at(i).outTensorIds) << "]";
    }
    return ss.str();
}

GraphOperation::GraphOperation(const std::string &opName) : Operation(opName)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " construct start, opName:" << opName_);
}

GraphOperation::~GraphOperation()
{
    ATB_SPEED_LOG_DEBUG(opName_ << " disconstruct");
}

std::vector<std::string> GraphOperation::GetInputNames()
{
    return inTensorNames_;
}

std::vector<std::string> GraphOperation::GetOutputNames()
{
    return outTensorNames_;
}

const atb::GraphParam &GraphOperation::GetGraphParam() const
{
    ATB_SPEED_LOG_DEBUG(opName_ << " get atb graph op param:\n" << AtbGraphParam2Str(atbGraphParam_));
    return atbGraphParam_;
}

int GraphOperation::AddInputOutput(
    const std::vector<std::string> &inTensorNames, const std::vector<std::string> &outTensorNames)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " input output inTensorNames:" << inTensorNames
                  << ", outTensorNames:" << outTensorNames);
    inTensorNames_ = inTensorNames;
    outTensorNames_ = outTensorNames;

    atbGraphParam_.name = opName_;
    uint32_t id = 0;
    for (const std::string &inTensorName : inTensorNames) {
        inTensorIds_[inTensorName] = id++;
    }
    for (const std::string &outTensorName : outTensorNames) {
        outTensorIds_[outTensorName] = id++;
    }
    atbGraphParam_.inTensorNum = inTensorIds_.size();
    atbGraphParam_.outTensorNum = outTensorIds_.size();

    return 0;
}

int GraphOperation::AddOperation(
    Operation *operation, const std::vector<std::string> &inTensorNames, const std::vector<std::string> &outTensorNames)
{
    if (!operation) {
        ATB_SPEED_LOG_ERROR(opName_ << " add operation fail, operation is null");
        return -1;
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " add operation:" << operation << ", opName:" << operation->GetOpName());

    atb::Node node;
    node.operation = operation->GetAtbOperation();
    operation->SetReleaseAtbOperation(false);
    subOperations_.push_back(operation);
    node.inTensorIds.resize(0);
    node.outTensorIds.resize(0);
    node.inTensorReshapeFuncs.resize(0);
    for (const std::string &inTensorName : inTensorNames) {
        node.inTensorIds.push_back(GetTensorId(inTensorName));
        if (viewTensorIds_.find(inTensorName) != viewTensorIds_.end()) {
            node.inTensorReshapeFuncs.push_back(viewTensorIds_[inTensorName].second);
        } else {
            node.inTensorReshapeFuncs.push_back(nullptr);
        }
    }
    for (const std::string &outTensorName : outTensorNames) {
        node.outTensorIds.push_back(GetTensorId(outTensorName));
    }
    atbGraphParam_.nodes.push_back(node);
    atbGraphParam_.internalTensorNum = internalTensorIds_.size();

    return 0;
}

int GraphOperation::AddReshape(
    const std::string &orgTensorName, const std::string &newTensorName, const ReshapeFunc &reshapeFunc)
{
    ATB_SPEED_LOG_DEBUG(opName_ << " add reshape orgTensorName:" << orgTensorName
                      << ", newTensorName:" << newTensorName);
    viewTensorIds_[newTensorName] = {GetTensorId(orgTensorName),
        std::bind(ReshapeFuncAdapter, reshapeFunc, std::placeholders::_1, std::placeholders::_2)};
    return 0;
}

void GraphOperation::Build()
{
    if (executeAsSingle_) {
        ATB_SPEED_LOG_DEBUG(opName_ << " build atb graph operation start");
        atb::Status st = atb::CreateOperation(atbGraphParam_, &atbOperation_);
        if (st != 0) {
            ATB_SPEED_LOG_ERROR(opName_ << " build atb graph operation fail, error:" << st);
            throw std::runtime_error("build atb graph operation fail");
        }
        if (!atbOperation_) {
            ATB_SPEED_LOG_ERROR(opName_ << " build atb graph operation fail, operation is null");
            throw std::runtime_error("build atb graph operation fail");
        }
    } else {
        for (Operation *operation : subOperations_) {
            operation->SetReleaseAtbOperation(true);
        }
    }
}

void GraphOperation::SetExecuteAsSingle(bool asSingle) noexcept
{
    executeAsSingle_ = asSingle;
}

bool GraphOperation::GetExecuteAsSingle() const noexcept
{
    return executeAsSingle_;
}

void GraphOperation::ExecuteAtbTensor(
    const std::vector<atb::Tensor> &inTensors, const std::vector<atb::Tensor> &outTensors)
{
    if (executeAsSingle_) {
        ATB_SPEED_LOG_DEBUG(opName_ << " execute graph as single");
        Operation::ExecuteAtbTensor(inTensors, outTensors);
        return;
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " execute graph operation start");
    if (!executeGraph_) {
        ATB_SPEED_LOG_DEBUG(opName_ << " new ExecuteGraph");
        executeGraph_ = std::make_unique<ExecuteGraph>();
        if (!executeGraph_) {
            ATB_SPEED_LOG_ERROR(opName_ << " new ExecuteGraph fail");
            throw std::runtime_error("new ExecuteGraph fail");
        }

        ATB_SPEED_LOG_DEBUG(opName_ << " AtbGraphParam convert to ExecuteGraph");
        AtbGraphParam2ExecuteGraph(atbGraphParam_, *executeGraph_);
        executeGraph_->InitTensorMaxNodeMap();
    }

    if (executeGraph_->inTensors.size() != inTensors.size()) {
        ATB_SPEED_LOG_ERROR(opName_ << " execute in tensor num:" << inTensors.size()
                       << " != graph in tensor num:" << executeGraph_->inTensors.size());
        throw std::runtime_error("in tensor num not equal graph in tensor");
    }

    if (executeGraph_->outTensors.size() != outTensors.size()) {
        ATB_SPEED_LOG_ERROR(opName_ << " execute out tensor num:" << outTensors.size()
                       << " != graph out tensor num:" << executeGraph_->outTensors.size());
        throw std::runtime_error("out tensor num not equal graph out tensor");
    }

    executeGraph_->cachedTorchTensors.clear();
    executeGraph_->inTensors = inTensors;
    executeGraph_->outTensors = outTensors;

    ATB_SPEED_LOG_DEBUG(opName_ << " execute all node start");
    for (size_t nodeId = 0; nodeId < executeGraph_->nodes.size(); ++nodeId) {
        BuildSingleNodeVariantPack(nodeId);
        ExecuteSingleNode(nodeId);
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " execute all node success");
}

int32_t GraphOperation::GetTensorId(const std::string &tensorName)
{
    if (inTensorIds_.find(tensorName) != inTensorIds_.end()) {
        return inTensorIds_[tensorName];
    } else if (outTensorIds_.find(tensorName) != outTensorIds_.end()) {
        return outTensorIds_[tensorName];
    } else if (viewTensorIds_.find(tensorName) != viewTensorIds_.end()) {
        return viewTensorIds_[tensorName].first;
    } else if (internalTensorIds_.find(tensorName) != internalTensorIds_.end()) {
        return internalTensorIds_[tensorName];
    } else {
        int32_t internalTensorId = inTensorIds_.size() + outTensorIds_.size() + internalTensorNum++;
        internalTensorIds_[tensorName] = internalTensorId;
        return internalTensorId;
    }
}

void GraphOperation::AtbGraphParam2ExecuteGraph(const atb::GraphParam &atbGraphParam, ExecuteGraph &executeGraph) const
{
    executeGraph.inTensors.resize(atbGraphParam.inTensorNum);
    executeGraph.outTensors.resize(atbGraphParam.outTensorNum);
    executeGraph.internalTensors.resize(atbGraphParam.internalTensorNum);
    executeGraph.nodes.resize(atbGraphParam.nodes.size());

    std::vector<std::pair<atb::Tensor *, TensorType>> fullTensorPtrs(
        atbGraphParam.inTensorNum + atbGraphParam.outTensorNum + atbGraphParam.internalTensorNum);
    BuildFullTensorPtrs(fullTensorPtrs, executeGraph);

    for (size_t i = 0; i < atbGraphParam.nodes.size(); ++i) {
        auto &atbNode = atbGraphParam.nodes[i];
        auto &executeNode = executeGraph.nodes[i];
        executeNode.atbOperation = atbNode.operation;
        executeNode.inTensors.resize(atbNode.inTensorIds.size());
        executeNode.outTensors.resize(atbNode.outTensorIds.size());
        executeNode.inTensorReshapeFuncs.resize(atbNode.inTensorReshapeFuncs.size());
        executeNode.inTensorTypes.resize(atbNode.inTensorIds.size());
        executeNode.outTensorTypes.resize(atbNode.outTensorIds.size());
        executeNode.variantPack.inTensors.resize(atbNode.inTensorIds.size());
        executeNode.variantPack.outTensors.resize(atbNode.outTensorIds.size());

        for (size_t j = 0; j < atbNode.inTensorIds.size(); ++j) {
            executeNode.inTensors[j] = fullTensorPtrs.at(atbNode.inTensorIds[j]).first;
            executeNode.inTensorTypes[j] = fullTensorPtrs.at(atbNode.inTensorIds[j]).second;
        }
        for (size_t j = 0; j < atbNode.outTensorIds.size(); ++j) {
            executeNode.outTensors[j] = fullTensorPtrs.at(atbNode.outTensorIds[j]).first;
            executeNode.outTensorTypes[j] = fullTensorPtrs.at(atbNode.outTensorIds[j]).second;
        }
        for (size_t j = 0; j < atbNode.inTensorReshapeFuncs.size(); ++j) {
            executeNode.inTensorReshapeFuncs[j] = atbNode.inTensorReshapeFuncs[j];
        }
    }
}

void GraphOperation::BuildFullTensorPtrs(
    std::vector<std::pair<atb::Tensor *, TensorType>> &fullTensorPtrs, ExecuteGraph &executeGraph) const
{
    size_t offset = 0;
    for (size_t i = 0; i < executeGraph.inTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = std::make_pair(&executeGraph.inTensors.at(i), TensorType::TENSOR_TYPE_IN);
    }
    for (size_t i = 0; i < executeGraph.outTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = std::make_pair(&executeGraph.outTensors.at(i), TensorType::TENSOR_TYPE_OUT);
    }
    for (size_t i = 0; i < executeGraph.internalTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = std::make_pair(&executeGraph.internalTensors.at(i),
            TensorType::TENSOR_TYPE_INTERNAL);
    }
}

void GraphOperation::BuildSingleNodeVariantPack(size_t nodeId)
{
    ExecuteNode &node = executeGraph_->nodes[nodeId];

    ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] infer shape start");
    atb::SVector<atb::TensorDesc> inTensorDescs;
    inTensorDescs.resize(node.variantPack.inTensors.size());
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        CHECK_THROW(node.inTensors.at(i) == nullptr, " nodes[" << nodeId << "] inTensors[" << i << "] is null");
        node.variantPack.inTensors.at(i) = *node.inTensors.at(i);
        if (i < node.inTensorReshapeFuncs.size() && node.inTensorReshapeFuncs[i]) {
            atb::Dims newShape;
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] inTensors[" << i
                          << "] reshape start, org:" << Utils::TensorToString(node.variantPack.inTensors.at(i)));
            node.inTensorReshapeFuncs[i](node.variantPack.inTensors.at(i).desc.shape, newShape);
            node.variantPack.inTensors.at(i).desc.shape = newShape;
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] inTensors[" << i
                          << "] reshape end, new:" << Utils::TensorToString(node.variantPack.inTensors.at(i)));
        } else {
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] inTensors[" << i
                          << "]:" << Utils::TensorToString(node.variantPack.inTensors.at(i)));
        }

        inTensorDescs.at(i) = node.variantPack.inTensors.at(i).desc;
    }

    atb::SVector<atb::TensorDesc> outTensorDescs;
    outTensorDescs.resize(node.atbOperation->GetOutputNum());
    atb::Status st = node.atbOperation->InferShape(inTensorDescs, outTensorDescs);
    CHECK_THROW(st != 0,
        opName_ << " nodes[" << nodeId
                << "] infershape  fail, enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1");
    for (size_t i = 0; i < outTensorDescs.size(); ++i) {
        ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] outTensors[" << i
                      << "]:" << Utils::TensorDescToString(outTensorDescs.at(i)));
    }
    ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] infer shape end");

    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        CHECK_THROW(node.inTensors.at(i) == nullptr, " nodes[" << nodeId << "] outTensors[" << i << "] is null");
        node.variantPack.outTensors.at(i) = *node.outTensors.at(i);
        if (node.outTensorTypes.at(i) == TensorType::TENSOR_TYPE_INTERNAL) {
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] outTensors[" << i << "] is internal tensor");
            at::Tensor atInternalTensor = MallocInternalTensor(nodeId, i, outTensorDescs.at(i));
            node.variantPack.outTensors.at(i) = Utils::AtTensor2Tensor(atInternalTensor);
            node.variantPack.outTensors.at(i).desc = outTensorDescs.at(i);
            *node.outTensors.at(i) = node.variantPack.outTensors.at(i);
        } else {
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] outTensors[" << i << "] not internal tensor");
        }
    }
    FreeGraphInternalTensor(nodeId);
}

void GraphOperation::FreeGraphInternalTensor(size_t nodeId)
{
    auto it = executeGraph_->maxNodeIdTensorMap.find(nodeId);
    if (it != executeGraph_->maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            FreeInternalTensor(nodeId, tensorIt->deviceData);
        }
    }
}

void GraphOperation::ExecuteSingleNode(size_t nodeId)
{
    ExecuteNode &node = executeGraph_->nodes[nodeId];
    if (node.inTensorReshapeFuncs.size() > 0) {
        for (size_t i = 0; i < node.inTensors.size() && node.inTensorReshapeFuncs.at(i) != nullptr; i++) {
            node.inTensorReshapeFuncs.at(i)(node.inTensors.at(i)->desc.shape, node.inTensors.at(i)->desc.shape);
        }
    }

    atb::Context *atbContext = atbContext_.get();

    ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] atb operation setup start" << node.workspaceSize);
    atb::Status st = node.atbOperation->Setup(node.variantPack, node.workspaceSize, atbContext);
    CHECK_THROW(st != 0,
        opName_ << " nodes[" << nodeId
                << "] setup  fail, enable log: export ASDOPS_LOG_LEVEL=ERROR, export ASDOPS_LOG_TO_STDOUT=1");

    ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId
                  << "] atb operation setup success, workspace size:" << node.workspaceSize);

    if (node.workspaceSize > 0) {
        node.workspace = CreateWorkspace(node.workspaceSize);
    }

    if (Config::Instance().IsTaskQueueEnable()) {
        ExecuteNode *executeNode = &node;
        at_npu::native::OpCommand cmd;
        cmd.Name(opName_);
        cmd.SetCustomHandler([=]() {
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] atb operation execute start");
            atb::Status st = executeNode->atbOperation->Execute(
                executeNode->variantPack, (uint8_t *)executeNode->workspace, executeNode->workspaceSize, atbContext);
            if (st == 0) {
                ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] atb operation execute success");
            } else {
                ATB_SPEED_LOG_ERROR(opName_ << " nodes[" << nodeId << "] atb operation execute fail, error:" << st, \
                    ATB_MODELS_EXECUTION_FAILURE);
            }
            return st;
        });
        cmd.Run();
    } else {
        st = node.atbOperation->Execute(node.variantPack, (uint8_t *)(node.workspace), node.workspaceSize, atbContext);
        if (st == 0) {
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] atb operation execute success");
        } else {
            ATB_SPEED_LOG_ERROR(opName_ << " nodes[" << nodeId << "] atb operation execute fail, error:" << st, \
                ATB_MODELS_EXECUTION_FAILURE);
            throw std::runtime_error("Atb operation setup fail");
        }
    }
}

at::Tensor GraphOperation::MallocInternalTensor(size_t nodeId, size_t outTensorId, const atb::TensorDesc &tensorDesc)
{
    static std::map<at::ScalarType, aclDataType> dtypeMap = {
        {at::ScalarType::Bool, ACL_BOOL},
        {at::ScalarType::Byte, ACL_UINT8},
        {at::ScalarType::Char, ACL_INT8},
        {at::ScalarType::Half, ACL_FLOAT16},
        {at::ScalarType::Float, ACL_FLOAT},
        {at::ScalarType::Int, ACL_INT32},
        {at::ScalarType::Long, ACL_INT64},
        {at::ScalarType::BFloat16, ACL_BF16},
    };
    size_t tensorSize = atb::Utils::GetTensorSize(tensorDesc);
    auto atbTensorPtr = executeGraph_->nodes[nodeId].outTensors.at(outTensorId);
    size_t cacheTensorId = 0;
    for (cacheTensorId = 0; cacheTensorId < executeGraph_->cachedTorchTensors.size(); ++cacheTensorId) {
        auto &cachedTorchTensor = executeGraph_->cachedTorchTensors[cacheTensorId];
        if (cachedTorchTensor.atbTensorPtr == atbTensorPtr) { // write inplace
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] write inplace");
            return cachedTorchTensor.atTensor;
        }
        if (cachedTorchTensor.used) {  // Tensor被使用中，不能被分配其他Op
            continue;
        }

        aclFormat format = static_cast<aclFormat>(Utils::GetTensorNpuFormat(cachedTorchTensor.atTensor));
        auto it = dtypeMap.find(cachedTorchTensor.atTensor.scalar_type());
        if (it == dtypeMap.end()) {
            ATB_SPEED_LOG_ERROR(opName_ << " nodes[" << nodeId << "] atb operation execute fail, dtype error", \
                ATB_MODELS_EXECUTION_FAILURE);
            throw std::runtime_error("dtype error, check ATB_LOG");
        }
        if (tensorSize == cachedTorchTensor.size && tensorDesc.dtype == it->second && format == tensorDesc.format) {
            cachedTorchTensor.used = true;
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] use old internal torch tensor[" << cacheTensorId
                          << "]");
            return cachedTorchTensor.atTensor;
        }
    }

    ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId << "] create internal torch tensor["
                  << executeGraph_->cachedTorchTensors.size() << "], outTensor[" << outTensorId
                  << "]:" << Utils::TensorDescToString(tensorDesc)
                  << ", dataSize:" << atb::Utils::GetTensorSize(tensorDesc));
    CachedTorchTensor cachedTorchTensor;
    cachedTorchTensor.atTensor = Utils::CreateAtTensorFromTensorDesc(tensorDesc);
    cachedTorchTensor.used = true;
    cachedTorchTensor.size = tensorSize;
    cachedTorchTensor.atbTensorPtr = atbTensorPtr;
    executeGraph_->cachedTorchTensors.push_back(cachedTorchTensor);
    return cachedTorchTensor.atTensor;
}

void GraphOperation::FreeInternalTensor(size_t nodeId, const void *tensorDeviceData)
{
    for (size_t cacheTensorId = 0; cacheTensorId < executeGraph_->cachedTorchTensors.size(); ++cacheTensorId) {
        auto &cachedTorchTensor = executeGraph_->cachedTorchTensors[cacheTensorId];
        if (cachedTorchTensor.atTensor.data_ptr() == tensorDeviceData) {
            cachedTorchTensor.used = false;
            ATB_SPEED_LOG_DEBUG(opName_ << " nodes[" << nodeId
                << "] free internal torch tensor[" << cacheTensorId << "]");
            break;
        }
    }
}
}  // namespace atb_torch
