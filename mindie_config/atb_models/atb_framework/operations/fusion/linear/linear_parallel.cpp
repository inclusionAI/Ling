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
#include <cmath>
#include <numeric>
#include "atb_speed/log.h"
#include "operations/fusion/utils.h"
#include "atb_speed/utils/check_util.h"
#include "operations/fusion/linear/linear_parallel.h"

namespace atb_speed {
namespace common {

std::map<std::string, std::vector<std::string>> GetLinearParallelInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearPrallelInTensorCandidates = {
        {"default", {
            "in_input", "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"}
        },
        {"reduce_quant", {
            "in_reduce_quant_scale", "in_reduce_quant_offset", "in_gather_quant_scale", "in_gather_quant_offset"}
        },
        {"lora", {"in_seq_len_cum_sum", "in_lora_a", "in_lora_b"}},
        {"lora_with_mask", {"in_im_mask"}}
    };
    return linearPrallelInTensorCandidates;
}

std::map<std::string, std::vector<std::string>> GetLinearParallelIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> linearPrallelIntermediateTensorCandidates = {
        {"linear_out", {"intermediate_linear_out"}},
        {"sync_out", {"intermediate_sync_out"}},
        {"quant_out", {"intermediate_quant_out"}},
    };
    return linearPrallelIntermediateTensorCandidates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const LinearParallelParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum,
    bool enableLcoc)
{
    auto linearPrallelInTensorCandidates = GetLinearParallelInTensorCandidates();
    auto linearPrallelIntermediateTensorCandidates = GetLinearParallelIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    AddTensorToList(linearPrallelInTensorCandidates, "default", inTensorList);

    // 添加额外的中间Tensor
    if (enableLcoc) {
        if (param.biasAfterSync) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "sync_out", intermediateTensorList);
        }
    } else {
        AddTensorToList(linearPrallelIntermediateTensorCandidates, "linear_out", intermediateTensorList);
        if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "quant_out", intermediateTensorList);
            }
        // All gather场景下卡间通信的输出无法原地写
        if (param.parallelType == COLUMN_PARALLEL) {
            AddTensorToList(linearPrallelIntermediateTensorCandidates, "sync_out", intermediateTensorList);
        }
    }

    // 添加Lora特性的Tensor
    if (param.fusionLinearParam.supportLora) {
        if (param.fusionLinearParam.useImMask) {
            AddTensorToList(linearPrallelInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(linearPrallelInTensorCandidates, "lora", inTensorList);
    }
    // 添加lccl reduce int8特性的Tensor
    if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
        AddTensorToList(linearPrallelInTensorCandidates, "reduce_quant", inTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

int64_t AddAllReduceOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb::Node allReduceNode;
    atb::infer::AllReduceParam allReduceParam;
    allReduceParam.rank = param.tensorParallelInfo.rank;
    allReduceParam.rankSize = param.tensorParallelInfo.worldSize;
    allReduceParam.backend = param.tensorParallelInfo.backend;
    allReduceParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
    allReduceParam.quantType = param.tensorParallelInfo.quantType;
    allReduceParam.outDataType = param.tensorParallelInfo.outDataType;
    allReduceParam.commDomain = param.tensorParallelInfo.commDomain;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allReduceParam, &allReduceNode.operation));
    if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL) {
        bool isQuant = param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT;
        std::vector<std::string> allReduceInTensors = {
            isQuant ? "intermediate_quant_out" : "intermediate_linear_out", \
            "in_reduce_quant_scale", "in_gather_quant_offset"
        };
        allReduceNode.inTensorIds = {GetTensorIdxList(tensorMap, allReduceInTensors)};
    } else {
        allReduceNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
    }
    allReduceNode.outTensorIds = {
        GetTensorIdx(tensorMap, param.biasAfterSync ? "intermediate_linear_out" : "out")
    };
    opGraph.nodes.push_back(allReduceNode);

    if (param.biasAfterSync) {
        atb::Node addNode;
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
        addNode.inTensorIds = GetTensorIdxList(tensorMap, {
            "intermediate_linear_out", "in_bias"
        });
        addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(addNode);
    }
    return atb::NO_ERROR;
}

int64_t AddCommunicationOp(const LinearParallelParam &param, atb::GraphParam &opGraph,
    std::map<std::string, uint32_t> &tensorMap)
{
    if (param.parallelType == ROW_PARALLEL) {
        CHECK_OPERATION_STATUS_RETURN(AddAllReduceOp(param, opGraph, tensorMap));
    } else {
        atb::Node allGatherNode;
        atb::infer::AllGatherParam allGatherParam;
        allGatherParam.rank = param.tensorParallelInfo.rank;
        allGatherParam.rankSize = param.tensorParallelInfo.worldSize;
        allGatherParam.backend = param.tensorParallelInfo.backend;
        allGatherParam.rankTableFile = param.tensorParallelInfo.rankTableFile;
        allGatherParam.commDomain = param.tensorParallelInfo.commDomain;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(allGatherParam, &allGatherNode.operation));
        allGatherNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
        allGatherNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_sync_out")};
        opGraph.nodes.push_back(allGatherNode);

        atb::Node transposeNode;
        atb::infer::TransposeParam transposeParam;
        if (param.unpadInputs) {
            transposeParam.perm = {1, 0, 2};
        } else {
            transposeParam.perm = {1, 2, 0, 3};
        }
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(transposeParam, &transposeNode.operation));
        transposeNode.inTensorIds = {GetTensorIdx(tensorMap, "intermediate_sync_out")};
        transposeNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(transposeNode);
    }
    return atb::NO_ERROR;
}

void LinearParallelInferShape(atb::GraphParam &opGraph, const LinearParallelParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        uint32_t inputIdx = GetTensorIdx(tensorMap, "in_input");
        uint32_t weightIdx = GetTensorIdx(tensorMap, "in_weight");
        uint32_t biasIdx = GetTensorIdx(tensorMap, "in_bias");
        outTensorDescs.at(0) = inTensorDescs.at(inputIdx);
        CHECK_TENSORDESC_DIMNUM_VALID(inTensorDescs.at(inputIdx).shape.dimNum);
        auto dimLast = inTensorDescs.at(inputIdx).shape.dimNum - 1;
        if (param.parallelType == COLUMN_PARALLEL) {
            outTensorDescs.at(0).shape.dims[dimLast] = \
                CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[0], param.tensorParallelInfo.worldSize);
        } else {
            int nDim = param.fusionLinearParam.transposeType == TransposeType::TRANSPOSE ? 0 : 1;
            if (param.fusionLinearParam.quantType == LINEAR_W8A8_SC_DEQUANT || \
                param.fusionLinearParam.quantType == LINEAR_W8A8_SC_QUANT) {
                outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(biasIdx).shape.dims[0];
            } else if (param.fusionLinearParam.quantType == W4A16) {
                outTensorDescs.at(0).shape.dims[dimLast] = \
                    CheckIntMulOverFlow(inTensorDescs.at(weightIdx).shape.dims[nDim], 2);  // 2: n维shape * 2
            } else {
                outTensorDescs.at(0).shape.dims[dimLast] = inTensorDescs.at(weightIdx).shape.dims[nDim];
            }
        }
        return atb::NO_ERROR;
    };
}

atb::Status CreateLinearParallel(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    if (param.parallelType == ROW_PARALLEL && !param.biasAfterSync) {
        opGraph.name = "LinearRowParallelNoAdd";
    } else {
        opGraph.name = param.parallelType == COLUMN_PARALLEL ?  "LinearColumnParallel" : "LinearRowParallelAndAdd";
    }

    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum, false);

    atb::Node linearNode;
    atb_speed::common::FusionLinearParam linearParam = param.fusionLinearParam;
    CHECK_OPERATION_STATUS_RETURN(FusionLinear(linearParam, &linearNode.operation));
    std::vector<std::string> linearInTensor = {
        "in_input", "in_weight", "in_scale", "in_offset", "in_descale", "in_bias", "in_compress_idx"
    };
    if (param.fusionLinearParam.supportLora) {
        if (param.fusionLinearParam.useImMask) {
            linearInTensor.push_back("in_im_mask");
        }
        linearInTensor.push_back("in_seq_len_cum_sum");
        linearInTensor.push_back("in_lora_a");
        linearInTensor.push_back("in_lora_b");
    }
    linearNode.inTensorIds = GetTensorIdxList(tensorMap, linearInTensor);
    linearNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_linear_out")};
    opGraph.nodes.push_back(linearNode);

    if (param.tensorParallelInfo.quantType == atb::infer::AllReduceParam::QuantType::QUANT_TYPE_PER_CHANNEL && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
        atb::Node quantNode;
        atb::infer::ElewiseParam quantParam;
        quantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(quantParam, &quantNode.operation));
        quantNode.inTensorIds = GetTensorIdxList(tensorMap, {
            "intermediate_linear_out", "in_reduce_quant_scale", "in_reduce_quant_offset"
        });
        quantNode.outTensorIds = {GetTensorIdx(tensorMap, "intermediate_quant_out")};
        opGraph.nodes.push_back(quantNode);
    }

    CHECK_OPERATION_STATUS_RETURN(AddCommunicationOp(param, opGraph, tensorMap));

    LinearParallelInferShape(opGraph, param, tensorMap);

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

atb::Status CreateLinearParallelLcoc(const LinearParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "LinearParallelLcoc";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum, true);
    ATB_SPEED_LOG_DEBUG("linear parallel lcoc opGraph.inTensorNum " << opGraph.inTensorNum);
    ATB_SPEED_LOG_DEBUG("linear parallel lcoc opGraph.outTensorNum " << opGraph.outTensorNum);
    ATB_SPEED_LOG_DEBUG("linear parallel lcoc opGraph.internalTensorNum " << opGraph.internalTensorNum);

    atb::Node linearParallelNode;
    atb::infer::LinearParallelParam linearParallelParam;
    linearParallelParam.transWeight = param.fusionLinearParam.transposeType == TransposeType::TRANSPOSE;
    linearParallelParam.rank = param.tensorParallelInfo.rank;
    linearParallelParam.rankSize = param.tensorParallelInfo.worldSize;
    linearParallelParam.hasResidual = false;
    linearParallelParam.backend = "lcoc";
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(linearParallelParam, &linearParallelNode.operation));

    linearParallelNode.inTensorIds = GetTensorIdxList(tensorMap, {
        "in_input", "in_weight"
    });
    linearParallelNode.outTensorIds = {
        GetTensorIdx(tensorMap, param.biasAfterSync ? "intermediate_sync_out" : "out")
    };
    opGraph.nodes.push_back(linearParallelNode);

    if (param.biasAfterSync) {
        atb::Node addNode;
        atb::infer::ElewiseParam addParam;
        addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &addNode.operation));
        addNode.inTensorIds = GetTensorIdxList(tensorMap, {
            "intermediate_sync_out", "in_bias"
        });
        addNode.outTensorIds = {GetTensorIdx(tensorMap, "out")};
        opGraph.nodes.push_back(addNode);
    }

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

atb::Status LinearParallel(const LinearParallelParam &param, atb::Operation **operation)
{
    if (param.tensorParallelInfo.worldSize <= 1) {
        return FusionLinear(param.fusionLinearParam, operation);
    } else if (param.parallelType == ROW_PARALLEL) {
        if (param.tensorParallelInfo.backend == "lccl" && \
            param.supportLcoc && !param.fusionLinearParam.supportLora && \
            param.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT) {
            return CreateLinearParallelLcoc(param, operation);
        }
        return CreateLinearParallel(param, operation);
    } else if (param.parallelType == COLUMN_PARALLEL) {
        return CreateLinearParallel(param, operation);
    } else {
        ATB_SPEED_LOG_ERROR("LinearParallel operation doesn't support parallelType: " << param.parallelType
            << " Possible values are 1 (row parallel) or 2 (column parallel).");
        return atb::ERROR_INVALID_PARAM;
    }
}

} // namespace common
} // namespace atb_speed