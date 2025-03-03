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
#include <atb/atb_infer.h>
#include "operations/fusion/utils.h"
#include "operations/fusion/mlp/mlp.h"

namespace atb_speed {
namespace common {

std::map<std::string, atb::SVector<std::string>> GetMlpInTensorCandidates()
{
    std::map<std::string, atb::SVector<std::string>> mlpInTensorCandidates = {
        {"default", {
            "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
            "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
            "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
            "in_weight_down", "in_scale_down", "in_offset_down", "in_descale_down", "in_bias_down",
            "in_compress_idx_down"}
        },
        {"add_norm", {"in_residual_add"}},
        {"lora", {
            "in_seq_len_cum_sum", "in_lora_a_0", "in_lora_b_0", "in_lora_a_1", "in_lora_b_1",
            "in_down_lora_a", "in_down_lora_b"}
        },
        {"reduce_quant", {
            "in_reduce_quant_scale", "in_reduce_quant_offset",
            "in_gather_quant_scale", "in_gather_quant_offset"}},
        {"lora_with_mask", {"in_im_mask"}}
    };
    return mlpInTensorCandidates;
}

std::map<std::string, atb::SVector<std::string>> GetMlpOutTensorCandidates()
{
    std::map<std::string, atb::SVector<std::string>> mlpOutTensorCandidates = {
        {"default", {"out_linear"}},
        {"add_norm", {"out_add"}},
    };
    return mlpOutTensorCandidates;
}

template <typename NormParamType>
atb::SVector<std::string> ConstructMlpInTensorList(const MlpParam<NormParamType> &param)
{
    auto mlpInTensorCandidates = GetMlpInTensorCandidates();

    atb::SVector<std::string> inTensorList = {};

    // 添加默认的Tensor
    AddTensorToList(mlpInTensorCandidates, "default", inTensorList);

    // 添加add norm特性的Tensor
    if (param.enableAddNorm) {
        AddTensorToList(mlpInTensorCandidates, "add_norm", inTensorList);
    }

    // 添加lora特性的Tensor
    if (param.supportLora) {
        if (param.useImMask) {
            AddTensorToList(mlpInTensorCandidates, "lora_with_mask", inTensorList);
        }
        AddTensorToList(mlpInTensorCandidates, "lora", inTensorList);
    }

    // 添加lccl reduce int8特性的Tensor
    if (param.downLinearTensorParallelInfo.quantType != \
        atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        AddTensorToList(mlpInTensorCandidates, "reduce_quant", inTensorList);
    }

    return inTensorList;
}

template <typename NormParamType>
atb::SVector<std::string> ConstructMlpOutTensorList(const MlpParam<NormParamType> &param)
{
    auto mlpOutTensorCandidates = GetMlpOutTensorCandidates();

    atb::SVector<std::string> outTensorList = {};

    // 添加outTensor
    AddTensorToList(mlpOutTensorCandidates, "default", outTensorList);
    if (param.enableAddNorm) {
        AddTensorToList(mlpOutTensorCandidates, "add_norm", outTensorList);
    }

    return outTensorList;
}

template <typename NormParamType>
atb::Status AddMlpNormLinearGateUp(const MlpParam<NormParamType> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* normLinearGateUpOp = nullptr;
    atb_speed::common::NormLinearParam<NormParamType> gateUpNormLinearParam;
    gateUpNormLinearParam.isAntiOutlier = isAntiOutlier;
    gateUpNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[GATE_LINEAR_INDEX], param.enableNormQuantOp);
    gateUpNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    gateUpNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
    gateUpNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    gateUpNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    gateUpNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    gateUpNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[GATE_LINEAR_INDEX];
    gateUpNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    gateUpNormLinearParam.skipNorm = param.skipNorm;
    gateUpNormLinearParam.normHasBias = param.normHasBias;
    gateUpNormLinearParam.enableAddNorm = param.enableAddNorm;
    gateUpNormLinearParam.normParamType = param.normParamType;
    gateUpNormLinearParam.normQuantParamType = param.normQuantParamType;
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(gateUpNormLinearParam, &normLinearGateUpOp));

    atb::SVector<std::string> gateUpInTensorNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
        "in_weight_0", "in_scale_0", "in_offset_0", "in_descale_0", "in_bias_0", "in_compress_idx_0",
    };
    if (param.enableAddNorm) {
        gateUpInTensorNames.push_back("in_residual_add");
    }
    if (param.supportLora) {
        if (param.useImMask) {
            gateUpInTensorNames.push_back("in_im_mask");
        }
        gateUpInTensorNames.push_back("in_seq_len_cum_sum");
        gateUpInTensorNames.push_back("in_lora_a_0");
        gateUpInTensorNames.push_back("in_lora_b_0");
    }

    atb::SVector<std::string> gateUpOutTensorNames = {};
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        gateUpOutTensorNames = {"intermediate_gate_up"} ;
    } else if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        gateUpOutTensorNames = {"intermediate_gate"};
    } else {
        gateUpOutTensorNames = {"intermediate_up"};
    }
    if (param.enableAddNorm) {
        gateUpOutTensorNames.push_back("out_add");
    }

    graphBuilder->AddOperation(normLinearGateUpOp, gateUpInTensorNames, gateUpOutTensorNames);
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpNormLinearUp(const MlpParam<NormParamType> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* normLinearUpOp = nullptr;
    atb_speed::common::NormLinearParam<NormParamType> upNormLinearParam;
    upNormLinearParam.isAntiOutlier = isAntiOutlier;
    upNormLinearParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.packQuantType, param.layerLinearQuantType[UP_LINEAR_INDEX], param.enableNormQuantOp);
    upNormLinearParam.fusionLinearParam.isBF16 = param.isBF16;
    upNormLinearParam.fusionLinearParam.hasBias = param.gateUpHasBias;
    upNormLinearParam.fusionLinearParam.supportLora = param.supportLora;
    upNormLinearParam.fusionLinearParam.useImMask = param.useImMask;
    upNormLinearParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    upNormLinearParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[UP_LINEAR_INDEX];
    upNormLinearParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    upNormLinearParam.skipNorm = param.skipNorm;
    upNormLinearParam.normHasBias = param.normHasBias;
    upNormLinearParam.normParamType = param.normParamType;
    upNormLinearParam.normQuantParamType = param.normQuantParamType;
    CHECK_OPERATION_STATUS_RETURN(NormLinear<NormParamType>(upNormLinearParam, &normLinearUpOp));

    atb::SVector<std::string> upInTensorNames = {
        "in_input", "in_norm_weight", "in_norm_bias", "in_norm_new_weight", "in_norm_new_bias",
        "in_weight_1", "in_scale_1", "in_offset_1", "in_descale_1", "in_bias_1", "in_compress_idx_1",
    };
    if (param.supportLora) {
        if (param.useImMask) {
            upInTensorNames.push_back("in_im_mask");
        }
        upInTensorNames.push_back("in_seq_len_cum_sum");
        upInTensorNames.push_back("in_lora_a_1");
        upInTensorNames.push_back("in_lora_b_1");
    }

    graphBuilder->AddOperation(normLinearUpOp, upInTensorNames, {"intermediate_up"});
    return atb::NO_ERROR;
}

atb::Status AddMlpSplit(atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* splitOp = nullptr;
    atb::infer::SplitParam splitParam;
    splitParam.splitDim = -1; // [batchSize, seqLen, 2 * hiddenSize]
    splitParam.splitNum = 2;  // 进行二等分
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(splitParam, &splitOp));

    graphBuilder->AddOperation(splitOp, {"intermediate_gate_up"}, {"intermediate_gate", "intermediate_up"});
    return atb::NO_ERROR;
}

atb::Status AddMlpSwiGLUConcat(atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* concatOp = nullptr;
    atb::infer::ConcatParam concatParam;
    concatParam.concatDim = -1;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(concatParam, &concatOp));

    graphBuilder->AddOperation(concatOp, {"intermediate_gate", "intermediate_up"}, {"intermediate_gate_up"});
    return atb::NO_ERROR;
}

atb::Status AddMlpMul(atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* mulOp = nullptr;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(elewiseParam, &mulOp));

    graphBuilder->AddOperation(
        mulOp, {"intermediate_activation_out", "intermediate_up"}, {"intermediate_activation_out"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpActivation(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpSplit(graphBuilder));
    }

    atb::Operation* activationOp = nullptr;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.activationParam, &activationOp));
    graphBuilder->AddOperation(
        activationOp,
        {param.mlpPackType == MlpPackType::UP_WEIGHT_ONLY ? "intermediate_up" : "intermediate_gate"},
        {"intermediate_activation_out"});

    if (param.mlpPackType != MlpPackType::UP_WEIGHT_ONLY) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpMul(graphBuilder));
    }
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpLiteActivation(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpSplit(graphBuilder));
    }

    atb::Operation* sigmoidOp = nullptr;
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SIGMOID;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(activationParam, &sigmoidOp));
    graphBuilder->AddOperation(sigmoidOp, {"intermediate_gate"}, {"intermediate_activation_out"});

    atb::Operation* sigmoidMulOp = nullptr;
    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(elewiseParam, &sigmoidMulOp));
    graphBuilder->AddOperation(
        sigmoidMulOp, {"intermediate_gate", "intermediate_activation_out"}, {"intermediate_activation_out"});

    if (param.mlpPackType != MlpPackType::UP_WEIGHT_ONLY) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpMul(graphBuilder));
    }

    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpSwiGLUActivation(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpSwiGLUConcat(graphBuilder));
    }

    atb::Operation* activationOp = nullptr;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param.activationParam, &activationOp));
    graphBuilder->AddOperation(activationOp, {"intermediate_gate_up"}, {"intermediate_activation_out"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status AddMlpLinearDown(const MlpParam<NormParamType> &param, atb::GraphOpBuilder* &graphBuilder)
{
    atb::Operation* linearDownOp = nullptr;
    atb_speed::common::LinearParallelParam downLinearParallelParam;
    downLinearParallelParam.parallelType = atb_speed::common::ROW_PARALLEL;
    downLinearParallelParam.fusionLinearParam.quantType = GetLinearQuantType(
        param.downQuantType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED \
            ? param.packQuantType : param.downQuantType,
        param.layerLinearQuantType[DOWN_LINEAR_INDEX], false);
    downLinearParallelParam.biasAfterSync = param.downLinearTensorParallelInfo.worldSize > 1 && \
        downLinearParallelParam.fusionLinearParam.quantType == atb_speed::common::LinearQuantType::NO_QUANT && \
        param.downHasBias;
    downLinearParallelParam.fusionLinearParam.hasBias = param.downHasBias && !downLinearParallelParam.biasAfterSync;
    downLinearParallelParam.fusionLinearParam.isBF16 = param.isBF16;
    downLinearParallelParam.fusionLinearParam.supportLora = param.supportLora;
    downLinearParallelParam.fusionLinearParam.useImMask = param.useImMask;
    downLinearParallelParam.fusionLinearParam.loraEnableGMM = param.loraEnableGMM;
    downLinearParallelParam.fusionLinearParam.transposeType = param.layerLinearTransposeType[DOWN_LINEAR_INDEX];
    downLinearParallelParam.fusionLinearParam.quantGroupSize = param.quantGroupSize;
    downLinearParallelParam.tensorParallelInfo = param.downLinearTensorParallelInfo;
    downLinearParallelParam.supportLcoc = param.supportLcoc;
    CHECK_OPERATION_STATUS_RETURN(LinearParallel(downLinearParallelParam, &linearDownOp));

    atb::SVector<std::string> downInTensorNames = {
        "intermediate_activation_out",
        "in_weight_down", "in_scale_down", "in_offset_down", "in_descale_down", "in_bias_down",
        "in_compress_idx_down"
    };
    if (param.supportLora) {
        if (param.useImMask) {
            downInTensorNames.push_back("in_im_mask");
        }
        downInTensorNames.push_back("in_seq_len_cum_sum");
        downInTensorNames.push_back("in_down_lora_a");
        downInTensorNames.push_back("in_down_lora_b");
    }
    if (param.downLinearTensorParallelInfo.quantType != atb::infer::AllReduceParam::QuantType::QUANT_TYPE_UNDEFINED) {
        downInTensorNames.push_back("in_reduce_quant_scale");
        downInTensorNames.push_back("in_reduce_quant_offset");
        downInTensorNames.push_back("in_gather_quant_scale");
        downInTensorNames.push_back("in_gather_quant_offset");
    }

    graphBuilder->AddOperation(linearDownOp, downInTensorNames, {"out_linear"});
    return atb::NO_ERROR;
}

template <typename NormParamType>
atb::Status Mlp(const MlpParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphOpBuilder* graphOpBuilder = nullptr;
    CHECK_OPERATION_STATUS_RETURN(CreateGraphOpBuilder(&graphOpBuilder));
    atb::Status res = CreateMlp(param, graphOpBuilder, operation, false);
    if (DestroyGraphOpBuilder(graphOpBuilder) != atb::NO_ERROR) {
        ATB_SPEED_LOG_WARN("Destroy graph builder failed. This may leads to memory leak, please check");
    }
    return res;
}

template <typename NormParamType>
atb::Status MlpSwiGLU(const MlpParam<NormParamType> &param, atb::Operation **operation)
{
    atb::GraphOpBuilder* graphOpBuilder = nullptr;
    CHECK_OPERATION_STATUS_RETURN(CreateGraphOpBuilder(&graphOpBuilder));
    atb::Status res = CreateMlp(param, graphOpBuilder, operation, true);
    if (DestroyGraphOpBuilder(graphOpBuilder) != atb::NO_ERROR) {
        ATB_SPEED_LOG_WARN("Destroy graph builder failed. This may leads to memory leak, please check");
    }
    return res;
}

template <typename NormParamType>
atb::Status CreateMlp(
    const MlpParam<NormParamType> &param,
    atb::GraphOpBuilder* &graphOpBuilder,
    atb::Operation **operation, bool isSwiGLU)
{
    bool isAntiOutlier = CheckAntiOutlier(param.packQuantType);
    if (CheckParamVectorSize(param.layerLinearQuantType, DOWN_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearQuantType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }
    if (CheckParamVectorSize(param.layerLinearTransposeType, DOWN_LINEAR_INDEX + 1) != atb::NO_ERROR) {
        ATB_SPEED_LOG_ERROR("The size of param.layerLinearTransposeType is wrong, please check");
        return atb::ERROR_INVALID_PARAM;
    }

    std::string graphName = isSwiGLU ? "MlpSwiGLU" : "Mlp";
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_PACK) {
        graphName += "GateUpWeightPack";
    } else if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        graphName += "GateUpWeightNoPack";
    } else {
        graphName += "UpWeightOnly";
    }

    atb::InferShapeFunc inferShapeFunc = [param](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.enableAddNorm) {
            outTensorDescs.at(1) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(graphOpBuilder->Init(
        graphName,
        inferShapeFunc,
        ConstructMlpInTensorList(param),
        ConstructMlpOutTensorList(param)
    ));

    // Gate Up
    CHECK_OPERATION_STATUS_RETURN(AddMlpNormLinearGateUp(param, isAntiOutlier, graphOpBuilder));
    if (param.mlpPackType == MlpPackType::GATE_UP_WEIGHT_NO_PACK) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpNormLinearUp(param, isAntiOutlier, graphOpBuilder));
    }
    // Activation
    if (param.isLite) {
        CHECK_OPERATION_STATUS_RETURN(AddMlpLiteActivation(param, graphOpBuilder));
    } else {
        if (isSwiGLU) {
            CHECK_OPERATION_STATUS_RETURN(AddMlpSwiGLUActivation(param, graphOpBuilder));
        } else {
            CHECK_OPERATION_STATUS_RETURN(AddMlpActivation(param, graphOpBuilder));
        }
    }
    // Down
    CHECK_OPERATION_STATUS_RETURN(AddMlpLinearDown(param, graphOpBuilder));

    *operation = graphOpBuilder->Build();
    return atb::NO_ERROR;
}

MlpPackType GetMlpPackType(const int &packQuantType, bool upWeightOnly)
{
    if (upWeightOnly) {
        return atb_speed::common::UP_WEIGHT_ONLY;
    }
    if (CheckPack(packQuantType)) {
        return atb_speed::common::GATE_UP_WEIGHT_PACK;
    } else {
        return atb_speed::common::GATE_UP_WEIGHT_NO_PACK;
    }
}

template atb::SVector<std::string> ConstructMlpInTensorList(const MlpParam<atb::infer::RmsNormParam> &param);

template atb::SVector<std::string> ConstructMlpInTensorList(const MlpParam<atb::infer::LayerNormParam> &param);

template atb::SVector<std::string> ConstructMlpOutTensorList(const MlpParam<atb::infer::RmsNormParam> &param);

template atb::SVector<std::string> ConstructMlpOutTensorList(const MlpParam<atb::infer::LayerNormParam> &param);

template atb::Status AddMlpNormLinearGateUp(const MlpParam<atb::infer::RmsNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpNormLinearGateUp(const MlpParam<atb::infer::LayerNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpLinearDown(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpLinearDown(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpNormLinearUp(const MlpParam<atb::infer::RmsNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpNormLinearUp(const MlpParam<atb::infer::LayerNormParam> &param,
    bool isAntiOutlier, atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpActivation(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpActivation(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpLiteActivation(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpSwiGLUActivation(const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status AddMlpSwiGLUActivation(const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphBuilder);

template atb::Status Mlp(const MlpParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template atb::Status Mlp(const MlpParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

template atb::Status MlpSwiGLU(const MlpParam<atb::infer::RmsNormParam> &param, atb::Operation **operation);

template atb::Status MlpSwiGLU(const MlpParam<atb::infer::LayerNormParam> &param, atb::Operation **operation);

template atb::Status CreateMlp(
    const MlpParam<atb::infer::RmsNormParam> &param,
    atb::GraphOpBuilder* &graphOpBuilder, atb::Operation **operation, bool isSwiGLU);

template atb::Status CreateMlp(
    const MlpParam<atb::infer::LayerNormParam> &param,
    atb::GraphOpBuilder* &graphOpBuilder, atb::Operation **operation, bool isSwiGLU);

} // namespace common
} // namespace atb_speed