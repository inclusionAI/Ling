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
#include "vector"
#include "nlohmann/json.hpp"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "models/minicpm/model/decoder_model.h"

namespace atb_speed {
namespace minicpm {

MiniCPMDecoderModel::MiniCPMDecoderModel(const std::string &param):atb_speed::base::DecoderModel(param)
{
    this->param.FromString(param);
    this->inTensorCandidates["compress_head"] = {"wins_global", "in_ra_seqlens"};
}

void MiniCPMModelParam::ParseParam(const nlohmann::json &paramJson)
{
    atb_speed::base::ModelParam::ParseParam(paramJson);
    if (paramJson.contains("hiddenSize")) {
        this->hiddenSize = paramJson["hiddenSize"].get<uint32_t>();
    }
    if (paramJson.contains("scale_emb")) {
        this->scaleEmb = paramJson["scale_emb"].get<float>();
    }
    if (paramJson.contains("scale_depth")) {
        this->scaleDepth = paramJson["scale_depth"].get<float>();
    }
    if (paramJson.contains("dim_model_base")) {
        this->dimModelBase = paramJson["dim_model_base"].get<float>();
    }
}

void MiniCPMModelParam::PrintParam()
{
    atb_speed::base::ModelParam::PrintParam();
    ATB_SPEED_LOG_DEBUG("MiniCPMModelParam: hiddenSize: " << this->hiddenSize
                  << ", scaleEmb: " << this->scaleEmb
                  << ", scaleDepth:" << this->scaleDepth
                  << ", dimModelBase:" << this->dimModelBase);
}

MiniCPMDecoderModel::~MiniCPMDecoderModel() {}

atb::Status MiniCPMDecoderModel::AddOperationToGraph()
{
    if (!this->param.skipWordEmbedding) { CHECK_OPERATION_STATUS_RETURN(this->AddWordEmbedding()); }
    CHECK_OPERATION_STATUS_RETURN(AddMuls());
    CHECK_OPERATION_STATUS_RETURN(AddPositionalEmbedding());
    CHECK_OPERATION_STATUS_RETURN(AddLayer());
    CHECK_OPERATION_STATUS_RETURN(AddFinalNorm());
    CHECK_OPERATION_STATUS_RETURN(AddLmMuls());
    CHECK_OPERATION_STATUS_RETURN(AddLmhead());
    return atb::NO_ERROR;
}

int64_t MiniCPMDecoderModel::AddMuls()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node mulsNode;
    float embedingScale = param.scaleEmb;
    ATB_SPEED_LOG_DEBUG("embeding scale = " << embedingScale);
    atb::infer::ElewiseParam embedingScaleMulsParam;
    embedingScaleMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    embedingScaleMulsParam.mulsParam.varAttr = embedingScale;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(embedingScaleMulsParam, &op));
    mulsNode.operation.reset(op);
    mulsNode.inTensors = {
        this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"))};
    mulsNode.outTensors = mulsNode.inTensors;
    graph_.nodes.push_back(mulsNode);

    return atb::NO_ERROR;
}

atb::Status MiniCPMDecoderModel::CreateLayerOperation(atb::Operation **op, uint32_t layerId)
{
    MiniCPMLayerParam layerParam;
    this->SetLayerParam(layerParam, layerId);
    MiniCPMDecoderLayer decoderLayer(layerParam);
    CHECK_OPERATION_STATUS_RETURN(decoderLayer.BuildGraph(op));

    return atb::NO_ERROR;
}

void MiniCPMDecoderModel::SetLayerParam(MiniCPMLayerParam &layerParam, uint32_t layerId)
{
    atb_speed::base::DecoderModel::SetLayerParam(layerParam, layerId);
    layerParam.numHiddenLayers = this->param.numHiddenLayers;
    layerParam.scaleDepth = this->param.scaleDepth;
}

int64_t MiniCPMDecoderModel::AddLmMuls()
{
    atb::Operation *op = nullptr;
    atb_speed::Model::Node lmMulsNode;
    float lmScale = 1.0 / (param.hiddenSize / param.dimModelBase) ;
    ATB_SPEED_LOG_DEBUG("lm head scale = " << lmScale);
    atb::infer::ElewiseParam lmMulsParam;
    lmMulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    lmMulsParam.mulsParam.varAttr = lmScale;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(lmMulsParam, &op));
    lmMulsNode.operation.reset(op);
    lmMulsNode.inTensors = {
        this->param.skipWordEmbedding ? \
            &graph_.inTensors.at(atb_speed::common::GetTensorIdx(this->inTensorMap, "input_embedding")) : \
            &graph_.internalTensors.at(atb_speed::common::GetTensorIdx(this->internalTensorMap, "hidden_states"))};
    lmMulsNode.outTensors = lmMulsNode.inTensors;
    graph_.nodes.push_back(lmMulsNode);
    return atb::NO_ERROR;
}
} // namespace minicpm
} // namespace atb_speed