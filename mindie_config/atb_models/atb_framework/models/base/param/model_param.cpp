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

#include "models/base/param/model_param.h"

namespace atb_speed {
namespace base {

nlohmann::json StringToJson(const std::string &param)
{
    nlohmann::json paramJson;
    try {
        paramJson = nlohmann::json::parse(param);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "parse param fail, please check param's format, error: " << e.what() << std::endl;
        throw std::runtime_error(ss.str());
    }
    return paramJson;
}

template <typename T>
T VerifyParam(const nlohmann::json& paramJson, const std::string& key, bool isVector)
{
    try {
        if (isVector) {
            return paramJson.get<T>();
        } else {
            return paramJson.at(key).get<T>();
        }
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "Failed to parse parameter " << key << ": " << e.what() << ". Please check the type of param.";
        ATB_SPEED_LOG_ERROR(ss.str(), ATB_MODELS_MODEL_PARAM_JSON_INVALID);
        throw std::runtime_error(ss.str());
    }
}

void ModelParam::FromString(const std::string &param)
{
    nlohmann::json paramJson = StringToJson(param);
    ParseParam(paramJson);
    CheckParam();
    PrintParam();
}

void ModelParam::PrintParam()
{
    Param::PrintParam();
    ATB_SPEED_LOG_DEBUG("Model Param: isEmbeddingParallel: " << this->isEmbeddingParallel
                  << ", isLmHeadParallel: " << this->isLmHeadParallel
                  << ", lmHeadTransposeType: " << this->lmHeadTransposeType
                  << ", numHiddenLayers: " << this->numHiddenLayers
                  << ", rank: " << this->rank
                  << ", worldSize: " << this->worldSize
                  << ", backend: " << this->backend
                  << ", rankTableFile: " << this->rankTableFile);
}

void ModelParam::ParseParam(const nlohmann::json &paramJson)
{
    this->isUnpadInputs = VerifyParam<bool>(paramJson, "isUnpadInputs");
    this->isPrefill = VerifyParam<bool>(paramJson, "isPrefill");
    this->isBF16 = VerifyParam<bool>(paramJson, "isBF16");
    this->normEps = VerifyParam<float>(paramJson, "normEps");
    this->normType = VerifyParam<NormType>(paramJson, "normType");
    if (paramJson.contains("isLite")) { this->isLite = VerifyParam<bool>(paramJson, "isLite"); }
    if (paramJson.contains("enableSwiGLU")) {
        this->enableSwiGLU = VerifyParam<bool>(paramJson, "enableSwiGLU");
    }
    this->numHiddenLayers = CheckNumHiddenLayersValid(VerifyParam<uint32_t>(paramJson, "numHiddenLayers"));
    if (paramJson.contains("enableAddNorm")) {
        this->enableAddNorm = VerifyParam<bool>(paramJson, "enableAddNorm");
    }
    if (paramJson.contains("skipWordEmbedding")) {
        this->skipWordEmbedding = VerifyParam<bool>(paramJson, "skipWordEmbedding");
    }
    if (paramJson.contains("positionEmbeddingType")) {
        this->positionEmbeddingType = VerifyParam<PositionEmbeddingType>(paramJson, "positionEmbeddingType");
    }
    if (paramJson.contains("enablePrefixCache")) {
        this->enablePrefixCache = VerifyParam<bool>(paramJson, "enablePrefixCache");
    }
    if (paramJson.contains("weightQuantType")) {
        this->weightQuantType = VerifyParam<std::string>(paramJson, "weightQuantType");
    }

    ParseAttentionParam(paramJson);
    ParseMatmulParam(paramJson);
    ParseTensorParallelParam(paramJson);
}

void ModelParam::ParseAttentionParam(const nlohmann::json &paramJson)
{
    this->isFA = VerifyParam<bool>(paramJson, "isFA");
    this->numAttentionHeadsPerRank = VerifyParam<uint32_t>(paramJson, "numAttentionHeadsPerRank");
    this->hiddenSizePerAttentionHead = VerifyParam<uint32_t>(paramJson, "hiddenSizePerAttentionHead");
    this->numKeyValueHeadsPerRank = VerifyParam<uint32_t>(paramJson, "numKeyValueHeadsPerRank");
    if (paramJson.contains("enableKvQuant")) {
        this->enableKvQuant = VerifyParam<bool>(paramJson, "enableKvQuant");
    }
    if (paramJson.contains("enableFA3")) { this->enableFA3 = VerifyParam<bool>(paramJson, "enableFA3"); }
    if (paramJson.contains("attnBackend")) {
        this->attnBackend = VerifyParam<atb_speed::common::OpBackend>(paramJson, "attnBackend");
    }
    if (paramJson.contains("enableSpeculate")) {
        this->enableSpeculate = VerifyParam<bool>(paramJson, "enableSpeculate");
    }
    if (paramJson.contains("enableSplitFuse")) {
        this->enableSplitFuse = VerifyParam<bool>(paramJson, "enableSplitFuse");
    }
    if (paramJson.contains("enableCompressHead")) {
        this->enableCompressHead = VerifyParam<bool>(paramJson, "enableCompressHead");
    }
}

void ModelParam::ParseMatmulParam(const nlohmann::json &paramJson)
{
    this->lmHeadTransposeType = VerifyParam<int>(paramJson, "lmHeadTransposeType");
    for (auto item : paramJson["packQuantType"]) {
        this->packQuantType.push_back(VerifyParam<std::vector<int>>(item, "packQuantType", true));
    }
    if (paramJson.contains("linearQuantType")) {
        for (auto item : paramJson["linearQuantType"]) {
            this->linearQuantType.push_back(VerifyParam<std::vector<int>>(item, "linearQuantType", true));
        }
        CheckLinearPackParamsSufficient(this->linearQuantType, this->numHiddenLayers);
    }
    if (paramJson.contains("linearTransposeType")) {
        for (auto item : paramJson["linearTransposeType"]) {
            this->linearTransposeType.push_back(VerifyParam<std::vector<int>>(item, "linearTransposeType", true));
        }
        CheckLinearPackParamsSufficient(this->linearTransposeType, this->numHiddenLayers);
    }
    if (paramJson.contains("linearHasBias")) {
        for (auto item : paramJson["linearHasBias"]) {
            this->linearHasBias.push_back(VerifyParam<std::vector<bool>>(item, "linearHasBias", true));
        }
        CheckLinearHasBiasSufficient(this->linearHasBias, this->numHiddenLayers);
    }
    if (paramJson.contains("enableLcoc")) {
        this->enableLcoc = VerifyParam<bool>(paramJson, "enableLcoc");
    }
    if (paramJson.contains("enableReduceQuant")) {
        this->enableReduceQuant = VerifyParam<bool>(paramJson, "enableReduceQuant");
    }
    if (paramJson.contains("enableLora")) {
        this->enableLora = VerifyParam<bool>(paramJson, "enableLora");
    }
    if (paramJson.contains("loraEnableGMM")) {
        this->loraEnableGMM = VerifyParam<bool>(paramJson, "loraEnableGMM");
    }
    if (paramJson.contains("quantGroupSize")) {
        this->quantGroupSize = VerifyParam<uint32_t>(paramJson, "quantGroupSize");
    }
}

void ModelParam::ParseTensorParallelParam(const nlohmann::json &paramJson)
{
    if (paramJson.contains("isEmbeddingParallel")) {
        this->isEmbeddingParallel = VerifyParam<bool>(paramJson, "isEmbeddingParallel");
    }
    if (paramJson.contains("isLmHeadParallel")) {
        this->isLmHeadParallel = VerifyParam<bool>(paramJson, "isLmHeadParallel");
    }
    this->backend = VerifyParam<std::string>(paramJson, "backend");
    this->rank = VerifyParam<int>(paramJson, "rank");
    this->worldSize = VerifyParam<int>(paramJson, "worldSize");
    this->worldSize = CheckPositive(this->worldSize);
    if (paramJson.contains("rankTableFile")) {
        this->rankTableFile = VerifyParam<std::string>(paramJson, "rankTableFile");
    }
}

void ModelParam::CheckParam()
{
    if (this->rank >= this->worldSize) {
        throw std::runtime_error("worldSize must be greater than rank, please check.");
    }
    if (this->positionEmbeddingType != ROPE && this->positionEmbeddingType != ALIBI && \
        this->positionEmbeddingType != ABSOLUTE) {
        throw std::runtime_error("positionEmbeddingType is an enumeration variable with possible values: ROPE = 0, "
            "ALIBI = 1 or ABSOLUTE = 2, please check.");
    }
    if (this->normType != RMS_NORM && this->normType != LAYER_NORM) {
        throw std::runtime_error("normType is an enumeration variable with possible values: RMS_NORM = 0 or "
            "LAYER_NORM = 1, please check.");
    }
    if (this->attnBackend != atb_speed::common::ATB && this->attnBackend != atb_speed::common::ACLNN) {
        throw std::runtime_error("attnBackend is an enumeration variable with possible values: ACLNN = 0 or "
        "ATB = 1, please check.");
    }
    if (this->lmHeadTransposeType != atb_speed::common::TRANSPOSE_INVALID && this->lmHeadTransposeType != \
        atb_speed::common::NOT_TRANSPOSE && this->lmHeadTransposeType != atb_speed::common::TRANSPOSE) {
        throw std::runtime_error("lmHeadTransposeType is an enumeration variable with possible values: "
        "TRANSPOSE_INVALID = -1, NOT_TRANSPOSE = 0 or TRANSPOSE = 1, please check.");
    }
    auto packType = atb_speed::common::ConvertQuantTypeToPackType(this->weightQuantType);
    if (packType == atb_speed::common::PackQuantType::PACK_QUANT_UNDEFINED && !this->weightQuantType.empty()) {
        throw std::runtime_error(
            "weightQuantType should be float, w8a8, w8a8s, w8a8sc, w8a8_dynamic, w8a16, w4a16 or an empty string.");
    }
}
} // namespace base
} // namespace atb_speed