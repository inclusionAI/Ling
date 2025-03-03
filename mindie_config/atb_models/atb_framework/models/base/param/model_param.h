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
#ifndef ATB_SPEED_BASE_MODEL_PARAM_H
#define ATB_SPEED_BASE_MODEL_PARAM_H
#include <vector>
#include <nlohmann/json.hpp>
#include "models/base/param/param.h"

namespace atb_speed {
namespace base {

nlohmann::json StringToJson(const std::string &param);

class ModelParam : public Param {
public:
    ModelParam() {};
    ~ModelParam() override {};
    void FromString(const std::string &param);
    void PrintParam() override;
    void CheckParam() override;

    // skipWordEmbedding为true会跳过Word Embedding阶段，直接使用入参中的IN_TENSOR_INPUT_EMBEDDING
    bool skipWordEmbedding = false;
    // isEmbeddingParallel为true时，embedding的权重在hiddenSize维度进行切分; 反之，则不对权重进行切分; 测试表明embedding切分并不会带来性能提升
    bool isEmbeddingParallel = false;
    // isLmHeadParallel为true时，LmHead的权重在vacobSize维度进行切分; 反之，则不对权重进行切分
    bool isLmHeadParallel = true;
    // LmHead Matmul B矩阵是否转置
    int lmHeadTransposeType = -1;
    uint32_t numHiddenLayers = 0;
    int rank = 0;
    int worldSize = 1;
    std::string backend = "hccl";
    std::string rankTableFile = "";
    std::vector<std::vector<int>> packQuantType = {};
    std::vector<std::vector<int>> linearQuantType = {};
    std::vector<std::vector<int>> linearTransposeType = {};
    std::vector<std::vector<bool>> linearHasBias = {};

protected:
    virtual void ParseParam(const nlohmann::json &paramJson);
    virtual void ParseAttentionParam(const nlohmann::json &paramJson);
    virtual void ParseMatmulParam(const nlohmann::json &paramJson);
    virtual void ParseTensorParallelParam(const nlohmann::json &paramJson);
};

template <typename T>
T VerifyParam(const nlohmann::json& paramJson, const std::string& key, bool isVector = false);
} // namespace base
} // namespace atb_speed
#endif