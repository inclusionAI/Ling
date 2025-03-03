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
#ifndef ATB_SPEED_MODELS_MOE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_MOE_DECODER_MODEL_H

#include <vector>
#include <models/base/model/decoder_model.h>
#include "atb_speed/base/model.h"
#include "models/base/param/model_param.h"
#include "models/moe/layer/decoder_layer.h"

namespace atb_speed {
namespace moe {
class MoeModelParam : public atb_speed::base::ModelParam {
public:
    void PrintParam() override;
    void CheckParam() override;

    bool normTopkProb = false;
    bool normHasBias = false;
    bool hasSharedExpert = true;
    bool hasSharedExpertGate = false;
    // moe相关成员变量
    int numOfExperts = 8;
    int expertParallelDegree = 1;
    int firstKDenseReplace = 1;
    int numOfSharedExperts = 2;
    int maskStartIdx = 0;
    bool enableFusedRouting = false;
    std::string routingMethod = "softMaxTopK";
    std::string processLogits = "normalization";
    std::vector<std::vector<int>> moeLinearQuantType = {};
    std::vector<std::vector<int>> mlpLinearQuantType = {};
    std::vector<std::vector<int>> moeLinearTransposeType = {};
    std::vector<std::vector<int>> mlpLinearTransposeType = {};
    atb::SVector<int32_t> numOfSelectedExperts = {};

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
    virtual void CheckRoutingMethodValid();
    virtual void CheckProcessLogitsValid();
};

class MoeDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit MoeDecoderModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;
    void SetLayerParam(MoeLayerParam &layerParam, uint32_t layerId);
    void SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId) override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;

    MoeModelParam param;
};

}  // namespace moe
}  // namespace atb_speed
#endif