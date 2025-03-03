/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#ifndef ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "models/qwen/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"
#include "models/base/param/model_param.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace qwen {
class GteDecoderModelParam : public atb_speed::base::ModelParam {
public:
    bool isEmbedding = false;
    bool withEmbedding = true;
    bool enableLogN = false;

    uint32_t quantGroupSize = 64;
    void PrintParam() override;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};
    

class GteDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit GteDecoderModel(const std::string &param);
    ~GteDecoderModel() override;
protected:
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;
    atb::Status AddFinalNorm() override;
    int64_t BuildGraph() override;
    void SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId);
    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    atb::Status AddOperationToGraph() override;
    GteDecoderModelParam param;
};

REGISTER_MODEL(qwen, GteDecoderModel);

} // namespace base
} // namespace atb_speed
#endif