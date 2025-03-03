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
#ifndef ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H
#define ATB_SPEED_MODELS_QWEN_DECODER_MODEL_H

#include "models/base/model/decoder_model.h"
#include "models/qwen/layer/decoder_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace qwen {

class QwenModelParam : public atb_speed::base::ModelParam {
public:
    void PrintParam() override;

    // withEmbedding为true时，模型包含word embedding层; 反之输入为hidden states; 该选项用于多模态模型适配
    bool withEmbedding = true;

    // 是否使用kv cache int8 量化
    bool enableLogN = false;
    bool isLongSeq = false;
    bool isYarn = false;
    float mscale = 1.0;
    bool enableQScale = false;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class QwenDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit QwenDecoderModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;

    void ConstructInternalTensorMap() override;

    atb::Status AddPositionalEmbedding() override;

    void SetLayerParam(QwenLayerParam &layerParam, uint32_t layerId);

    void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId) override;

    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;

    atb::Status AddOperationToGraph() override;

    QwenModelParam param;

private:
    atb::Status AddMuls();
    atb::Status AddDynamicNTK();
};

REGISTER_MODEL(qwen, QwenDecoderModel);

} // namespace qwen
} // namespace atb_speed
#endif
