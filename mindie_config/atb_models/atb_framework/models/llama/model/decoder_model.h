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
#ifndef ATB_SPEED_MODELS_LLAMA_DECODER_MODEL_H
#define ATB_SPEED_MODELS_LLAMA_DECODER_MODEL_H

#include "atb_speed/base/model.h"
#include "models/base/param/model_param.h"
#include "atb_speed/utils/model_factory.h"
#include "models/llama/layer/decoder_layer.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace llama {

class LlamaModelParam : public atb_speed::base::ModelParam {
public:
    LlamaModelParam() {};
    ~LlamaModelParam() override {};

    void PrintParam() override;

    // 是否需要在QKV切分之前进行reshape
    bool splitWithStride = false;
    // 输入是否为长序列
    bool isLongSeq = false;

protected:
    void ParseParam(const nlohmann::json &paramJson) override;
};

class LlamaDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit LlamaDecoderModel(const std::string &param);

protected:
    void ConstructInTensorMap() override;
    void ConstructInternalTensorMap() override;
    atb::Status ParseParam(const std::string &paramString) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    void BuildNodeOutTensors(int nodeId, atb_speed::Model::Node &node, atb::SVector<atb::TensorDesc>& inTensorDescs);
    void BuildNodeVariantPack(int nodeId) override;
    void SetLayerParam(LlamaLayerParam &layerParam, uint32_t layerId);
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    atb::Status AddOperationToGraph() override;
    atb::Status AddPositionalEmbedding()  override;

    LlamaModelParam param;
    std::vector<int> blockNumsList_;

private:
    atb::Status AddDynamicNTK();
};

REGISTER_MODEL(llama, LlamaDecoderModel);

}  // namespace llama
}  // namespace atb_speed
#endif
