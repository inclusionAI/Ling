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
#ifndef ATB_SPEED_MODELS_BASE_DECODER_MODEL_H
#define ATB_SPEED_MODELS_BASE_DECODER_MODEL_H

#include <vector>
#include "atb_speed/base/model.h"
#include "models/base/param/model_param.h"
#include "models/base/layer/decoder_layer.h"
#include "operations/fusion/embedding/word_embedding.h"
#include "operations/fusion/embedding/positional_embedding.h"
#include "operations/fusion/lmhead/lmhead.h"
#include "operations/fusion/utils.h"
#include "atb_speed/utils/tensor_util.h"

namespace atb_speed {
namespace base {

class DecoderModel : public Model {
public:
    explicit DecoderModel(const std::string &param);
    ~DecoderModel() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

protected:
    virtual void ConstructInTensorMap();
    virtual void ConstructInternalTensorMap();
    atb::Status ParseParam(const std::string &paramString) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    virtual void SetWordEmbeddingParam(atb_speed::common::WordEmbeddingParam &wordEmbeddingParam);
    virtual void SetLayerParam(LayerParam &layerParam, uint32_t layerId);
    virtual void SetFinalNormParam(atb::infer::RmsNormParam &normParam);
    virtual void SetFinalNormParam(atb::infer::LayerNormParam &normParam);
    virtual void SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam);
    virtual void SetLayerNodeInput(atb_speed::Model::Node &layerNode, uint32_t layerId);
    virtual void SetLayerNodeDefaultInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    virtual void SetLayerNodeRaInput(atb_speed::Model::Node &layerNode, uint32_t &inTensorId);
    virtual void SetLayerNodeLoraInput(atb_speed::Model::Node &layerNode, uint32_t layerId, uint32_t &inTensorId);
    virtual atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId);
    virtual atb::Status AddWordEmbedding();
    virtual atb::Status AddPositionalEmbedding();
    virtual atb::Status AddLayer();
    virtual atb::Status AddFinalNorm();
    virtual atb::Status AddLmhead();
    virtual atb::Status AddOperationToGraph();
    int64_t BuildGraph() override;

    // 每次forward时动态变化
    std::vector<int> tokenOffset = {};
    std::vector<int> seqLen = {};
    std::vector<int> qLen = {};

    std::map<std::string, std::vector<std::string>> inTensorCandidates = {};
    std::map<std::string, std::vector<std::string>> internalTensorCandidates = {};
    std::map<std::string, uint32_t> inTensorMap = {};
    std::map<std::string, uint32_t> internalTensorMap = {};
    uint32_t weightCountPerLayer = 50;
    uint32_t weightCountWordEmbedding = 1;
    uint32_t weightCountFinalNorm = 1;
    uint32_t weightCountLmHead = 1;

    ModelParam param;
};

}  // namespace base
}  // namespace atb_speed
#endif
