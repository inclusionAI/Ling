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
#ifndef ATB_SPEED_MODELS_VLMO_FLASH_ATTENTION_MODEL_H
#define ATB_SPEED_MODELS_VLMO_FLASH_ATTENTION_MODEL_H

#include "atb_speed/base/model.h"
#include "atb_speed/log.h"

namespace atb_speed {
namespace vlmo {
class FlashAttentionModel : public Model {
public:
    struct Param {
        double layerNormEps = 0;
        int headNum = 0;
        int dk = 0;
        int layerNum = 0;
        int rank = 0;
        int rankSize = 1;
        int maxTextLen = 40;
        int vlLayerIndex = 10;
        std::string backend = "vlmo";
        void FromString(const std::string &param);
    };

    explicit FlashAttentionModel(const std::string &param);
    ~FlashAttentionModel() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

private:
    int64_t BuildGraph() override;
    Param param_;
    int64_t ADDLayer(int &layerId, atb::Tensor *&firstInTensor);
    int64_t ADDVlLayer(int &layerId, atb::Tensor *&firstInTensor);
    int64_t ConstructLayerInTensors(size_t &inTensorId, atb::Tensor *&firstInTensor, atb_speed::Model::Node &layerNode,
                                    const int &layerId);
    atb::Status ParseParam(const std::string &param) override;
    atb::Status BindParamHostTensor(uint32_t nodeId) override;
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
};
} // namespace vlmo
} // namespace atb_speed
#endif
