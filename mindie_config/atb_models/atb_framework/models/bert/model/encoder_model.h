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

#ifndef ATB_MODELS_BERT_ENCODER_MODEL_H
#define ATB_MODELS_BERT_ENCODER_MODEL_H


#include <atb/atb_infer.h>
#include <nlohmann/json.hpp>
#include "atb_speed/base/model.h"


namespace atb_speed::bert {

    class EncoderModel : public Model {
    public:
        struct Param {
            int dk = 0;
            int64_t geluApproximate = -1;
            int headNum = 0;
            float layerNormEps = 0;
            int64_t layerNormImplMode = 0;
            int layerNum = 0;
            float qkScale = 1.0;
            int rank = 0;
            int rankSize = 1;
            void FromString(const std::string &param);
        };

        explicit EncoderModel(const std::string &param);
        ~EncoderModel() override;

        [[nodiscard]] uint32_t GetInputNum() const override;
        [[nodiscard]] uint32_t GetOutputNum() const override;

        atb::Status InferShape(
            const std::vector<atb::TensorDesc> &inTensorDescs,
            std::vector<atb::TensorDesc> &outTensorDescs
        ) override;

    private:
        int64_t BuildGraph() override;
        int64_t Embedding();
        int64_t Layer();
        Param param_;
        atb::Status ParseParam(const std::string &param) override;
        atb::Status BindParamHostTensor(uint32_t nodeId) override;
        std::vector<int32_t> tokenOffset_;
        std::vector<int32_t> seqLen_;
    };

}  // namespace atb_speed::bert

#endif  // ATB_MODELS_BERT_ENCODER_MODEL_H
