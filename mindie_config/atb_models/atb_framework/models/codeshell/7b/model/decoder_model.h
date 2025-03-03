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
#ifndef ATB_SPEED_MODELS_CODESHELL_7B_DECODER_MODEL_H
#define ATB_SPEED_MODELS_CODESHELL_7B_DECODER_MODEL_H

#include "atb_speed/utils/model_factory.h"
#include "models/codeshell/7b/layer/decoder_layer.h"
#include "models/base/model/decoder_model.h"

namespace atb_speed {
namespace codeshell_7b {

class DecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit DecoderModel(const std::string &param);
    atb::Status InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                           std::vector<atb::TensorDesc> &outTensorDescs) override;

protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
    void SetLmHeadParam(atb_speed::common::LmHeadParam &lmHeadParam) override;
};

REGISTER_MODEL(codeshell_7b, DecoderModel);
}  // namespace codeshell_7b
}  // namespace atb_speed
#endif
