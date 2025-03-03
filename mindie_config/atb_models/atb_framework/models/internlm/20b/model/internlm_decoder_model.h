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
#ifndef ATB_SPEED_MODELS_INTERNLM_20B_PARALLEL_DECODER_MODEL_H
#define ATB_SPEED_MODELS_INTERNLM_20B_PARALLEL_DECODER_MODEL_H

#include "models/base/model/decoder_model.h"
#include "models/internlm/20b/layer/internlm_decoder_layer.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace internlm_20b_parallel {

class InternlmDecoderModel : public atb_speed::base::DecoderModel {
public:
    explicit InternlmDecoderModel(const std::string &param);

protected:
    atb::Status CreateLayerOperation(atb::Operation **op, uint32_t layerId) override;
};

REGISTER_MODEL(internlm_20b_parallel, InternlmDecoderModel);

}  // namespace internlm_20b_parallel
}  // namespace atb_speed
#endif
