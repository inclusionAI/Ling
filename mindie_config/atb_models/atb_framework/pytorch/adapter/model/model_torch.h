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
#ifndef MODEL_MODEL_TORCH_H
#define MODEL_MODEL_TORCH_H
#include <memory>
#include <string>
#include <vector>

#include <torch/custom_class.h>
#include <torch/script.h>

#include "atb_speed/base/model.h"

#include "models/llama/model/decoder_model.h"

namespace atb_speed {
class ModelTorch : public torch::CustomClassHolder {
public:
    explicit ModelTorch(std::string modelName);
    ~ModelTorch() override;
    int64_t SetParam(std::string param);
    int64_t SetWeight(std::vector<torch::Tensor> atWeightTensors);
    int64_t SetKVCache(std::vector<torch::Tensor> atKCacheTensors, std::vector<torch::Tensor> atVCacheTensors);
    std::vector<torch::Tensor> Execute(std::vector<torch::Tensor> atInTensors, std::string param);
    int64_t ExecuteOut(std::vector<torch::Tensor> atInTensors, std::vector<torch::Tensor> atOutTensors,
        std::string param);
    c10::intrusive_ptr<ModelTorch> clone() const { return c10::make_intrusive<ModelTorch>(modelName_); }

private:
    void AtTensor2Tensor(std::vector<torch::Tensor> &atTensors, std::vector<atb::Tensor> &opsTensors) const;
    int64_t ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                        const std::string &param);
    void* GetWorkSpace(const uint64_t bufferSize) const;
    atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc);
    void RunTask(std::string taskName, std::function<int()> task) const;
private:
    std::string modelName_;
    std::shared_ptr<atb_speed::Model> model_;
    uint64_t executeCount_ = 0;
    uint64_t modelId_ = 0;
    std::shared_ptr<atb::Context> context_;
    std::vector<torch::Tensor> atInternalTensors_;
};
}

#endif