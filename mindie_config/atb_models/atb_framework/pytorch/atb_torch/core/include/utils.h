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
#ifndef EXAMPLE_UTIL_H
#define EXAMPLE_UTIL_H
#include <vector>
#include <string>
#include <atb/types.h>
#include <torch/torch.h>
#include "atb/operation.h"

namespace atb_torch {
class Utils {
public:
    static void *GetCurrentStream();
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor);
    static at::Tensor NpuFormatCast(const at::Tensor &tensor);
    static void BuildVariantPack(const std::vector<torch::Tensor> &inTensors,
                                 const std::vector<torch::Tensor> &outTensors, atb::VariantPack &variantPack);
    static bool AtTensorShapeEqualToTensor(const at::Tensor &atTensor, const atb::TensorDesc &tensorDesc);
    static atb::Tensor AtTensor2Tensor(const at::Tensor &atTensor);
    static at::Tensor CreateAtTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);
    static void ContiguousAtTensor(std::vector<torch::Tensor> &atTensors);
    static void ContiguousAtTensor(torch::Tensor &atTensor);
    static std::string AtTensor2String(const at::Tensor &atTensor);
    static std::string TensorToString(const atb::Tensor &tensor);
    static std::string TensorDescToString(const atb::TensorDesc &tensorDesc);
    static void *CreateWorkspae(uint64_t workspaceSize);
};
}

#endif