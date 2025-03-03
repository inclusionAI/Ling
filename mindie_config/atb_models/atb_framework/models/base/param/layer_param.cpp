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

#include "models/base/param/layer_param.h"

namespace atb_speed {
namespace base {

void LayerParam::PrintParam()
{
    Param::PrintParam();
    std::stringstream ss;
    ss << "Base Layer Param:"
       << ", tensorParallelInfo.rank:" << this->tensorParallelInfo.rank
       << ", tensorParallelInfo.worldSize:" << this->tensorParallelInfo.worldSize
       << ", tensorParallelInfo.backend:" << this->tensorParallelInfo.backend
       << ", tensorParallelInfo.rankTableFile:" << this->tensorParallelInfo.rankTableFile
       << ", tensorParallelInfo.quantType:" << this->tensorParallelInfo.quantType
       << ", tensorParallelInfo.outDataType:" << this->tensorParallelInfo.outDataType;
    for (size_t i = 0; i < packQuantType.size(); ++i) {
        ss << "packQuantType[" << i << "]:" << packQuantType.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearQuantType.size(); ++i) {
        ss << "linearQuantType[" << i << "]:" << linearQuantType.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearHasBias.size(); ++i) {
        ss << "linearHasBias[" << i << "]:" << linearHasBias.at(i) << std::endl;
    }
    for (size_t i = 0; i < linearTransposeType.size(); ++i) {
        ss << "linearTransposeType[" << i << "]:" << linearTransposeType.at(i) << std::endl;
    }
    ATB_SPEED_LOG_DEBUG(ss.str());
}

void LayerParam::CheckParam()
{
    if (this->hiddenSizePerAttentionHead == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param hiddenSizePerAttentionHead is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
}
} // namespace base
} // namespace atb_speed