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
#include "atb_speed/utils/model_factory.h"
#include "atb_speed/log.h"

namespace atb_speed {
bool ModelFactory::Register(const std::string &modelName, CreateModelFuncPtr createModel)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        if (it->second == nullptr) {
            ATB_SPEED_LOG_ERROR("find modelName error: " << modelName);
            return false;
        }
        ATB_SPEED_LOG_WARN(modelName << " model already exists, but the duplication doesn't matter.");
        return false;
    }
    ModelFactory::GetRegistryMap()[modelName] = createModel;
    return true;
}

std::shared_ptr<atb_speed::Model> ModelFactory::CreateInstance(const std::string &modelName, const std::string &param)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        ATB_SPEED_LOG_DEBUG("find model: " << modelName);
        return it->second(param);
    }
    ATB_SPEED_LOG_WARN("ModelName: " << modelName << " not find in model factory map");
    return nullptr;
}

std::unordered_map<std::string, CreateModelFuncPtr> &ModelFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateModelFuncPtr> modelRegistryMap;
    return modelRegistryMap;
}
} // namespace atb_speed