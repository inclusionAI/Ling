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
#include "atb_context_factory.h"
#include "atb_speed/log.h"
#include "config.h"

namespace atb_torch {
AtbContextFactory &AtbContextFactory::Instance()
{
    static AtbContextFactory instance;
    return instance;
}

std::shared_ptr<atb::Context> AtbContextFactory::GetAtbContext(void *stream)
{
    if (atbContext_) {
        ATB_SPEED_LOG_DEBUG("AtbContextFactory return localContext");
        return atbContext_;
    }

    ATB_SPEED_LOG_DEBUG("AtbContextFactory create atb::Context start");
    atb::Context *context = nullptr;
    atb::Status st = atb::CreateContext(&context);
    if (st != 0) {
        ATB_SPEED_LOG_ERROR("AtbContextFactory create atb::Context fail");
    }
    if (context) {
        context->SetExecuteStream(stream);
        if (Config::Instance().IsUseTilingCopyStream()) {
            ATB_SPEED_LOG_DEBUG("AtbContextFactory use tiling copy stream");
            context->SetAsyncTilingCopyStatus(true);
        } else {
            ATB_SPEED_LOG_DEBUG("AtbContextFactory not use tiling copy stream");
        }
    }
    this->atbContext_ = std::shared_ptr<atb::Context>(
        context, [](atb::Context* context) {atb::DestroyContext(context);});

    return atbContext_;
}

void AtbContextFactory::FreeAtbContext()
{
    ATB_SPEED_LOG_DEBUG("AtbContextFactory FreeAtbContext start");
    if (!atbContext_) {
        return;
    }

    ATB_SPEED_LOG_DEBUG("AtbContextFactory localContext use_count: " << atbContext_.use_count());
    if (atbContext_.use_count() != 1) {
        return;
    }
    ATB_SPEED_LOG_DEBUG("AtbContextFactory localContext reset");
    atbContext_.reset();
}
} // namespace atb_torch
