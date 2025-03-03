/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#ifndef MIES_ATB_SPEED_ERROR_H
#define MIES_ATB_SPEED_ERROR_H

#include <string>
#include <utility>

namespace atb_speed {
class Error {
public:
    enum class Code {
        OK,
        ERROR,
        INVALID_ARG,
        NOT_FOUND,
    };

    explicit Error(Code code = Code::OK) : code_(code) {}
    explicit Error(Code code, std::string msg) : code_(code), msg_(std::move(msg)) {}

    Code ErrorCode() const
    {
        return code_;
    }

    const std::string &Message() const
    {
        return msg_;
    }

    bool IsOk() const
    {
        return code_ == Code::OK;
    }

    std::string ToString() const;

    static const char *CodeToString(const Code code);

protected:
    Code code_;
    std::string msg_;
};
} // namespace atb_speed

#endif