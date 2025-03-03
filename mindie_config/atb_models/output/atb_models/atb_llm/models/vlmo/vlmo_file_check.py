# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import argparse
import stat
import logging as logger
from enum import Enum

MAX_PATH_LENGTH = 4096


class ErrorCode(str, Enum):
    #ATB_MODELS
    ATB_MODELS_PARAM_OUT_OF_RANGE = "MIE05E000000"
    ATB_MODELS_MODEL_PARAM_JSON_INVALID = "MIE05E000001"
    ATB_MODELS_EXECUTION_FAILURE = "MIE05E000002"

    def __str__(self):
        return self.value

FILE_PATH_ERROR = "The file path should not be None."
FILE_NOT_EXIST_ERROR = "The file path not exist."
ERROR_CODE = 'error_code'


def file_check(path: str):
    if path is None:
        logger.error(FILE_PATH_ERROR,
                     extra={ERROR_CODE: ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise argparse.ArgumentTypeError(FILE_PATH_ERROR)
    if not os.path.exists(path):
        logger.error(FILE_NOT_EXIST_ERROR,
                     extra={ERROR_CODE: ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise argparse.ArgumentTypeError(FILE_NOT_EXIST_ERROR)
    if path.__len__() > MAX_PATH_LENGTH:
        logger.error(" The length of path should not be greater than {MAX_PATH_LENGTH}",
                     extra={ERROR_CODE: ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE})
        raise argparse.ArgumentTypeError(f"The length of path should not be greater than {MAX_PATH_LENGTH}.")
    if os.path.islink(os.path.normpath(path)):
        logger.error("The path should not be a symbolic link file.",
                     extra={ERROR_CODE: ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise argparse.ArgumentTypeError("The path should not be a symbolic link file.")
    path = os.path.realpath(path)
    file_dir = os.path.dirname(path)
    path_stat = os.stat(file_dir)
    path_owner, path_gid = path_stat.st_uid, path_stat.st_gid
    user_check = path_owner == os.getuid() and path_owner == os.geteuid()
    if not (path_owner == 0 or path_gid in os.getgroups() or user_check):
        logger.error("The path is not owned by current user or root",
                     extra={ERROR_CODE: ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE})
        raise argparse.ArgumentTypeError("The path is not owned by current user or root")
    mode = path_stat.st_mode
    # check the write permission for others
    if mode & stat.S_IWOTH:
        logger.error("The file should not be writable by others "
                     "who are neither the owner nor in the group",
                     extra={ERROR_CODE: ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE})
        raise argparse.ArgumentTypeError("The file should not be writable by others "
                                         "who are neither the owner nor in the group")
    return path