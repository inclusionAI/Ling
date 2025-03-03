#!/usr/bin/env python
# coding:utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved
"""
logging
"""
import logging
from logging.handlers import RotatingFileHandler
from atb_llm.utils.env import ENV
from atb_speed.common.log.multiprocess_logging_handler import install_logging_handler


def init_logger(logger: logging.Logger, file_name: str):
    """
    日志初始化
    :param logger:
    :param file_name:
    :return:
    """
    logger.setLevel(logging.INFO)
    # 创建日志记录器，指明日志保存路径,每个日志的大小，保存日志的上限
    flask_file_handle = RotatingFileHandler(
        filename=file_name,
        maxBytes=ENV.python_log_maxsize,
        backupCount=10,
        encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] pid: %(process)d %(filename)s-%(lineno)d: %(message)s')
    # 将日志记录器指定日志的格式
    flask_file_handle.setFormatter(formatter)
    # 为全局的日志工具对象添加日志记录器
    logger.addHandler(flask_file_handle)

    # 添加控制台输出日志
    console_handle = logging.StreamHandler()
    console_handle.setFormatter(formatter)
    logger.addHandler(console_handle)
    install_logging_handler(logger)
    return logger
