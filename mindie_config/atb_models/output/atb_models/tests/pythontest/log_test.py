# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import unittest
from io import StringIO
import logging
import os
import time

from atb_llm.utils.file_utils import safe_open
from mindie_llm.utils.log.logging import init_logger
from mindie_llm.utils.log.error_code import ErrorCode
from mindie_llm.utils.env import ENV


class TestCustomLoggerAndFormatter(unittest.TestCase):
    
    def setUp(self):
        ENV.log_to_file = "1"
        ENV.log_file_maxnum = 10
        ENV.log_file_maxsize = 1000
        self.log_stdout = StringIO()
        self.log_file = 'test_log.log'
        self.logger = init_logger(logging.getLogger(__name__), self.log_file, self.log_stdout)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        # 清理 handlers 和删除日志文件
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_logger_error_with_error_code(self):
        # 测试 error 方法传递错误码的情况
        error_code = ErrorCode.BACKEND_CONFIG_INVALID
        self.logger.error("Test error message with error code", error_code)
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        # 刷新日志处理器输出
        for handler in self.logger.handlers:
            handler.flush()

        # 获取控制台日志输出
        console_output = self.log_stdout.getvalue().strip()

        # 修改正则表达式来匹配控制台输出
        expected_output_pattern = (
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] "  
            r"\[\d+\] "  
            r"\[\d+\] "  
            r"\[llm\] "  
            r"\[ERROR\]\[logging\.py-\d+\] : "  
            r"\[MIE05E020000\] Test error message with error code" 
        )

        # 使用正则表达式匹配控制台输出
        self.assertRegex(console_output, expected_output_pattern)

        # 验证日志文件内容同样使用正则表达式
        with safe_open(self.log_file, 'r') as log_file:
            file_output = log_file.read().strip()

        self.assertRegex(file_output, expected_output_pattern)

    def test_logger_error_without_error_code(self):
        # 测试 error 方法不传递错误码的情况
        self.logger.error("Test error message without error code")
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] "  
            r"\[\d+\] "  
            r"\[\d+\] "  
            r"\[llm\] "  
            r"\[ERROR\]\[logging\.py-\d+\] : "  
            r"Test error message without error code" 
        )

        self.assertRegex(console_output, expected_output_pattern)

        with safe_open(self.log_file, 'r') as log_file:
            file_output = log_file.read().strip()

        self.assertRegex(file_output, expected_output_pattern)

    def test_logger_critical_with_error_code(self):
        # 测试 critical 方法传递错误码的情况
        error_code = ErrorCode.BACKEND_CONFIG_INVALID
        self.logger.critical("Test critical message with error code", error_code)
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] "  
            r"\[\d+\] "  
            r"\[\d+\] "  
            r"\[llm\] "  
            r"\[CRITICAL\]\[logging\.py-\d+\] : "  
            r"\[MIE05E020000\] Test critical message with error code" 
        )

        self.assertRegex(console_output, expected_output_pattern)

        with safe_open(self.log_file, 'r') as log_file:
            file_output = log_file.read().strip()

        self.assertRegex(file_output, expected_output_pattern)

    def test_logger_critical_without_error_code(self):
        # 测试 critical 方法传递错误码的情况
        self.logger.critical("Test critical message without error code")
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] "  
            r"\[\d+\] "  
            r"\[\d+\] "  
            r"\[llm\] "  
            r"\[CRITICAL\]\[logging\.py-\d+\] : "  
            r"Test critical message without error code" 
        )

        self.assertRegex(console_output, expected_output_pattern)

        with safe_open(self.log_file, 'r') as log_file:
            file_output = log_file.read().strip()

        self.assertRegex(file_output, expected_output_pattern)

    def test_logger_info_without_error_code(self):
        self.logger.info("Test info message")
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] "  
            r"\[\d+\] "  
            r"\[\d+\] "  
            r"\[llm\] "  
            r"\[INFO\]\[log_test\.py-\d+\] : "  
            r"Test info message" 
        )

        self.assertRegex(console_output, expected_output_pattern)

        with safe_open(self.log_file, 'r') as log_file:
            file_output = log_file.read().strip()

        self.assertRegex(file_output, expected_output_pattern)

if __name__ == "__main__":
    unittest.main()