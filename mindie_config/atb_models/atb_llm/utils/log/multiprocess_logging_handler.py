#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
"""
multiprocess_logging_handler
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import atexit
import logging
import multiprocessing
import threading


def install_logging_handler(logger=None):
    """
    Wraps the handlers in the given Logger with an MultiProcessingHandler.
    :param logger: whose handlers to wrap. By default, the root logger.
    """
    if logger is None:
        logger = logging.getLogger("service_operation")

    for index, org_handler in enumerate(list(logger.handlers)):
        handler = MultiLoggingHandler('mp-handler-{0}'.format(index), log_handler=org_handler)
        logger.removeHandler(org_handler)
        logger.addHandler(handler)


class MultiLoggingHandler(logging.Handler):
    """
    multiprocessing handler.
    """

    def __init__(self, name, log_handler=None):
        """
        Init multiprocessing handler
        :param name:
        :param log_handler:
        :return:
        """
        super().__init__()

        if log_handler is None:
            log_handler = logging.StreamHandler()

        self.log_handler = log_handler
        self.queue = multiprocessing.Queue(-1)
        self.setLevel(self.log_handler.level)
        self.set_formatter(self.log_handler.formatter)
        # The thread handles receiving records asynchronously.
        self._is_closed = False
        self.t_thd = threading.Thread(target=self.receive, name=name)
        self.t_thd.daemon = True
        self.t_thd.exception = None
        self.t_thd.start()
        atexit.register(self.close)

    def set_formatter(self, fmt):
        """

        :param fmt:
        :return:
        """
        logging.Handler.setFormatter(self, fmt)
        self.log_handler.setFormatter(fmt)

    def receive(self):
        """

        :return:
        """
        while True:
            try:
                if self._is_closed and self.queue.empty():
                    break
                record = self.queue.get()
                self.log_handler.emit(record)
            except KeyboardInterrupt as err:
                self.t_thd.exception = err
                raise err
            except EOFError:
                break
            except ValueError:
                pass
            except Exception as err:
                self.t_thd.exception = err
                raise err

    def send(self, message):
        """

        :param message:
        :return:
        """
        self.queue.put_nowait(message)

    def emit(self, record):
        """

        :param record:
        :return:
        """
        try:
            sd_record = self._format_record(record)
            self.send(sd_record)
        except KeyboardInterrupt as err:
            raise err
        except ValueError:
            self.handleError(record)

    def close(self):
        """

        :return:
        """
        if not self._is_closed:
            self._is_closed = True
            self.t_thd.join(5.0) # waits 5.0 secs for receive queue to empty 
            self.log_handler.close()
            logging.Handler.close(self)

    def handle(self, record):
        """

        :param record:
        :return:
        """
        if self.t_thd.exception:
            raise self.t_thd.exception
        rsv_record = self.filter(record)
        if rsv_record:
            self.emit(record)
        return rsv_record

    def _format_record(self, org_record):
        """

        :param org_record:
        :return:
        """
        if org_record.args:
            org_record.msg = org_record.msg % org_record.args
            org_record.args = None
        if org_record.exc_info:
            self.format(org_record)
            org_record.exc_info = None
        return org_record