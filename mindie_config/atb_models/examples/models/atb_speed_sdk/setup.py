#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
setup
"""

from setuptools import find_packages, setup

setup(name='atb_speed',
      version='1.1.0',
      description='atb speed sdk',
      license='MIT',
      keywords='atb_speed',
      packages=find_packages(),
      install_requires=["pandas"],
      package_data={"atb_speed": ["**/*.json"]},
      include_package_data=True
      )
