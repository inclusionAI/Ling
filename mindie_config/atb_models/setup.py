# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

from setuptools import setup


setup(
    name="atb_llm",
    version="0.0.1",
    author="",
    author_email="",
    description="ATB LLM Project",
    long_description="",
    package_dir={'atb_llm': 'atb_llm'},
    package_data={
        '': ['*.xlsx', '*.h5', '*.csv', '*.so', '*.avsc', '*.xml', '*.pkl', '*.sql', '*.ini']
    },
    zip_safe=False,
    python_requires=">=3.7",
)
