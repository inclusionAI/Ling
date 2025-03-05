#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
cd ${SCRIPT_DIR}/atb_llm
cp -f ./atb_llm-models-__init__.py /usr/local/Ascend/atb-models/atb_llm/models/__init__.py
cp -f ./atb_llm-models-base-router.py /usr/local/Ascend/atb-models/atb_llm/models/base/router.py
cp -f ./config_deepseek.py /usr/local/Ascend/atb-models/atb_llm/models/deepseek/config_deepseek.py
cp -f ./flash_causal_deepseek.py /usr/local/Ascend/atb-models/atb_llm/models/deepseek/flash_causal_deepseek.py
cp -f ./input_builder_deepseek.py /usr/local/Ascend/atb-models/atb_llm/models/deepseek/input_builder_deepseek.py
cp -f ./modeling_deepseek.py /usr/local/Ascend/atb-models/atb_llm/models/deepseek/modeling_deepseek.py
cp -f ./router_deepseek.py /usr/local/Ascend/atb-models/atb_llm/models/deepseek/router_deepseek.py
cp -f ./utils-file_utils.py /usr/local/Ascend/atb-models/atb_llm/utils/file_utils.py
cp -f ./utils-layers-__init__.py /usr/local/Ascend/atb-models/atb_llm/utils/layers/__init__.py
cp -f ./utils-layers-linear-__init__.py /usr/local/Ascend/atb-models/atb_llm/utils/layers/linear/__init__.py
cp -f ./utils-weights.py /usr/local/Ascend/atb-models/atb_llm/utils/weights.py