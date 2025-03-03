#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

path="${BASH_SOURCE[0]}"

if [[ -f "$path" ]] && [[ "$path" =~ 'set_env.sh' ]];then
	atb_models_path=$(cd $(dirname $path); pwd)
	export ATB_SPEED_HOME_PATH="${atb_models_path}"
	export LD_LIBRARY_PATH=$ATB_SPEED_HOME_PATH/lib:$LD_LIBRARY_PATH
	export PYTHONPATH=$ATB_SPEED_HOME_PATH:$PYTHONPATH

	export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
	export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
	export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
	export LD_LIBRARY_PATH=$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH

	export TASK_QUEUE_ENABLE=1 #是否开启TaskQueue, 该环境变量是PyTorch的
	export ATB_OPERATION_EXECUTE_ASYNC=1 # Operation 是否异步运行
	export ATB_CONTEXT_HOSTTILING_RING=1
	export ATB_CONTEXT_HOSTTILING_SIZE=102400
	export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
	export ATB_USE_TILING_COPY_STREAM=0 #是否开启双stream功能
	export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1  # 设置op runner的本地cache 槽位数
	export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=16  # 设置op runner的全局cache 槽位数
else
	echo "There is no 'set_env.sh' to import"
fi