# Large Language Model ModelTest

[🌐 MindIE LLM官方repo](https://gitee.com/ascend/MindIE-LLM.git)
[📖 Modelers社区](https://modelers.cn/MindIE/data.git)
[📊 ModelTest沟通矩阵](https://wiki.huawei.com/domains/12174/wiki/111089/WIKI202408234389473)
[🤔 ModelTest问题收集](https://onebox.huawei.com/v/b31e3181b66ca80bbb68396c7352476b?type=0)
[📘 ModelTest文档教程](./docs/get_started/)
[🛠️ 数据集安装教程](./docs/user_guides/data_preparation.md)
[🚩 ModelTest老版run.sh使用教程](./README.md)

## 🧭 欢迎

欢迎来到 **`ModelTest`**！

经过不懈努力，`ModelTest`希望帮助您评估大模型在不同场景和任务下的性能与精度。`ModelTest`提供了丰富的功能和高效的工具，帮助您轻松完成`LLM`的精确测试与优化。

欢迎加入`ModelTest`！我们目前 **正在完善`ModelTest`的接口与功能**。如果您对大语言模型和`ModelTest`充满热情，请随时通过 [Welink沟通矩阵](https://wiki.huawei.com/domains/12174/wiki/111089/WIKI202408234389473) 与我们联系。期待您的加入！

> **注意**<br />
> ModelTest 工具的共建进行中，诚邀大家为 ModelTest 提供更多具有代表性和可信度的评测数据集和问题反馈！点击 [Issue](https://onebox.huawei.com/v/b31e3181b66ca80bbb68396c7352476b?type=0) 获取和交流更多信息。

## 🚀 最新进展 <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **[2024.11.1]** 支持 TextVQA数据集及qwen_vl多模态模型，欢迎尝试！

- **[2024.10.21]** ModelTest现已支持 BoolQ、CEval、CMMLU、HumanEval、HumanEval_X、GSM8K、LongBench、MMLU、NeedleBench、TruthfulQA，欢迎尝试！

- **[2024.09.26]** ModelTest新架构发布，欢迎尝试！

## 🛠️ 安装指南

### 💻 基础安装

- 你可以通过以下命令从代码仓库安装 `modeltest` 工具：

```bash
git clone https://gitee.com/Ascend/MindIE-LLM.git
cd MindIE-LLM/examples/atb_models/tests/modeltest
pip install -e .
```

- 或者，你可以从安装包中获取 modeltest 工具：

```bash
cd {解压后的目录}/tests/modeltest
pip install -e .
```

### 📂 数据集准备

- 关于如何下载和准备数据集的详细步骤，请参考[data_preparation.md](./docs/user_guides/data_preparation.md)

#### 魔乐社区(内部使用)

网址：`https://modelers.cn/`
##### - 纯语言数据集
在modeltest根目录`MindIE-LLM/examples/atb_models/tests/modeltest`下执行：

```bash
git clone https://modelers.cn/MindIE/data.git
```
##### - 多模态数据集
TextVQA：
```bash
git clone https://modelers.cn/MindIE/textvqa.git
```
VideoBench：
```bash
git clone https://modelers.cn/MindIE/videobench.git
```
VocalSound：
```bash
git clone https://modelers.cn/MindIE/vocalsound.git
```

**注意事项：**
    1. 进入网站请提前切换国内代理。
    2. 进行网站注册，并申请加入MindIE组织。
    3. 用户名：魔乐社区用户名
    4. 密码：参照`https://modelers.cn/docs/zh/openmind-hub-client/quick_start.html`[**访问令牌**]章节进行配置

#### 官网下载(外部使用)

- 首先，需要在test/modeltest路径下新建名为temp_data的文件目录，然后在temp_data文件目录下新建对应数据集文件目录:

|    支持数据集  |     目录名称   |
|---------------|---------------|
|      BoolQ    |     boolq     |
|     CEval     |     ceval     |
|      CMMLU    |     cmmlu     |
|    HumanEval  |   humaneval   |
|   HumanEval_X |  humaneval_x  |
|      GSM8K    |     gsm8k     |
|   LongBench   |   longbench   |
|       MMLU    |     mmlu      |
|  NeedleBench  |   needlebench |
|  VideoBench   |   VideoBench  |
|  Vocalsound   |   Vocalsound  |
|   TextVQA     |   TextVQA     |
|   TruthfulQA  |   truthfulqa  |

- 获取数据集：需要访问huggingface和github的对应网址，手动下载对应数据集

|    支持数据集   |         下载地址            |
|----------------|-----------------------------|
|   BoolQ   |[dev.jsonl](https://storage.cloud.google.com/boolq/dev.jsonl)|
|   CEval   |[ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip)|
|   CMMLU   |[cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip)|
| HumanEval |[humaneval](https://github.com/openai/human-eval/raw/refs/heads/master/data/HumanEval.jsonl.gz)|
|HumanEval_X|[cpp](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/cpp/data)<br>[java](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/java/data)<br>[go](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/go/data)<br>[js](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/js/data)<br>[python](https://huggingface.co/datasets/THUDM/humaneval-x/tree/main/data/python/data)|
|  GSM8K    |[gsm8k](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl)|
| LongBench |[longbench](https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip)|
|    MMLU   |[mmlu](https://people.eecs.berkeley.edu/~hendrycks/data.tar)|
|NeedleBench|[PaulGrahamEssays](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[multi_needle_reasoning_en](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[multi_needle_reasoning_zh](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[names](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[needles](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_finance](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_game](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_general](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_government](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_movie](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)<br>[zh_tech](https://huggingface.co/datasets/opencompass/NeedleBench/tree/main)|
|TextVQA|[train_val_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)<br>[textvqa_val.jsonl](https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val.jsonl)<br>[textvqa_val_annotations.json](https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json)<br>|
|VideoBench|[Eval_QA/](https://github.com/PKU-YuanGroup/Video-Bench)<br>[Video-Bench](https://huggingface.co/datasets/LanguageBind/Video-Bench/tree/main)<br>|
|VocalSound|[VocalSound 16kHz Version](https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1)<br>|
|TruthfulQA|[truthfulqa](https://huggingface.co/datasets/domenicrosati/TruthfulQA/tree/main)|

- 将对应下载的数据集文件放置在对应的数据集目录下，并在modeltest根目录`MindIE-LLM/examples/atb_models/tests/modeltest`下执行：

```bash
python3 scripts/data_prepare.py [可选参数]
```

| 参数名  | 含义                     |
|--------|------------------------------|
| dataset_name | 可选，需要下载的数据集名称，支持的数据集列表参见[**功能**]章节，多个名称以','隔开                 |
| remove_cache | 可选，是否在下载前清除数据集缓存    |

### 💻 安装依赖

#### Python 依赖安装

- 安装项目所需的基础依赖：

```bash
pip install -r requirements/base.txt
```

#### AI 框架依赖安装

- 根据你的环境（NPU 或 GPU），安装相应的依赖：

```bash
pip install -r requirements/npu.txt # NPU
pip install -r requirements/vllm.txt # GPU + VLLM
```

- 对于特定任务的依赖：

```bash
pip install -r requirements/[task_name].txt
```

#### 数据集依赖安装：

- `HumanEval_X`数据集任务的环境配置

在测试`HumanEval_X`时，需要额外安装和配置多语言环境。请参考[humaneval_x_environment.md](./docs/user_guides/humaneval_x_environment.md)

### 🌍 环境变量设置

#### 公共环境变量

##### 日志相关（NPU/GPU）

```bash
# modeltest日志级别设置
当前日志级别设置可以通过以下两种环境变量来控制，MINDIE_LOG_LEVEL的优先级更高

export MINDIE_LOG_LEVEL = "[LEVEL]" #默认为INFO
export MODELTEST_LOG_LEVEL="[LEVEL]" # 默认为INFO

# modeltest是否存储日志到目录
当前是否存储日志到目录可以通过以下两种环境变量来控制，MINDIE_LOG_TO_FILE优先级更高

export MINDIE_LOG_TO_FILE = "[0/1], [false/true]"
export MODELTEST_LOG_TO_FILE="[0/1]" # 保存为1，不保存为0

# modeltest保存的文件名
当前保存文件路径可以通过以下的环境变量来控制
export MINDIE_LOG_PATH = "[path]" #默认写入~/mindie/log路径
```

##### NPU场景下以下环境变量默认设置

```bash
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
export ATB_OPERATION_EXECUTE_ASYNC=1
export ATB_CONVERT_NCHW_TO_ND=1
export TASK_QUEUE_ENABLE=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export ATB_CONTEXT_WORKSPACE_SIZE=0
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

##### NPU精度测试场景下以下环境变量默认设置

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export MODELTEST_DATASET_SPECIFIED=[任务配置yaml中的task_name项]
```

#### NPU

对于 NPU 环境，请使用以下命令设置环境变量：

```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/nnal/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
# 设置使用卡号
export ASCEND_RT_VISIBLE_DEVICES="[卡号]" # NPU场景，如"0,1,2,3,4,5,6,7"
```

#### GPU

对于 GPU 环境，设置以下环境变量：

```shell
export CUDA_VISIBLE_DEVICES="[卡号（设备ID）]" # GPU场景，如"0,1,2,3,4,5,6,7"
```

## 🏗️ 测试

- ModelTest 支持对大模型在多个数据集上的评测。以下步骤展示了如何进行基本操作：模型配置、执行评测、查看结果。

### 📌 用户指引

- 命令行中执行 `modeltest -h`可展示所有支持的参数类型。命令中的`modeltest`可被`model_test`或者`model-test`替换使用。

    | 参数名  | 含义                                             |
    |--------|--------------------------------------------------|
    | model_config_path | 必选，模型配置路径，绝对路径或相对路径（基于modeltest根目录）                 |
    | task_config_path | 必选，任务配置路径，绝对路径或相对路径（基于modeltest根目录）    |
    | batch_size | 可选，batch数，默认为1数                                 |
    | tp | 可选，tensor并行数 ，默认为1                       |
    | output_dir | 可选，输出文件夹路径，默认为modeltest根目录/outputs           |
    | lcoc_disable | 可选，关闭通信计算掩盖，默认开启              |
    | save_debug_enable | 可选，开启保存debug信，默认关闭              |

### 📝 配置文件

- 在`modeltest/config`路径下存储有模型以及任务的yaml格式配置文件，用户在使用前根据测试目的需要对其进行修改。

#### 模型配置

模型的yaml文件在`modeltest/config/model`路径下，参数介绍：

```
model_name: str
model_path: str
model_type: str
data_type: str
use_chat_template: bool
max_position_embedding: int
block_size: int
requested_gpu_framework: str
trust_remote_code: bool
env: dict
mm_model: dict
```

| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| model_name | 模型名称                  |
| model_path | 模型权重文件所在路径    |
| model_type | 模型结构类型，支持`fa`/`pa`                                 |
| data_type | 模型数据类型，支持`fp16`/`bf16`，需要提前配在权重路径下`config.json`中修改`torch_dtype`为对应的参数类型                        |
| use_chat_template | 是否使用chat模板，需要提前在`atb_llm`中进行相关适配           |
| max_position_embedding | 模型最长位置编码长度              |
| block_size | 在`pa`场景下时，block大小              |
| requested_gpu_framework | 在`gpu`环境下需要使用的执行框架，支持`huggingface`/`vllm`              |
| trust_remote_code | 是否信任远程模型代码            |
| env | 模型执行前需要设置的环境变量              |
|mm_model| 【多模态】多模态模型所需设置的参数 |
|mm_model.path|【多模态】llm_model/examples/models/{MODEL}/ 路径下的包含Runner（及其子类）的文件名|
|mm_model.classname|【多模态】名为{path}的文件所含的Runner（及其子类）的类名|
|mm_model.infer_params| 【多模态】为一个字典，其包含的属性为llm_model/examples/models/{MODEL}/run_{model_type}.sh所调用的PARunner或FARunner（及其子类）的infer函数的第二个至最后一个参数的名称（key）及其取值（value）|


#### 任务配置

任务的yaml文件在`modeltest/config/task`路径下，参数介绍：

```
task_type: str
task_name: str
hf_dataset_path: str
om_dataset_path: str
local_dataset_path: str
prompt: str
choices: List
shots: int
requested_max_input_length: int
requested_max_output_length: int
need_logits: bool
need_truncate_input: bool
metric: Dict[str, Union[str, float]]
metric_type: str
metadata_version: str
humaneval_x_datasets_selector: List[str]
subject_mapping: Dict
```

| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| task_type | 任务类型，支持`precision`                  |
| task_name | 任务名称    |
| hf_dataset_path | 预留                                 |
| om_dataset_path | 预留                        |
| local_dataset_path | 数据集路径，基于modeltest根目录（对于TextVQA数据集，为textvqa_val.jsonl的路径           |
| prompt | 数据集提示语              |
| choices | 数据集的选项              |
| shots | 数据集shot数，当task_name为`ceval`/`mmlu`              |
| requested_max_input_length | 数据集要求的最大输入长度              |
| requested_max_output_length | 数据集要求的最大输出长度              |
| need_logits | 数据集运行中是否会基于logits计算（不可更改）              |
| need_truncate_input | 是否需要基于requested_max_input_length对输入进行截断，支持`longbench`              |
| metric | 结果矩阵（不可修改）              |
| metadata_version | 版本号（不可修改）              |
| humaneval_x_datasets_selector | `humaneval_x`数据集类型列表，支持cpp/go/java/js/python              |
| subject_mapping | 任务执行的文件列表，可根据需要增删              |

### 📊 运行简单的模型评测

- 你可以通过以下命令开始评测模型：

#### NPU

- 请先配置好NPU的相关环境变量，然后运行以下命令：

##### 单卡（举例）

```bash
modeltest \
    --model_config_path modeltest/config/model/llama2_7b.yaml \
    --task_config_path modeltest/config/task/boolq.yaml \
    --batch_size 1 \
    --tp 1 \
    --output_dir ./outputs \
    --lcoc_disable \
    --save_debug_enable
```

##### 多卡（举例）

```bash
torchrun \
    --nproc_per_node 4 \
    --master_port 12345 \
    --no-python \
modeltest \
    --model_config_path modeltest/config/model/llama2_7b.yaml \
    --task_config_path modeltest/config/task/boolq.yaml \
    --batch_size 1 \
    --tp 4 \
    --output_dir ./outputs \
    --lcoc_disable \
    --save_debug_enable
```

#### GPU

- 请先配置好GPU的相关环境变量，然后运行以下命令：

##### 单/多卡（举例）

```bash
modeltest \
    --model_config_path modeltest/config/model/llama2_7b.yaml \
    --task_config_path modeltest/config/task/boolq.yaml \
    --batch_size 1 \
    --tp 1 \
    --output_dir ./outputs \
    --lcoc_disable \
    --save_debug_enable
```

### 📚 输出文件说明

- 在每次测试任务完成后，生成并保存相应的测试结果和调试信息，具体如下：

#### 📄 测试结果文件

- 默认生成

- 生成路径：

```
[output_dir]/results/[device_type]/[task_type]_test/[task_name]/[data_type]/[model_name]
```

- 文件名格式：

```
[task_name]_[model_type]_batch[batch_size]_tp[tp]_result.csv
```

- 示例文件路径：

```
./outputs/results/NPU/precision_test/boolq/fp16/llama2_7b/boolq_pa_batch1_tp1_result.csv
```

- 文件内容：测试结果文件会保存每次测试任务的最终评测结果。具体内容包括：任务名称、模型配置、批处理大小 (`batch_size`)、并行度 (`tp`) 等信息。文件的保存路径会记录在日志中，以便用户追踪和访问结果。

#### 🗂️ 调试信息文件

- 由`save_debug_enable`参数控制，`TruthfulQA`数据集不支持生成调试文件。

- 生成路径：

```
[output_dir]/debug/[device_type]/[task_type]_test/[task_name]/[data_type]/[model_name]
```

- 文件名格式：
```
[task_name]_[model_type]_batch[batch_size]_tp[tp]_debug_info.csv
```

- 示例文件路径：

```
./outputs/debug/NPU/precision_test/boolq/fp16/llama2_7b/boolq_pa_batch1_tp1_debug_info.csv
```

- 文件内容：调试信息文件包含详细的调试数据，帮助分析和调试模型性能。文件中的内容来自测试过程中生成的数据，具体包括以下字段：
    - `key`: 测试用例的唯一标识符。
    - `queries`: 测试过程中发送的查询请求。
    - `input_token_ids`: 输入的 token ID 序列，用于模型的推理。
    - `output_token_ids`: 模型生成的输出 token ID 序列。
    - `test_result`: 模型的测试结果。
    - `golden_result`: 参考的标准答案，用于对比测试结果，评估模型输出的准确性。
    - `pass`: 测试结果是否通过的标志（True/False）。
这些调试数据被保存为CSV文件，文件的保存路径会记录在日志中，方便用户定位调试信息文件。

#### 🧪 其他生成文件

##### HumanEval 和 HumanEval_X 的推理结果jsonl文件

- 生成路径：

```
[output_dir]/results/[device_type]/[task_type]_test/[task_name]/[data_type]/[model_name]
```

- 文件名格式：

```
[task_name](_[humaneval_x_datasets_selector])_infer_results.csv
```

- 示例文件路径：

```
./outputs/results/NPU/precision_test/humaneval/fp16/llama2_7b/humaneval_infer_results.csv
```

- 文件内容：针对`humaneval`和`humaneval_x`任务，精度测试的子类会生成`humaneval_infer_results.csv`文件。此文件包含模型在推理任务中的结果和测试精度信息等。`humaneval_x`任务的文件名中包含数据集的选择器`humaneval_x_datasets_selector`，用于区分不同数据集的评测结果，便于后续分析。

**参数解释**

| 参数名  | 含义                                             |
|--------|--------------------------------------------------|
|   output_dir	|   输出文件的根目录，生成所有测试结果、调试信息和日志的存储路径。默认路径为modeltest根目录下的`./outputs`  |
|   device_type |   设备类型，例如 NPU 或 GPU   |
|   task_type   |	任务类型，来自于任务yaml配置文件，在`modeltest/config/task`路径下，表示测试的类型，支持`precision`（精度测试）    |
|   task_name   |	任务名称，来自于任务yaml配置文件，在`modeltest/config/task`路径下    |
|   data_type   |   模型数据类型，来自于模型yaml配置文件，在`modeltest/config/model`路径下，支持`fp16`/`bf16`，需要提前配在权重路径下`config.json`中修改`torch_dtype`为对应的参数类型     |
|   model_name  |	模型名称，来自于模型yaml配置文件，在`modeltest/config/model`路径下    |
|   model_type  |	模型结构类型，来自于模型yaml配置文件，在`modeltest/config/task`路径下，支持`fa`/`pa`    |
|   batch_size  |   batch数，默认为1  |
|   tp  |   tensor并行数 ，默认为1  |

### 📌 补充说明

- `NeedleBench`大海捞针目前只支持单一信息检索任务：评估LLM在长文本中提取单一关键信息的能力，测试其对广泛叙述中特定细节的精确回忆能力。
- `TruthfulQA`数据集仅支持 1 batch。
- 目前数据集精度测试暂不支持多卡同时起多进程测试任务。

##  📖 数据集支持

### NPU

- 下游数据集精度测试
    - BoolQ
    - CEval
    - CMMLU
    - HumanEval
    - HumanEval_X
    - GSM8K
    - LongBench
    - MMLU
    - NeedleBench
    - TextVQA
    - VideoBench
    - VocalSound
    - TruthfulQA

### GPU

- 下游数据集精度测试
    - BoolQ
    - CEval
    - CMMLU
    - HumanEval
    - HumanEval_X
    - GSM8K
    - LongBench
    - MMLU
    - NeedleBench
    - TruthfulQA

##  🤖 模型支持

- LLaMA
    - LLaMA-7B
    - LLaMA-13B
    - LLaMA-33B
    - LLaMA-65B
    - LLaMA2-7B
    - LLaMA2-13B
    - LLaMA2-70B
    - LLaMA3-8B
    - LLaMA3-70B
    - LLaMA3.1-8B
    - LLaMA3.1-70B
    - LLaMA3.1-405B
- Starcoder
    - Starcoder-15.5B
    - Starcoder2-15B
- ChatGLM
    - ChatGLM2-6B
    - ChatGLM3-6B
    - ChatGLM3-6b-32k
    - Glm4-9B-Chat
    - GLM-4-9B-Chat-1M
- CodeGeeX2-6B
- Cogvlm2-llama3-chinese-chat-19B
- Baichuan1
    - Baichuan1-7B
- Baichuan2
    - Baichuan2-7B
    - Baichuan2-13B
- Qwen
    - Qwen-7B
    - Qwen-14B
    - Qwen-72B
    - Qwen1.5-14B
    - Qwen-14B-chat
    - Qwen-72B-chat
    - Qwen-VL
    - Qwen1.5-0.5B-chat
    - Qwen1.5-4B-chat
    - Qwen1.5-7B
    - Qwen1.5-14B-chat
    - Qwen1.5-32B-chat
    - Qwen1.5-72B
    - Qwen1.5-110B
    - Qwen1.5-MoE-A2.7B
    - Qwen2-57B-A14B
    - Qwen2-72b-instruct
    - Qwen2-Audio-7B-Instruct
    - Qwen2-VL-7B-Instruct
- Aquila
    - Aquila-7B
- Deepseek
    - Deepseek16B
    - Deepseek-LLM-7B
    - Deepseek-LLM-67B
    - Deepseek-Coder-1.3B
    - Deepseek-Coder-6.7B
    - Deepseek-Coder-7B
    - Deepseek-Coder-33B
- Mixtral
    - Mixtral-8x7B
    - Mixtral-8x22B
- Bloom-7B
    - Baichuan1-13B
- CodeLLaMA
    - CodeLLaMA-7B
    - CodeLLaMA-13B
    - CodeLLaMA-34B
    - CodeLLaMA-70B
- Yi
    - Yi-6B-200K
    - Yi-34B
    - Yi-34B-200K
    - Yi-VL-6B
    - Yi-VL-34B
- Chinese Alpaca
    - Chinese-Alpaca-13B
- Vicuna
    - Vicuna-7B
    - Vicuna-13B
- Internlm
    - Internlm_20b
    - Internlm2_7b
    - Internlm2_20b
    - Internlm2.5_7b
- Internvl
    - InternVL2-8B
    - InternVL2-40B
- Gemma
    - Gemma_2b
    - Gemma-7b
- Mistral
    - Mistral-7B-Instruct-v0.2
- Ziya
    - Ziya-Coding-34B
- CodeShell
    - CodeShell-7B
- Yi1.5
    - Yi-1.5-6B
    - Yi-1.5-9B
    - Yi-1.5-34B
- gptneox_20b
    - GPT-NeoX-20B
- telechat
    - Telechat-7B
    - Telechat-12B
- Phi-3
    - Phi-3-mini-128k-instruct

## 👷‍♂️ 贡献

我们感谢所有的贡献者为改进和提升`ModelTest`所作出的努力。

Modeltest工具还在持续完善，如有问题，请点击以下链接：
- [📊 ModelTest沟通矩阵](https://wiki.huawei.com/domains/12174/wiki/111089/WIKI202408234389473)
- [🤔 ModelTest问题收集](https://onebox.huawei.com/v/b31e3181b66ca80bbb68396c7352476b?type=0)
