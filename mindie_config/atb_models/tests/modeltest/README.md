# ModelTest README

ModelTest为大模型的性能和精度提供测试功能。

目前支持：

1. NPU，PA场景，性能/性能摸高测试/精度测试，float16/bfloat16，单机/多机多卡；FA场景，性能测试/CEval数据集精度测试，float16，单机多卡
2. GPU，FA场景，精度测试，float16

功能：

1. 性能测试：
    1. 指定batch，指定输入输出长度的e2e性能、吞吐，首Token以及非首Token性能，吞吐。
    2. 性能摸高，指定输入输出长度，指定摸高区间，指定最大非首token时延下的最大batch_size的信息，性能数据以及中间信息。
2. 精度测试：CEval, BoolQ, HumanEval, HumanEval_X, CMMLU, MMLU, TruthfulQA, GSM8K下游数据集

数据集支持：

需要在运行之前先进行数据集的拉取，请参考[data_preparation.md](./docs/user_guides/data_preparation.md)进行配置
1. BoolQ
2. CEval
3. CMMLU
4. HumanEval
5. HumanEval_X
6. GSM8K
7. LongBench
8. MMLU
9. NeedleBench
10. TruthfulQA

PA模型支持：

1. LLaMA (LLaMA-7B, LLaMA-13B, LLaMA-33B,LLaMA-65B, LLaMA2-7B, LLaMA2-13B, LLaMA2-70B, LLaMA3-8B, LLaMA3-70B, LLaMA3.1-8B, LLaMA3.1-70B, LLaMA3.1-405B, LLaMA3.2-1B-Instruct, LLaMA3.2-3B-Instruct)
2. Starcoder (Starcoder-15.5B, Starcoder2-15B)
3. ChatGLM（ChatGLM2-6B, ChatGLM3-6B, ChatGLM3-6b-32k, Glm4-9B-Chat, GLM-4-9B-Chat-1M）
4. CodeGeeX2-6B
5. Baichuan2 (Baichuan2-7B, Baichuan2-13B)
6. Qwen (Qwen-7B,Qwen-14B, Qwen-72B,Qwen1.5-14B,Qwen-14B-chat,Qwen-72B-chat,Qwen1.5-0.5B-chat,Qwen1.5-4B-chat,Qwen1.5-7B,Qwen1.5-14B-chat,Qwen1.5-32B-chat,Qwen1.5-72B,Qwen1.5-110B,Qwen1.5-MoE-A2.7B,Qwen2-57B-A14B,Qwen2-72b-instruct)
7. Aquila (Aquila-7B)
8. Deepseek (Deepseek16B, Deepseek-LLM-7B, Deepseek-LLM-67B, Deepseek-Coder-1.3B, Deepseek-Coder-6.7B, Deepseek-Coder-7B, Deepseek-Coder-33B)
9. Mixtral (Mixtral-8x7B, Mixtral-8x22B)
10. Bloom-7B
11. Baichuan1 (Baichuan1-7B, Baichuan1-13B)
12. CodeLLaMA (CodeLLaMA-7B, CodeLLaMA-13B, CodeLLaMA-34B, CodeLLaMA-70B)
13. Yi (Yi-6B-200K, Yi-34B, Yi-34B-200K)
14. Chinese Alpaca (Chinese-Alpaca-13B)
15. Vicuna (Vicuna-7B, Vicuna-13B)
16. Internlm (Internlm_20b, Internlm2_7b, Internlm2_20b, Internlm2.5_7b)
17. Gemma（Gemma_2b, Gemma-7b）
18. Mistral（Mistral-7B-Instruct-v0.2）
19. Ziya（Ziya-Coding-34B）
20. CodeShell (CodeShell-7B)
21. Yi1.5 (Yi-1.5-6B, Yi-1.5-9B, Yi-1.5-34B)
22. gptneox_20b (GPT-NeoX-20B)
23. telechat (Telechat-7B,Telechat-12B)
24. Deepseek-V2 (Deepseek-V2-Chat)
25. Tencent-Hunyuan-Large (Hunyuan-A52B-Instruct)
26. Phi-3 (Phi-3-mini-128k-instruct)

# 使用说明

### 环境变量

```shell
# source cann环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source 加速库环境变量
source /usr/local/Ascend/nnal/atb/set_env.sh
# source 模型仓tar包解压出来后的环境变量
source set_env.sh
# 设置使用卡号
export ASCEND_RT_VISIBLE_DEVICES="[卡号]" # NPU场景，如"0,1,2,3,4,5,6,7"
或
export CUDA_VISIBLE_DEVICES="[卡号]" # GPU场景，如"0,1,2,3,4,5,6,7"

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

### 安装python依赖

```
pip install -r requirements.txt
```

### 运行指令
```
统一说明：
1. model_name:
    LLaMA-7B, LLaMA-13B, LLaMA-33B,LLaMA-65B, LLaMA2-7B, LLaMA2-13B, LLaMA2-70B, LLaMA3-8B, LLaMA3-70B, LLaMA3.1-8B, LLaMA3.1-70B, LLaMA3.1-405B, LLaMA3.2-1B-Instruct, LLaMA3.2-3B-Instruct: llama
    CodeLLaMA-7B, CodeLLaMA-13B, CodeLLaMA-34B, CodeLLaMA-70B: codellama
    Chinese-Alpaca-13B: llama
    Yi-6B-200K, Yi-34B, Yi-34B-200K: yi
    Vicuna-7B, Vicuna-13B: vicuna
    Starcoder-15.5B: starcoder
    Starcoder2-15B: starcoder2
    ChatGLM2-6B, ChatGLM3-6B, ChatGLM3-6B-32k, Glm4-9B-Chat, GLM-4-9B-Chat-1M: chatglm
    CodeGeeX2-6B: codegeex2_6b
    Baichuan2-7B: baichuan2_7b
    Baichuan2-13B: baichuan2_13b
    Internlm-20B, Internlm2-7B, Internlm2-20B, Internlm2.5-7B: internlm
    Qwen-7B,Qwen-14B, Qwen-72B,Qwen1.5-14B,Qwen-14B-chat,Qwen-72B-chat,Qwen1.5-0.5B-chat,Qwen1.5-4B-chat,Qwen1.5-7B,Qwen1.5-14B-chat,Qwen1.5-32B-chat,Qwen1.5-72B,Qwen1.5-110B,Qwen1.5-MoE-A2.7B,Qwen2-57B-A14B,Qwen2-72b-instruct: qwen
    Aquila-7B: aquila_7b
    Deepseek16B: deepseek
    Deepseek-LLM-7B, Deepseek-LLM-67B: deepseek_llm
    Deepseek-Coder-1.3B, Deepseek-Coder-6.7B, Deepseek-Coder-7B, Deepseek-Coder-33B: deepseek_coder
    Deepseek-V2-Chat: deepseekv2
    Mixtral-8x7B, Mixtral-8x22B: mixtral
    Mistral-7B-Instruct-v0.2: mistral
    Bloom-7B: bloom
    Baichuan1-7B: baichuan2_7b
    Baichuan1-13B: baichuan2_13b
    Gemma-2B,Gemma-7B：gemma
    Ziya-Coding-34B：ziya
    CodeShell-7B: codeshell_7b
    Yi-1.5-6B, Yi-1.5-9B, Yi-1.5-34B: yi1_5
    GPT-NeoX-20B: gptneox_20b
    Telechat-7B, Telechat-12B: telechat
    GPT-2: gpt2
    Hunyuan-A52B-Instruct: hunyuan
    Phi-3-mini-128k-instruct: phi3
2. data_type: 测试时使用的数据类型，fp16或bf16。注意：当传入的data_type与权重config文件中的data_type不一致时，会报错退出。
3. is_chat_instruct_model: 是否使用模型的chat版本，传入chat为使用chat版本模型，传入instruct为使用instruct版本模型，传入base或不传入为使用base版本模型
4. lora: 是否传入每个请求对应的lora adapter的名字。传入"lora"使能此特性。
5. lora_data_path: jsonl格式的文件路径，包含每个请求id对应lora adapter的名称，其中每一行的格式为`{"${req_idx}": "${lora_adapter_name}"}`
6. weight_dir: 权重路径
7. trust_remote_code: 可选参数，是否信任远程模型代码，传入"trust_remote_code"则为True，不传入默认为False
8. chip_num: 使用的卡数
9. max_position_embedding: 可选参数，不传入则使用config中的默认配置
```

#### 性能测试（指定batch_size）
```
# NPU

## 单机场景
### PA场景
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_instruct_model]) (lora [lora_data_path]) [weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])
### FA场景
bash run.sh fa performance [case_pair] [batch_size] [model_name] ([is_chat_instruct_model]) (lora [lora_data_path]) [weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])

##多机场景
### PA场景
bash run.sh pa_[data_type] performance [case_pair] [batch_size] ([prefill_batch_size]) [model_name] ([is_chat_instruct_model]) [weight_dir] ([trust_remote_code]) [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address]([max_position_embedding/max_sequence_length])

或

# GPU
不支持

说明:
1. case_pair接收一组或多组输入，格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 中间不接受空格，如[[256,256],[512,512]];值得关注的是，当输入多组时，应按照seq_in + seq_out从大到小的顺序排列, 如[[2048,2048],[1024,1024],[512,512],[256,256]], 否则可能会导致测试性能不准确。
2. 如果报错显示"Error caught during warm up:"，说明warm up过程出现问题：
    1. 如果是"out of memory"问题，请删除最大内存占用对应的case_pair和batch_size组合，并确保它们不再出现在后续测试中。
    2. 如果发生其他报错，请按照日志报错信息进行相应修改。
3. 如果报错显示"Error caught during inference:"，会提示失败的用例，请根据报错信息进行对应修改，并重新启动性能测试。
4. batch_size接收单个，多个或多组输入，其中：
    1. 单个输入：以数字或者[数字]格式输入，如1或[1]
    2. 多个输入：以多个数字逗号隔开或者[多个数字逗号隔开]格式输入，如1,4或[1,4]
    3. 多组输入：输入的组数要求与case_pair相同且一一对应，格式为[[bs1, bs2],...,[bs3,bs4]]，如当前case_pair输入为[[256,256],[512,512]], batch_size输入为[[1,4],[1,8]]，则对于[256,256]会测试 1,4 batch，对于[512,512]会测试 1,8 batch
5. 运行完成后，会在控制台末尾呈现保存数据的文件夹
6. 在多机场景下，
    1. rank_table_file: 路径字符串，为rank_table信息，一般为json文件，如/home/rank_table.json
    2. world_size: 数字，总卡数信息，多机卡数之和，如16
    3. node_num: 数字，节点数量，即多机数量， 如2
    4. rank_id_start: 数字，当前rank_id的起始值，如两台机器16卡的情况下，第一台的rank_id_start为0，第二台的rank_id_start为8
    说明：
    1. 当前默认world_size与node_num的商为每个节点的卡数，即每个节点卡数平均分配
    2. 当前默认每个节点使用的卡数为该节点前world_size / node_num张
    3. world_size，node_num，rank_id_start与rank_table_file信息保持一致

```
#### 性能测试（摸测最大batch_size）
```
# NPU

## 单机场景
bash run.sh pa_[data_type] performance_maxbs [case_pair] [batch_range] [time_limit] [model_name] ([is_chat_instruct_model]) [weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])

## 多机场景
不支持

或

# GPU
不支持

说明:
1. case_pair接收一组或多组输入，格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 中间不接受空格，如[[256,256],[512,512]]
2. batch_range接收一组或多组输入，数量与case_pair的组数一致，表达对应的case_pair会在给定的batch_range中寻找摸测满足time_limit的最大batch_size.
格式为[[lb1,rb1],...,[lbn,rbn]]，其中区间均为闭区间。如[[1,1000],[200,300]]
3. time_limit：摸测最大bs时的非首token时延最大值。
4. 结果保存: [...]/tests/modeltest/result/模型名/ 下，
    1. 以"_round_result.csv"结尾的文件内保存了过程数据
    2. 以"_final_result.csv"结尾的文件内保存了最终数据, 会呈现在控制台末尾
```

#### 精度测试（下游数据集）
```
# NPU

##单机场景
### PA场景
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_instruct_model]) (lora [lora_data_path]) [weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])
### FA场景
bash run.sh fa full_CEval ([shots]) [batch_size] [model_name] ([is_chat_instruct_model]) (lora [lora_data_path]) [weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])

##多机场景
### PA场景
bash run.sh pa_[data_type] [dataset] ([shots]) [batch_size] [model_name] ([is_chat_instruct_model]) [weight_dir] ([trust_remote_code]) [rank_table_file] [world_size] [node_num] [rank_id_start] [master_address]([max_position_embedding/max_sequence_length])

或

# GPU
bash run.sh fa [dataset] ([shots]) [batch_size] [model_name] ([is_chat_instruct_model]) [weight_dir] ([trust_remote_code]) [chip_num]

说明：
1. dataset接受一个输入，根据需要从以下列表中选择：
    [full_CEval|full_BoolQ|full_HumanEval|full_HumanEval_X|full_MMLU|full_CMMLU|full_TruthfulQA|full_LongBench|full_GSM8K|full_NeedleBench]
2. batch_size接收单个或多个输入，其中：
    1. 单个输入：以数字或者[数字]格式输入，如1或[1]
    2. 多个输入：以多个数字逗号隔开或者[多个数字逗号隔开]格式输入，如1,4或[1,4]
3. shots: 当测试full_CEval,full_MMLU和full_CMMLU时，shot为测试时使用的shot数，如0或5
4. 运行完成后，会在控制台末尾呈现保存数据的文件夹
5. 在多机场景下：
    同性能测试（指定batch_size)章节多机场景说明
6. 在测试full_HumanEval_X时，可参照dataset\full\HumanEval_X\install_humaneval_x_dependency.sh配置依赖环境
7. multi-lora特性仅支持BoolQ数据集
8. 测试full_NeedleBench大海捞针时，shots对应填的数据为上下文长度，目前支持的选项为4k,8k,32k,128k,200k,256k,1000k, 举例：
    bash run.sh pa_bf16 full_NeedleBench 4k 1 llama /data/llama3-8b-instruct/ 8
9. NeedleBench大海捞针目前只支持单一信息检索任务：评估LLM在长文本中提取单一关键信息的能力，测试其对广泛叙述中特定细节的精确回忆能力。
10. 目前数据集精度测试暂不支持多卡多进程测试任务。

``` 

#### 单用例测试（性能/精度）
```
# NPU

## 性能
bash run.sh pa_[data_type] performance_single [case_pair] [input_text_or_file] [batch_size] [model_name] ([is_chat_instruct_model])[weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])

## 精度
bash run.sh pa_[data_type] precision_single [case_pair] [input_text_or_file] [batch_size] [model_name] ([is_chat_instruct_model])[weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])

或

# GPU

## 性能
不支持

## 精度
bash run.sh fa precision_single [case_pair] [input_text_or_file] [batch_size] [model_name] ([is_chat_instruct_model])[weight_dir] ([trust_remote_code]) [chip_num] ([max_position_embedding/max_sequence_length])

说明:
1. case_pair接收一组或多组输入，格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 中间不接受空格，如[[256,256],[512,512]]
2. input_text_or_file接受一个或多个文本输入或者一个文件输入，如'["hello","hi"]'或input.txt
3. batch_size接收单个或多个输入，其中：
    1. 单个输入：以数字或者[数字]格式输入，如1或[1]
    2. 多个输入：以多个数字逗号隔开或者[多个数字逗号隔开]格式输入，如1,4或[1,4]
4. seq_in会根据实际输入决定(性能测试时会影响warmup时分配的空间)，batch数会根据实际输入数量决定（如果input_text_or_file中输入数量大于batch_size，那么会以实际输入数量作为batch_size）
5. 运行完成后，会在控制台末尾呈现保存数据的文件夹

```
举例：
```
1. 测试Llama-70B在8卡[512, 512]场景下，pa_fp16 16batch的性能
bash run.sh pa_fp16 performance [[512,512]] 16 llama [权重路径] 8
2. 测试Llama-65B在双机八卡的[256,256]场景下，pa_fp16 1batch的性能
node0：
bash run.sh pa_fp16 performance [[256,256]] 1 llama [权重路径] [rank_table文件路径] 8 2 0 [master_address]
node1:
bash run.sh pa_fp16 performance [[256,256]] 1 llama [权重路径] [rank_table文件路径] 8 2 4 [master_address]
3. 测试Llama-70B使用归一代码在8卡pa_fp16 [256,256]场景下，[600,700]范围内与[512,512]场景下，[300,400]范围内，非首token时延在50ms以下时的最大batch_size
bash run.sh pa_fp16 performance_maxbs [[256,256],[512,512]] [[600,700],[300,400]] 50 llama [权重路径] 8
4. 测试Starcoder-15.5B在8卡pa_fp16下1 batch下游数据集BoolQ的精度
bash run.sh pa_fp16 full_BoolQ 1 starcoder [权重路径] 8
5. 单用例性能/精度测试，测试llama7b在8卡pa_fp16 [256，256]场景下，2 batch输入文本为"hello"和"what is your name"的性能
bash run.sh pa_fp16 performance_single [[256,256]] '["hello","what is your name"]' 2 llama [权重路径] 8
6. 测试Llama3-8B在4卡pa_bf16下batchsize=16时下游数据集GSM8K的精度
bash run.sh pa_bf16 full_GSM8K 16 llama [权重路径] 4
7. 测试Llama3-8B在2卡pa_fp16下batchsize=2时下游数据集BoolQ的精度，每条数据使用指定lora权重推理
bash run.sh pa_fp16 full_BoolQ 2 llama lora [包含每个请求对应lora adapter名称的文件路径] [权重路径] 2
8. 测试Llama3-8B在2卡fa下batch_size=4,shots=5时下游数据集CEval的精度
bash run.sh fa full_CEval 5 4 llama [权重路径] 2
9. 测试Llama3.1-405B在4机32卡的[256,256]场景下，pa_bf16 1batch的性能
node0：
bash run.sh pa_bf16 performance [[256,256]] 1 llama [权重路径] [rank_table文件路径] 32 4 0 [master_address]
node1:
bash run.sh pa_bf16 performance [[256,256]] 1 llama [权重路径] [rank_table文件路径] 32 4 8 [master_address]
node2:
bash run.sh pa_bf16 performance [[256,256]] 1 llama [权重路径] [rank_table文件路径] 32 4 16 [master_address]
node3:
bash run.sh pa_bf16 performance [[256,256]] 1 llama [权重路径] [rank_table文件路径] 32 4 24 [master_address]
```


## starcoder 特别运行操作说明

- 对于300I DUO设置环境变量，修改core/starcoder_test.py中prepare_environ函数。

```python
os.environ['ATB_LAUNCH_KERNEL_WITH_TILING'] = "1"
os.environ['HCCL_OP_EXPANSION_MODE'] = "AI_CPU"
```

## baichuan2-13b 特别运行操作说明

- 对于300I DUO设置环境变量，修改core/baichuan2_13b_test.py中prepare_environ函数。

```shell
os.environ['ATB_OPERATION_EXECUTE_ASYNC'] = "0"
os.environ['TASK_QUEUE_ENABLE'] = "0"
```

### 多机推理
- 在性能测试时开启"AIV"提升性能，若有确定性计算需求时建议关闭"AIV"
```shell
   export HCCL_OP_EXPANSION_MODE="AIV"
```
- 建议同时运行双机命令