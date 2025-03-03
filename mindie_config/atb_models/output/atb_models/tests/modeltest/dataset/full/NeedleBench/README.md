# NeedleBench README

NeedleBench为基于OpenCompass的大海捞针评测标准。

目前支持：

1. 单一信息检索任务(Single-Needle Retrieval Task, S-RT)：评估LLM在长文本中提取单一关键信息的能力，测试其对广泛叙述中特定细节的精确回忆能力。


# 文件说明

1. needlebench_single.py 大海捞针数据集，支持4k,8k,32k,128k,200k,256k,1000k。其中：
    1. context_lengths 为序列长度列表， depths_list 为针插入长文本的位置
    2. 英文数据集为PaulGrahamEssays， 中文数据集为zh_finance

2. 9b5ad71b2ce5302211f9c61530b329a4922fc6a4文件为tiktoken需要的缓存文件

3. data目录为NeedleBench支持的数据文本。其中：
    1. needles.jsonl 存储了所有的针
    2. names.json 存储中英文对照的人名

# GPU长序列测试精度说明
如遇到gpu oom可以参考以下说明
1. 请修改needlebench_single.py中test_single_xxk的代码，用以单独测试以防oom
    - 例如：test_single_128k中的context_lengths = [16000, 32000, 48000, 64000, 80000, 96000, 112000, 128000]
    - 我们需要单独跑里面的用例，最后再手动取所有用例的平均值
    - 如：context_lengths = [16000]；context_lengths = [32000]