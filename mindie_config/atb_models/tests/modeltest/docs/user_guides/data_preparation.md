# 数据集下载指南

## 魔乐社区(内部使用)

网址：`https://modelers.cn/`

在modeltest根目录`MindIE-LLM/examples/atb_models/tests/modeltest`下执行：

```bash
git clone https://modelers.cn/MindIE/data.git
```   

**注意事项：**  
    1. 进入网站请提前切换国内代理。  
    2. 进行网站注册，并申请加入MindIE组织。  
    3. 用户名：魔乐社区用户名  
    4. 密码：参照`https://modelers.cn/docs/zh/openmind-hub-client/quick_start.html`[**访问令牌**]章节进行配置

## 官网下载(外部使用)  

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
|TruthfulQA|[truthfulqa](https://huggingface.co/datasets/domenicrosati/TruthfulQA/tree/main)|

- 将对应下载的数据集文件放置在对应的数据集目录下，并在modeltest根目录`MindIE-LLM/examples/atb_models/tests/modeltest`下执行：

```bash
python3 scripts/data_prepare.py [可选参数]
```

| 参数名  | 含义                     |
|--------|------------------------------|
| dataset_name | 可选，需要下载的数据集名称，支持的数据集列表参见[**功能**]章节，多个名称以','隔开                 |
| remove_cache | 可选，是否在下载前清除数据集缓存    |