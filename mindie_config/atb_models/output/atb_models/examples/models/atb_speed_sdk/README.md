# atb_speed_sdk

*提高加速库的易用性，统一下游任务，集成公共能力*  
优点：

1. 同时兼容GPU与NPU，最大程度减少迁移适配的工作量
2. 屏蔽NPU与GPU的差异，用户无感切换
3. 一个配置文件覆盖所有配置
4. 进程安全的日志

# sdk安装

```shell
pip install .
```

# 配置文件使用及样例

## 使用

```python
from atb_speed.common.config import atb_speed_config

config_path = "xxxx"
atb_speed_config.init_config(config_path)
```

## 样例

```
[model]
;模型路径
model_path=../model
;使用的设备号,多卡用逗号分隔，设置多卡，将默认使用并行模式
device_ids=2
;并行通信类型，默认是hccl，可选hccl/nccl(GPU)
;parallel_backend=hccl
;日志保存路径，默认是执行脚本所在路径
;log_dir=./
;是否绑核，0或1，默认是1表示开启
;bind_cpu=1

[precision]
;精度测试方法，默认为ceval，可选ceval/mmlu
mode=ceval
;精度测试工作路径
work_dir=./
;批量精度测试，默认是1
batch=1
;每个科目的shot数量，默认是5
shot=5
;每个问题的回答长度，默认是32
;seq_len_out=32

[performance]
;性能测试模型名称，用于结果文件的命名
model_name=vicuna_13b
;测试的batch size
batch_size=1
;测试的输入的最大2的幂
max_len_exp=10
;测试的输入的最小2的幂
min_len_exp=5
;特定用例测试，格式为[[seq_in,seq_out]],注意当设置这个参数时，max_len_exp min_len_exp不生效
;case_pair=[[1,2],[2,3]]
;生成的结果文件名称，默认会自动生成，一般不设置
;save_file_name=
;性能测试方法，detail / normal , 默认是normal.要使用detail需要配合装饰器计时，并加上环境变量 TIMEIT=1
;perf_mode=
;性能测试时是否只测试generate而跳过decode，0/1 默认是0
;skip_decode=
```

# 使用说明
-- trust_remote_code为可选参数代表是否信任本地的可执行文件：默认False。传入此参数为True，则信任本地可执行文件。

最核心的模块是launcher，所有的下游任务都围绕launcher来执行

## launcher [model]

用户通过继承Launcher，多卡继承ParallelLauncher 基类来实现自定义launcher。  
当前的launcher对GPU和NPU做了自适应适配，因此可以通用。  
使用launcher时，用户需要实现自定义的init_model方法，这里需要注意的是，self.model_path是从配置文件中读出的。  
如果要进行功能测试，则需要实现自定义的infer方法。

```python
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


class BaichuanLM(Launcher):

    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = safe_get_tokenizer_from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code, use_fast=False)
        model = safe_get_model_from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    atb_speed_config.init_config()
    baichuan = BaichuanLM()
    print("---------------warm-up---------------")
    baichuan.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->')

    print("---------------inference---------------")
    baichuan.infer('登鹳雀楼->王之涣\n夜雨寄北->')
    baichuan.infer('苹果公司的CEO是')

    query_list = ["谷歌公司的CEO是",
                  '登鹳雀楼->王之涣\n夜雨寄北->',
                  '苹果公司的CEO是',
                  '华为公司的CEO是',
                  '微软公司的CEO是']
    baichuan.infer_batch(query_list)

```

# 精度测试

SDK提供了两种精度测试方法，ceval和mmlu

## 配置说明 [precision]

| 配置项key      | 默认值   | 备注                                |
|-------------|-------|-----------------------------------|
| mode        | ceval | 精度测试方法。可选ceval/mmlu               |
| work_dir    |       | 精度测试工作路径。必填                       |
| batch       | 1     | 批量精度测试的批数，请注意batch大于1时精度会和等于1时有差别 |
| shot        | 5     | 每个科目的shot数量                       |
| seq_len_out | 32    | 每个问题的回答长度                         |

### 1. 下载测试数据集

ceval

```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d data
```

mmlu

```shell
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
```

注:wget网络不通请从网页下载并复制

### 2. 配置精度测试相关项

0. 按照推理指导,下载模型及配置路径，并安装atb_speed_sdk
1. 新建工作文件夹${precision_work_dir}。
2. 将下载的测试数据集进行解压后的数据和脚本放置在${precision_work_dir}
3. 修改config.ini文件设置，设置ceval相关路径

目录结构示例${ceval_work_dir}:  
--test_result 跑完之后生成  
--data (包含：数据文件夹dev、test、val三者)

## 运行脚本

只需要声明一个launcher即可使用

```python
from atb_speed.common.precision import get_precision_test_cls
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


class BaichuanLM(Launcher):
    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = safe_get_tokenizer_from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code, use_fast=False)
        model = safe_get_model_from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    atb_speed_config.init_config("config.ini")
    baichuan = BaichuanLM()
    c_t = get_precision_test_cls()(baichuan)
    c_t.run()
```

# 性能测试 [performance]

SDK提供了两种性能测试的方法,常规估计法，精确打点法。也提供了两种测试方案，2幂测试和特定case测试

## 配置说明

| 配置项key         | 默认值    | 备注                                                                                    |
|----------------|--------|---------------------------------------------------------------------------------------|
| model_name     |        | 性能测试模型名称，用于结果文件的命名                                                                    |
| batch_size     | 1      | 测试的batch size                                                                         |
| max_len_exp    | 10     | 测试的输入的最大2的幂                                                                           |
| min_len_exp    | 5      | 测试的输入的最小2的幂                                                                           |
| case_pair      |        | 特定用例测试，格式为[[seq_in,seq_out]],注意当设置这个参数时，max_len_exp min_len_exp不生效                    |
| save_file_name |        | 生成的结果文件名称，默认会自动生成，一般不设置                                                               |
| perf_mode      | normal | 性能测试方法，detail / normal , 默认是normal.要使用detail需要侵入式替换utils，并加上环境变量 RETURN_PERF_DETAIL=1 |
| skip_decode    | 0      | 性能测试时是否只测试generate而跳过decode，0/1 默认是0                                                  |

## 精确打点法

- 通过在modeling中使用sdk里的计时装饰器进行计时
- 不再需要侵入式修改任何的三方件中的源码，支持任意版本的transformers
- perf_mode设为detail
- 将环境变量`TIMEIT`设置成1来开启性能测试，为了不影响正常使用，默认是0

### Timer介绍

- 将环境变量`TIMEIT`设置成1来开计时，为了不影响正常使用，默认是0
- 计时的数据是累积的，使用 Timer.reset() 来重置计时器
- 硬件设备上的数据需要同步才能准确计时。在计时前，请使用`Timer.sync = getattr(torch, device_type).synchronize`设置计时器的同步函数

### 如何使用

只需要在最外层的forward函数上方增加timing的计时器即可。  
例如：

```python
import torch
from torch import nn

from atb_speed.common.timer import Timer


class AddNet(nn.Module):
    def __init__(self, in_dim, h_dim=5, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)

    @Timer.timing
    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    add_net = AddNet(in_dim=2)
    Timer.sync = torch.cuda.synchronize
    Timer.reset()
    for i in range(5):
        x = torch.randn(1, 1)
        y = torch.randn(1, 1)
        result = add_net.forward(x, y)
        print(result)
    print(Timer.timeit_res)
    print(Timer.timeit_res.first_token_delay)
    print(Timer.timeit_res.next_token_avg_delay)
```

## 常规估计法

- 通过第一次生成1个token，第2次生成n个token，计时作差来估计性能。
- *假设两次推理首token的时延相同*
- perf_mode设为normal

## 运行脚本

```python
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from atb_speed.common.performance.base import PerformanceTest
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_model_from_pretrained


class LMLauncher(Launcher):
    """
    Baichuan2_7B_NPU
    """

    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = safe_get_tokenizer_from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code, use_fast=False)
        model = safe_get_model_from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer


if __name__ == '__main__':
    atb_speed_config.init_config("config.ini")
    performance_test = PerformanceTest(LMLauncher())
    performance_test.warm_up()
    performance_test.run_test()
```