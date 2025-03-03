# Telechat README

星辰语义大模型TeleChat是由中国电信人工智能科技有限公司研发训练的大语言模型，采用1.5万亿 Tokens中英文高质量语料进行训练。
     
- 参考实现：
  ```
  https://github.com/Tele-AI/Telechat
  ```

# 特性矩阵
- 此矩阵罗列了TeleChat模型支持的特性

| 模型及参数量 | 800I A2 Tensor Parallelism | 300I DUO Tensor Parallelism | FP16 | BF16 | Flash Attention | Paged Attention | W8A8量化 | W8A16量化 | KV cache量化 | 稀疏量化 | MOE量化 | MindIE Service | TGI |  长序列 |
|-------------|----------------------------|-----------------------------|------|----------------------|-----------------|-----------------|---------|-----------|--------------|--------------------------|-----|--------|-----|-----|
| Telechat-7B    | 否     | 支持world size 1,2,4,8           | 是   | 否                   | 否              | 是              | 是       | 否        | 否           | 否                       | 否  | 否     | 否  | 否  |
| Telechat-12B-v2    | 否     | 支持world size 1,2,4,8           | 是   | 否                   | 否              | 是              | 否       | 否        | 否           | 是                       | 否  | 否     | 否  | 否  |

# 使用说明

## 路径变量解释
| 变量名  | 含义                                             |
|--------|--------------------------------------------------|
| working_dir | 加速库及模型库下载后放置的目录                  |
| llm_path | 模型仓所在路径。若使用编译好的包，则路径为`${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为`${working_dir}/MindIE-LLM/examples/atb_models`    |
| script_path | 脚本所在路径；telechat的工作脚本所在路径为${llm_path}/examples/models/telechat                          |
| weight_path | 模型权重路径 |

## 权重下载
- [Telechat-7B](https://huggingface.co/Tele-AI/Telechat-7B/tree/main)
- [Telechat-12B-v2](https://modelscope.cn/models/TeleAI/TeleChat-12B-v2/files)

## 权重转换
- 参考[此README文件](../../README.md)

## 量化权重转换（W8A8）
在`llm_path`目录下执行以下命令行
- 新增可选参数`trust_remote_code` 代表是否信任本地的可执行文件: 默认不执行，传入此参数，则信任本地可执行文件。
``` bash
python examples/models/telechat/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8量化权重路径} --w_bit 8 --a_bit 8 --disable_level L0 --device_type cpu --anti_method m2 --act_method 3 --calib_file ${llm_path}/examples/models/telechat/boolq.jsonl --trust_remote_code
```

## 稀疏量化权重转换（W8A8SC）
- 新增可选参数`trust_remote_code` 代表是否信任本地的可执行文件: 默认不执行，传入此参数，则信任本地可执行文件。
- Step 1
    ```shell
    # 设置CANN包的环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ${llm_path}
    python examples/models/telechat/convert_quant_weights.py --model_path {浮点权重路径} --save_directory {W8A8S量化权重路径} --w_bit 4 --a_bit 8 --calib_file ${llm_path}/examples/models/telechat/boolq.jsonl --fraction 0.011 --co_sparse True --trust_remote_code
    ```
    
- Step 2：量化权重切分及压缩
    > 运行前需要确保压缩工具编译过
    >
    > `cd /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/msmodelslim/pytorch/weight_compression/compress_graph`
    >
    > `bash build.sh /usr/local/Ascend/ascend-toolkit/latest`
    ```shell
    torchrun --nproc_per_node {TP数} -m examples.convert.model_slim.sparse_compressor --model_path {W8A8S量化权重路径} --save_directory {W8A8SC量化权重路径}
    ```
    - TP数为tensor parallel并行个数
    - 注意：若权重生成时以TP=4进行切分，则运行时也需以TP=4运行
    - 示例
      ```shell
      torchrun --nproc_per_node 2 -m examples.convert.model_slim.sparse_compressor --model_path /data1/weights/model_slim/telechat-12b_w8a8s --save_directory /data1/weights/model_slim/telechat-12b_w8a8sc
      ```

# 服务化推理

## 300I DUO 运行操作说明

### 对话测试
- telechat-12b-v2 由于hidden_size变大，传block_size时需要更改为96；服务化需要把config文件的第一个参数cacheBlockSize改为96.
- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash ${script_path}/run_300i_duo.sh ${weight_path} -trust_remote_code
    ```
- 环境变量说明
  - `export BIND_CPU=1`
    - 绑定CPU核心开关
    - 默认进行绑核
    - 若当前机器未设置NUMA或绑核失败，可将 BIND_CPU 设为 0
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`
    - 指定当前机器上可用的逻辑NPU核心，多个核心间使用逗号相连
    - 核心ID查阅方式见[此README文件](../../README.md)的【启动脚本相关环境变量】章节
    - 若要使用单卡双芯，请指定至少两个可见核心；若要使用双卡四芯，请指定至少四个可见核心
  - `export TP_WORLD_SIZE=2`
    - 指定模型运行时的TP数，即world size
    - 默认为单卡双芯
    - 各模型支持的TP数参考“特性矩阵”
    - “单卡双芯”运行请指定`TP_WORLD_SIZE`为`2`，“双卡四芯”运行请指定`TP_WORLD_SIZE`为`4`
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 默认使用20030端口
    - 目的是为了避免同一台机器同时运行多个多卡模型时出现通信冲突
    - 设置时端口建议范围为：20000-20050
  - `export PYTHONPATH=${llm_path}:$PYTHONPATH`
    - 将模型仓路径加入Python查询模块和包的搜索路径中
    - 将${llm_path}替换为实际路径

### 对话测试脚本参数说明
- `--model_path` 模型路径
- `--input_text` 输入问题
- `--max_input_length` 最大输入长度
- `--max_output_length` 最大输出长度
- `--max_batch_size` 每次运行时固定的batch数量
- `--trust_remote_code` 代表是否信任本地的可执行文件: 默认不执行，传入此参数，则信任本地可执行文件。
- 所有参数可见run_pa.py文件中

**运行W8A8量化**
- 获取量化权重后操作步骤同上

## 精度测试
- 参考[此README文件](../../../tests/modeltest/README.md)

## 性能测试
- 参考[此README文件](../../../tests/modeltest/README.md)
