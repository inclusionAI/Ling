# README
此部分为qwen2-audio多音频测试的使用说明。


## 路径变量解释

| 变量名      | 含义                                                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| working_dir | 加速库及模型库下载后放置的目录                                                                                                                               |
| llm_path    | 模型仓所在路径。若使用编译好的包，则路径为 `${working_dir}/MindIE-LLM/`；若使用gitee下载的代码，则路径为 `${working_dir}/MindIE-LLM/examples/atb_models` |
| script_path | 脚本所在路径；qwen2_audio的工作脚本所在路径为 `${llm_path}/examples/models/qwen2_audio`                                                                                          |
| weight_path | 模型权重路径                                                                     |
| audio_path  | 音频所在路径                                                                     |
| max_batch_size  | 最大bacth数                                                                  |
| max_input_length  | 多模态模型的最大embedding长度，                                             |
| max_output_length | 生成的最大token数                                                          |

**数据集下载**

- https://huggingface.co/datasets/maoxx241/meld_subset


**权重下载**

- [Qwen2-Audio](https://modelscope.cn/models/Qwen/Qwen2-Audio-7B-Instruct)


**基础环境变量**

- Python其他第三方库依赖，参考[requirements_qwen2_audio.txt](../../../requirements/models/requirements_qwen2_audio.txt)
- 参考[此README文件](../../../README.md)
- qwen2_audio需要再安装editdistance三方件 pip install editdistance==0.8.1


## 推理


**NPU纯模型推理**

- 运行启动脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    bash examples/models/qwen2_audio/run_pa.sh  --multiaudio ${weight_path} ${audio_path}
    ```
- 环境变量说明
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 设置时端口建议范围为：20000-20050

  以下环境变量可在run_pa_videobench.sh脚本内修改
  - `export ASCEND_RT_VISIBLE_DEVICES=0`
    - 指定当前机器上可用的NPU

  - 其他参数
    - ${weight_path}：下载的权重路径，如：/Qwen2-Audio-7B-Instruct
    - ${audio_path}：音频数据集路径


执行样例：
  ```shell
    bash examples/models/qwen2_audio/run_pa.sh --multiaudio /data/Qwen2-Audio-7B-Instruct /data/meld_dataset
  ```
运行结束后，在${script_path}/precision/目录下会生成结果文件qwen2_audio_npu_test_pure.csv。


**NPU服务化推理设置**

- Python第三方库依赖安装完毕之后，在测试服务化之前，需要在\${script_path}目录下执行以下指令
    ```shell
    python librosa_file_util.py --modify_librosa True
    ```
用于修改librosa，保证服务化能正常推理。
- 在所有服务化推理结束之后，可在\${script_path}目录下执行以下指令
    ```shell
    python librosa_file_util.py --restore_librosa True
    ```
将librosa改回至原文件状态。
- 执行以上指令，如果用户权限不够，请切换至root用户执行。

**NPU服务化推理**
- 在拉起服务化之后，运行测试脚本
  - 在\${llm_path}目录下执行以下指令
    ```shell
    python examples/models/qwen2_audio/precision/multi_audio_test_service.py  --port ${port} --req_method ${req_method} --audio_path ${audio_path} --save_path ${save_path}
    ```
   - 参数说明
    - ${port}：拉起服务化时设置的端口号
    - ${req_method}：接口请求方式
    - ${audio_path}音频数据集路径
    - ${save_path}：结果保存路径

执行样例：
  ```shell
    python examples/models/qwen2_audio/precision/multi_audio_test_service.sh --port 3025 --req_method VLLM --audio_path data/meld_dataset --save_path examples/models/qwen2_audio/precision/qwen2_audio_npu_test_service_VLLM.csv
  ```
运行结束后，结果保存为${save_path}文件。


**精度测试**

  得到上述csv文件后，进行精度测试，执行以下命令：


    ```shell
    python examples/models/qwen2_audio/precision/multiaudio_acc.py --first_answer_path ${first_answer_path} --second_answer_path ${second_answer_path}
    ```
  - 参数说明
    - ${first_answer_path}：生成的多音频回答的结果文件
    - ${second_answer_path}：需要与first_answer_path作精度对比的结果文件
  
  执行样例：
  ```shell
    python examples/models/qwen2_audio/precision/multiaudio_acc.py --first_answer_path examples/models/qwen2_audio/precision/qwen2_audio_npu_test_service_VLLM.csv --second_answer_path examples/models/qwen2_audio/precision/qwen2_audio_npu_test_gpu.csv
  ```
运行结束后，打印出结果的精度。



