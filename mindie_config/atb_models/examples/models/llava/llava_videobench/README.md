# README

- [Video-Bench](https://github.com/PKU-YuanGroup/Video-Bench/tree/main) 是一个旨在帮助视频分析领域的开发人员和用户的平台，用于系统地评估视频理解模型在各种能力上的性能，包括基于先验知识的QA、理解决策视频专属理解等。
- 此代码仓中实现了基于NPU硬件的LLaVa推理模型在Video-Bench上的支持。

# 使用说明

**权重下载**

- [LLava-1.6-video-7B](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/tree/main)
- [LLaVA-NeXT-Video-7B-32K-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-32K-hf/tree/main)
- [LLaVA-NeXT-Video-7B-DPO-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-DPO-hf/tree/main)
- [LLaVA-NeXT-Video-34B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-hf/tree/main)
- [LLaVA-NeXT-Video-34B-DPO-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-34B-DPO-hf/tree/main)

**数据集下载**

- https://huggingface.co/datasets/LanguageBind/Video-Bench

达到的效果如下所示，Eval_video目录下包含了多个子数据集：

```
egs/VideoBench/
├── Eval_video
│   └── ActivityNet
│       └── mp4等文件
│   └── Driving-decision-making
│       └── mp4等文件
|    ...
```


**文本下载**

- https://github.com/PKU-YuanGroup/Video-Bench/tree/main?tab=readme-ov-file

达到的效果如下所示：

```
egs/Video-Bench/
├── Eval_QA
│   └── QA等json文件
|    ...
```

**所有问题的正确答案下载**

- https://huggingface.co/spaces/LanguageBind/Video-Bench/resolve/main/file/ANSWER.json

下载后置于egs/Video-Bench目录下


**基础环境变量**

-1.Python其他第三方库依赖，参考[requirements_llava.txt](../../../requirements/models/requirements_llava.txt)
-2.参考[此README文件](../../../README.md)
-注意：保证先后顺序，否则llava的其余三方依赖会重新安装torch，导致出现别的错误
 llava-next-video需要再安装av三方件 pip install av,pip install imagesize且transformers版本大于等于4.42.0


## 推理

**数据集采样**

- 运行启动脚本，该脚本用于对整个video-bench数据集进行采样，得到1000组数据，以减少数据集的大小，从而加快推理速度。
  - 在\MindIE-LLM目录下执行以下指令
    ```shell
    cd examples/atb_models
    python examples/models/llava/llava_videobench/extract_dataset.py --Eval_QA_root ${Eval_QA_root} --Eval_Video_root ${Eval_Video_root} --sampling_output_folder ${sampling_output_folder} --json_output_path ${json_output_path} --correct_answer_file ${correct_answer_file} --nums ${nums}
    ```
  - 参数说明
    - ${Eval_QA_root}：文本路径，如：egs/Video-Bench/
    - ${Eval_Video_root}：数据集所在的路径，如：egs/VideoBench/
    - ${sampling_output_folder}：采样结果输出路径，用来保存采样后的视频文件，如：egs/VideoBench/Eval_video
    - ${json_output_path}：采样结果文件输出路径，如：egs/Video-Bench/Eval_QA，用来保存采样后的QA文件(sampling_datasets_QA_new.json)和正确答案(sampling_answer.json)
    - ${correct_answer_file}：正确答案文件路径，如：egs/Video-Bench/ANSWER.json
    - ${nums}：采样数量，默认为1000


  文件执行后，会在sampling_output_folder目录下生成采样后的数据集文件，并在${json_output_path}目录下生成采样的QA文件(sampling_datasets_QA_new.json)和答案文件(sampling_answer.json)。

**NPU推理**

- 运行启动脚本
  - 在\MindIE-LLM\examples\atb_models目录下执行以下指令
    ```shell
    bash examples/models/llava/llava_videobench/run_pa_videobench.sh  ${weight_path} ${video_path} ${eval_qa_root} ${chat_conversation_output_folder}
    ```
- 环境变量说明
  - `export MASTER_PORT=20030`
    - 设置卡间通信端口
    - 设置时端口建议范围为：20000-20050

  以下环境变量可在run_pa_videobench.sh脚本内修改
  - `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
    - 指定当前机器上可用的NPU
  - 以下环境变量与性能和内存优化相关，已在通常情况下无需修改
    ```shell
    export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1
    export LCCL_ENABLE_FALLBACK=1
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
    export ATB_CONTEXT_WORKSPACE_SIZE=0
    ```
  - 其他参数
    - ${weight_path}：下载的权重路径，如：/LLaVA-NeXT-Video-7B-hf
    - ${video_path}：视频数据集路径,要精确到子数据集所在路径，如：egs/VideoBench/Eval_video/sampling_data
    - ${eval_qa_root}：文本路径，仅需精确到文本路径根目录即可，如：egs/Video-Bench/
    - ${chat_conversation_output_folder}：对话结果输出路径，用于保存NPU的运行结果，如：./Chat_results

执行样例如：
  ```shell
    bash examples/models/llava/llava_videobench/run_pa_videobench.sh /LLaVA-NeXT-Video-7B-hf egs/VideoBench/Eval_video/Driving_exam egs/Video-Bench/ ./Chat_results
  ```
运行结束后，在chat_conversation_output_folder目录下会生成对话结果文件。


**GPU推理**

- 运行启动脚本，在GPU上进行推理测试
  - 执行以下指令
    ```shell
    bash examples/models/llava/llava_videobench/run_pa_videobench_gpu.sh ${dataset_name} ${Eval_QA_root} ${Eval_Video_root} ${weight_path} ${chat_conversation_output_folder}
    ```
  可修改脚本run_pa_videobench_gpu.sh中代码export CUDA_VISIBLE_DEVICES='1,2'指定所用的GPU
  - 参数说明
    - ${dataset_name}：数据集名称，指的是子数据集名，如sampling_data
    - ${Eval_QA_root}：文本路径，如：egs/Video-Bench/
    - ${Eval_Video_root}：数据集根路径，如：egs/VideoBench/
    - ${weight_path}：权重路径，如：/LLaVA-NeXT-Video-7B-hf
    - ${chat_conversation_output_folder}：对话结果输出路径，用于保存GPU的运行结果，如：./Chat_results

  运行结束后，在${chat_conversation_output_folder}目录下会生成gpu上对话结果文件。


**精度测试**

  得到上述NPU和GPU结果的json文件后，对两者精度进行测试，执行以下命令：

    ```shell
    python examples/models/llava/llava_videobench/llava_video_precision_test.py --eval_npu_path ${eval_npu_path} --eval_gpu_path ${eval_gpu_path} --sampling_answer_path ${sampling_answer_path}
    ```
  - 参数说明
    - ${eval_npu_path}：生成的npu上对话结果文件
    - ${eval_gpu_path}：生成的gpu上对话结果文件
    - ${sampling_answer_path}：正确答案文件路径，如数据采样一节中生成的sampling_answer.json
  
  打印出结果的精度



