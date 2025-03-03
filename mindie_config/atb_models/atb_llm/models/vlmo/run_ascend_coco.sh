#!/bin/bash
# export LD_PRELOAD=/usr/lib64/libgomp.so.1.0.0:$LD_PRELOAD

# ATB_LOG_TO_STDOUT=1 ATB_LOG_LEVEL=INFO TASK_QUEUE_ENABLE=0 ASDOPS_LOG_TO_STDOUT=1 ASDOPS_LOG_LEVEL=INFO
#coco
python run_ascend_coco.py with "<CONFIG_NAME>"  load_path="<Finetuned_VLMo_WEIGHT>" test_only=True