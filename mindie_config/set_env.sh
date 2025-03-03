source /usr/local/Ascend/ascend-toolkit/set_env.sh 
source /usr/local/Ascend/mindie/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh
# source /mnt/llm-benchmark/keshi.sk/models/sft/zzx_src/tr6/MindIE-LLM/examples/atb_models/output/atb_models/set_env.sh
source /home/HwHiAiUser/Ascend/Ling/mindie_config/atb_models/output/atb_models/set_env.sh
export OMP_NUM_THREADS=1
export ATB_LLM_COMM_BACKEND="hccl"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ATB_LLM_HCCL_ENABLE=1
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export INF_NAN_MODE_ENABLE=0
