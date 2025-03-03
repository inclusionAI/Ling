# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os

from dataclasses import dataclass
import ipaddress

from . import file_utils


@dataclass
class EnvVar:
    """
    环境变量
    """
    # ATB_LLM日志级别
    atb_llm_log_level: str = os.getenv("MINDIE_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))
    # ATB_LLM日志文件存储路径
    atb_llm_log_path: str = os.getenv("MINDIE_LOG_PATH", os.getenv("LOG_TO_FILE", ""))
    # ATB_LLM日志开关
    atb_llm_log_to_file: str = str(os.getenv("MINDIE_LOG_TO_FILE", 1))
    # ATB_LLM打印开关
    atb_llm_log_to_stdout:str = "1"
    # ATB_LLM日志文件最大容量
    atb_llm_log_maxsize: int = int(os.getenv("PYTHON_LOG_MAXSIZE", "1073741824")) # 1GB
    # ATB_LLM日志可选内容
    atb_llm_log_verbose: str = str(os.getenv("MINDIE_LOG_VERBOSE", 1))

    # 模型运行时动态申请现存池大小（单位：GB）
    reserved_memory_gb: int = int(os.getenv("RESERVED_MEMORY_GB", "3"))
    # 使用哪些卡
    visible_devices: str = os.getenv("ASCEND_RT_VISIBLE_DEVICES", None)
    # 是否绑核
    bind_cpu: bool = os.getenv("BIND_CPU", "1") == "1"
    # 是否清除generation后处理参数
    remove_generation_config_dict: bool = os.getenv("REMOVE_GENERATION_CONFIG_DICT", "0") == "1"

    cpu_binding_num: int | None = os.getenv("CPU_BINDING_NUM", None)

    memory_fraction = float(os.getenv("NPU_MEMORY_FRACTION", "1.0"))

    lcoc_enable: bool = os.getenv("ATB_LLM_LCOC_ENABLE", "1") == "1"

    compress_head_enable = os.getenv("ATB_LLM_RAZOR_ATTENTION_ENABLE", "0") == "1"
    compress_head_rope = os.getenv("ATB_LLM_RAZOR_ATTENTION_ROPE", "0") == "1"

    profiling_level = os.getenv("PROFILING_LEVEL", "Level0")
    profiling_enable: bool = os.getenv("ATB_PROFILING_ENABLE", "0") == "1"
    profiling_filepath = os.getenv("PROFILING_FILEPATH", os.path.join(os.getcwd(), "profiling"))

    benchmark_enable: bool = os.getenv("ATB_LLM_BENCHMARK_ENABLE", "0") == "1"
    benchmark_filepath = os.getenv("ATB_LLM_BENCHMARK_FILEPATH", None)

    logits_save_enable: bool = os.getenv("ATB_LLM_LOGITS_SAVE_ENABLE", "0") == "1"
    logits_save_folder = os.getenv("ATB_LLM_LOGITS_SAVE_FOLDER", './')

    token_ids_save_enable: bool = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_ENABLE", "0") == "1"
    token_ids_save_folder = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_FOLDER", './')

    modeltest_dataset_specified = os.getenv("MODELTEST_DATASET_SPECIFIED", None)

    hccl_enable = os.getenv("ATB_LLM_HCCL_ENABLE", "0") == "1"
    npu_vm_support_hccs = os.getenv("NPU_VM_SUPPORT_HCCS", "0") == "1"

    auto_transpose_enable: bool = os.getenv("ATB_LLM_ENABLE_AUTO_TRANSPOSE", "1") == "1"

    atb_speed_home_path: str = os.getenv("ATB_SPEED_HOME_PATH", None)
    ailbi_mask_enable: bool = os.getenv("IS_ALIBI_MASK_FREE", "0") == "1"
    python_log_maxsize: int = int(os.getenv('PYTHON_LOG_MAXSIZE', "1073741824"))
    long_seq_enable: bool = os.getenv("LONG_SEQ_ENABLE", "0") == "1"
    time_it: bool = os.getenv("TIME_IT", "0") == "1"

    rank: int = int(os.getenv("RANK", "0"))
    local_rank: int = int(os.getenv("LOCAL_RANK", "0"))
    world_size: int = int(os.getenv("WORLD_SIZE", "1"))
    rank_table_file: str = os.getenv("RANKTABLEFILE", "")

    matmul_nd_nz_enable: bool = os.getenv("MATMUL_ND_NZ_ENABLE", "0") == "1"

    def __post_init__(self):
        # 校验
        if self.atb_llm_log_maxsize < 0 or self.atb_llm_log_maxsize > 2147483648:   # 2GB
            raise ValueError("PYTHON_LOG_MAXSIZE should not be a number in the range of 0 to 2147483648.")

        if self.reserved_memory_gb >= 64 or self.reserved_memory_gb < 0:
            raise ValueError("RESERVED_MEMORY_GB should be in the range of 0 to 64, 64 is not inclusive.")

        if self.visible_devices is not None:
            try:
                self.visible_devices = list(map(int, self.visible_devices.split(',')))
            except ValueError as e:
                raise ValueError("ASCEND_RT_VISIBLE_DEVICES should be in format "
                                 "{device_id},{device_id},...,{device_id}") from e

        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError("NPU_MEMORY_FRACTION should be in the range of 0 to 1.0, 0.0 is not inclusive.")

        if self.atb_speed_home_path is not None: 
            self.atb_speed_home_path = file_utils.standardize_path(self.atb_speed_home_path)
            file_utils.check_path_permission(self.atb_speed_home_path)

        if self.world_size <= 0 or self.world_size > 1048576:
            raise ValueError("WORLD_SIZE should not be a number in the range of 0 to 1048576, 0 is not inclusive.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError("LOCAL_RANK should be in the range of 0 to WORLD_SIZE, WORLD_SIZE is not inclusive.")
            
        if not isinstance(self.python_log_maxsize, int) or self.python_log_maxsize < 0 or \
            self.python_log_maxsize > 2147483648:   # 2GB
            raise ValueError("PYTHON_LOG_MAXSIZE should be an int type variable, " \
                "and its value should be in the range of 0 to 2147483648.")
        if self.cpu_binding_num is not None:
            if not isinstance(self.cpu_binding_num, int):
                raise ValueError("CPU_BIDING_NUM should be an int type variable or None")
        
        if self.profiling_enable and not os.path.exists(self.profiling_filepath):
            os.makedirs(self.profiling_filepath, exist_ok=True)
        self.profiling_filepath = file_utils.standardize_path(self.profiling_filepath)
        file_utils.check_file_safety(self.profiling_filepath, 'w')
        self.check_ranktable()

        if os.getenv("INF_NAN_MODE_ENABLE", "1") == "0":
            os.environ['INF_NAN_MODE_FORCE_DISABLE'] = "1"

    @staticmethod
    def is_valid_ip(ip):
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    def dict(self):
        return self.__dict__

    def update(self):
        self.logits_save_enable = os.getenv("ATB_LLM_LOGITS_SAVE_ENABLE", "0") == "1"
        self.logits_save_folder = os.getenv("ATB_LLM_LOGITS_SAVE_FOLDER", './')
        self.token_ids_save_enable = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_ENABLE", "0") == "1"
        self.token_ids_save_folder = os.getenv("ATB_LLM_TOKEN_IDS_SAVE_FOLDER", './')
        self.modeltest_dataset_specified = os.getenv("MODELTEST_DATASET_SPECIFIED", None)

    def check_ranktable(self):
        if self.rank_table_file:
            with file_utils.safe_open(self.rank_table_file, 'r', encoding='utf-8') as device_file:
                ranktable = json.load(device_file)
            
            world_size = 0
            server_list = ranktable["server_list"]
            for server in server_list:
                server_devices = server["device"]
                world_size += len(server_devices)
            for server in server_list:
                server_devices = server["device"]
                for device in server_devices:
                    if int(device["rank_id"]) < world_size:
                        continue
                    else:
                        raise ValueError("rank_id should be a number less than world size.")
            
            for server in server_list:
                server_devices = server["device"]
                for device in server_devices:
                    if self.is_valid_ip(device["device_ip"]):
                        continue
                    else:
                        raise ValueError("device_ip is invalid.")
            
            for server in server_list:
                if self.is_valid_ip(server["server_id"]):
                    continue
                else:
                    raise ValueError("server_id is invalid.")


ENV = EnvVar()
