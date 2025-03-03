# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
import os
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
import secrets

import torch
import numpy as np
from safetensors.torch import save_file, safe_open
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from atb_llm.utils.log import logger
from atb_llm.utils.convert import _remove_duplicate_names
from atb_llm.utils import file_utils


class Encrypt(object):
    def __init__(self) -> None:
        # fill in your decryption parameters here
        self.key = None
        self.nonce = None

    def encrypt_files(
            self,
            local_weight_files: List[Path], 
            encrypted_sf_weight_files: List[Path], 
            discard_names: List[str]
    ):

        num_weight_files = len(local_weight_files)
        
        for i, (local_weight_file, encrypted_sf_weight_file) in \
            enumerate(zip(local_weight_files, encrypted_sf_weight_files)):

            blacklisted_keywords = ["arguments", "args", "training"]
            if any(substring in local_weight_file.name for substring in blacklisted_keywords):
                continue

            start_time = datetime.now(tz=timezone.utc)
            self.encrypt_file(local_weight_file, encrypted_sf_weight_file, discard_names)
            elapsed_time = datetime.now(tz=timezone.utc) - start_time
            try:
                logger.info(f"Encrypt : [{i + 1}/{num_weight_files}] -- Took: {elapsed_time}")
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e

    def encrypt_file(
            self,
            local_weight_file: Path, 
            encrypted_sf_weight_file: Path, 
            discard_names: List[str]
    ):
        metadata = {"format": "pt"}
        suffix = local_weight_file.suffix
        local_weight_file = file_utils.standardize_path(str(local_weight_file), check_link=False)
        file_utils.check_file_safety(local_weight_file, 'r', is_check_file_size=False)
        if suffix == '.bin':
            loaded_state_dict = torch.load(local_weight_file, map_location="cpu")
            if "state_dict" in loaded_state_dict:
                loaded_state_dict = loaded_state_dict["state_dict"]
            to_remove_dict = _remove_duplicate_names(loaded_state_dict, discard_names=discard_names)

            for kept_name, to_remove_list in to_remove_dict.items():
                for to_remove in to_remove_list:
                    if to_remove not in metadata:
                        metadata[to_remove] = kept_name
                    del loaded_state_dict[to_remove]
            loaded_state_dict = {k: v.contiguous() for k, v in loaded_state_dict.items()}
        elif suffix == '.safetensors':
            loaded_state_dict = safe_open(local_weight_file, framework='pt', device='cpu')
        else:
            raise NotImplementedError

        # conduct encrypting processing
        encrypted_state_dict, dtype_metadata = self.encrypt_state_dict(loaded_state_dict)
        metadata.update(dtype_metadata)

        # save encrypted weight to safetensors
        os.makedirs(os.path.dirname(encrypted_sf_weight_file), exist_ok=True)
        encrypted_sf_weight_file = file_utils.standardize_path(str(encrypted_sf_weight_file), check_link=False)
        file_utils.check_file_safety(encrypted_sf_weight_file, 'w', is_check_file_size=False)
        save_file(encrypted_state_dict, encrypted_sf_weight_file, metadata=metadata)

    def encrypt_state_dict(
            self,
            state_dict: Dict[str, torch.Tensor]
    ):
        encrypted_state_dict = {}
        metadata = {}
        for k in state_dict.keys():
            try:
                v_tensor = state_dict[k]
            except TypeError:
                v_tensor = state_dict.get_tensor(k)
            metadata[k] = str(v_tensor.dtype)
            if v_tensor.dtype == torch.bfloat16:
                encrypted_tensor = self.encrypt(v_tensor.to(torch.float32))
            else:
                encrypted_tensor = self.encrypt(v_tensor)
            encrypted_state_dict[k] = encrypted_tensor
        return encrypted_state_dict, metadata

    @abstractmethod
    def encrypt(self, tensor: torch.Tensor):
        # implement your encrypting algorithm here
        return tensor


class EncryptTools(Encrypt):
    '''
    写一个自己的加密类，以AES-256，CTR模式 加密算法为例。
    其中的密钥路径只是为了演示需要，私密信息路径请自行填写
    '''
    def __init__(self, key_path) -> None:
        super().__init__()

        # generate key and nonce
        self.key = secrets.token_bytes(32)  # AES-256
        self.nonce = secrets.token_bytes(16)  # 64-bit nonce

        # save key and nonce
        with file_utils.safe_open(os.path.join(key_path, 'aes_key.key'), 'wb') as f:
            f.write(self.key)
        with file_utils.safe_open(os.path.join(key_path, 'aes_nonce.nonce'), 'wb') as f:
            f.write(self.nonce)


    def encrypt(self, tensor: torch.Tensor):
        """
        输入是原始tensor，输出是加密后的tensor。
        保证加密前后，encrypted_tensor和tensor的shape一致。
        """
        # 此处的定义的加密函数只是一个使用范例，可以根据自己的需求改成自己的加密函数。
        # 创建一个 AES 计数器模式的 cipher对象
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(self.nonce), backend=default_backend())

        # 创建一个加密器对象
        encryptor = cipher.encryptor()

        # 加密密文
        tensor_numpy = tensor.numpy()
        encrypted_numpy_bytes = encryptor.update(tensor_numpy.tobytes()) + encryptor.finalize()

        # 转化为tesor类型
        encrypted_numpy = np.frombuffer(encrypted_numpy_bytes, dtype=tensor_numpy.dtype)
        encrypted_tensor = torch.from_numpy(encrypted_numpy).view(tensor.shape)

        return encrypted_tensor


