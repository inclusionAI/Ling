# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
import os
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
import importlib

import torch
import numpy as np
from safetensors.torch import save_file, safe_open
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from atb_llm.utils.log import logger
from atb_llm.utils import file_utils


class Decrypt(object):
    '''
    本解密类只是一个参考
    '''
    def __init__(self) -> None:
        # fill in your decryption parameters here
        self.key = None 
        self.nonce = None

    def decrypt_state_dict(self, encrypted_state_dict: Dict[str, torch.Tensor]):
        decrypted_state_dict = {}
        metadata = encrypted_state_dict.metadata()
        for k in encrypted_state_dict.keys():
            v_tensor = encrypted_state_dict.get_tensor(k)
            decrypted_tensor = self.decrypt(v_tensor)

            module_name, attribute_name = metadata[k].split('.')
            module = importlib.import_module(module_name)
            dtype_ = getattr(module, attribute_name)
            decrypted_state_dict[k] = decrypted_tensor.view(v_tensor.shape).to(dtype_)
        return decrypted_state_dict

    def decrypt_file(self, local_weight_file: Path, decrypted_sf_weight_file: Path):
        metadata = {"format": "pt"}
        local_weight_file = file_utils.standardize_path(str(local_weight_file), check_link=False)
        file_utils.check_file_safety(local_weight_file, 'r', is_check_file_size=False)
        loaded_state_dict = safe_open(local_weight_file, framework='pt', device='cpu')

        # conduct encrypting processing
        decrypted_state_dict = self.decrypt_state_dict(loaded_state_dict)

        # save encrypted weight to safetensors
        os.makedirs(os.path.dirname(decrypted_sf_weight_file), exist_ok=True)
        decrypted_sf_weight_file = file_utils.standardize_path(str(decrypted_sf_weight_file), check_link=False)
        file_utils.check_file_safety(decrypted_sf_weight_file, 'w', is_check_file_size=False)
        save_file(decrypted_state_dict, decrypted_sf_weight_file, metadata=metadata)

    def decrypt_files(self, encrypted_sf_weight_files: List[Path], decrypted_sf_weight_files: List[Path]):
        num_weight_files = len(encrypted_sf_weight_files)
        for i, (encrypted_sf_weight_file, decrypted_sf_weight_file) in \
            enumerate(zip(encrypted_sf_weight_files, decrypted_sf_weight_files)):

            start_time = datetime.now(tz=timezone.utc)
            self.decrypt_file(encrypted_sf_weight_file, decrypted_sf_weight_file)
            elapsed_time = datetime.now(tz=timezone.utc) - start_time
            try:
                logger.info(f"Decrypt : [{i + 1}/{num_weight_files}] -- Took: {elapsed_time}")
            except ZeroDivisionError as e:
                raise ZeroDivisionError from e

    @abstractmethod
    def decrypt(self, encrypted_tensor: torch.Tensor):
        # implement your decrypting algorithm here
        return encrypted_tensor


class DecryptTools(Decrypt):
    '''
    写一个自己的解密类，以AES-256，CTR模式 加密算法为例。
    '''
    def __init__(self, key_path, **kwargs) -> None:
        super().__init__()

        ## gain key and nonce
        with file_utils.safe_open(os.path.join(key_path, 'aes_key.key'), 'rb') as f:
            self.key = f.read()
        with file_utils.safe_open(os.path.join(key_path, 'aes_nonce.nonce'), 'rb') as f:
            self.nonce = f.read()

    def decrypt(self, encrypted_tensor: torch.Tensor):
        """
        输入是加密tensor，输出是解密tensor。 
        保证解密前后，encrypted_tensor和decrypted_tensor的shape一致。
        """
        # 此处的定义的解密函数只是一个使用范例，可以根据自己的需求改成自己的解密函数。
        # 创建一个 AES 计数器模式的 cipher对象
        cipher = Cipher(algorithms.AES(self.key), modes.CTR(self.nonce), backend=default_backend())

        # 创建一个解密器对象
        decryptor = cipher.decryptor()

        # 解密密文
        encrypted_numpy = encrypted_tensor.numpy()
        decrypted_numpy_bytes = decryptor.update(encrypted_numpy) + decryptor.finalize()

        # 转化为tesor类型
        decrypted_numpy = np.frombuffer(decrypted_numpy_bytes, dtype=encrypted_numpy.dtype)
        decrypted_tensor = torch.from_numpy(decrypted_numpy).view(encrypted_tensor.shape)

        return decrypted_tensor
