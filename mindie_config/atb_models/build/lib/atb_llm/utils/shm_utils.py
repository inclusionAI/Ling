# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
from multiprocessing import shared_memory
from multiprocessing.shared_memory import _SHM_NAME_PREFIX
import torch
import numpy as np

from atb_llm.utils import file_utils

if _SHM_NAME_PREFIX.startswith("/"):
    SHM_NAME_PREFIX = _SHM_NAME_PREFIX[1:]
else:
    SHM_NAME_PREFIX = _SHM_NAME_PREFIX
SHM_NAME_MAX_LENGTH = 14

MAX_SHM_SIZE = 10**9  # 假设最大 1GB，可以根据需求调整


def create_shm(size, shm_name_save_path):
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    if size > MAX_SHM_SIZE:
        raise ValueError(f"Size exceeds the maximum allowed limit of {MAX_SHM_SIZE} bytes.")

    try:
        shm = shared_memory.SharedMemory(create=True, size=size)
        with file_utils.safe_open(shm_name_save_path, 'a', is_exist_ok=True) as fw:
            fw.write(f"{shm.name}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to create shared memory: {e}") from e
    
    return shm


def release_shared_memory(file_path):
    if not os.path.exists(file_path):
        raise FileExistsError(f"{file_path} does not exists.")
    try:
        with file_utils.safe_open(file_path, "r", is_exist_ok=True) as file:
            shm_names = [line.strip() for line in file.readlines()]
        for name in shm_names:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()  
            shm.unlink() 

    except Exception as e:
        raise RuntimeError(f"release share memory error: {e}") from e


def encode_shm_name_to_int64(name):
    """
    Encodes a shared memory name into a 64-bit integer.

    Parameters:
        name (str): The shared memory name, limited to a maximum length of 14 characters.
    Returns:
        int: The encoded 64-bit integer, represented as a negative value.
    Raises:
        ValueError: If the name length exceeds 14 characters or the encoded value exceeds 64 bits.
    Encoding Logic:
        - The first 5 bits represent the length of the name.
        - Subsequent 5 bits represent the index of each character in the name.
        - The last 3 bits are used for checksum based on the sum of character indices.
    """

    if len(name) > SHM_NAME_MAX_LENGTH:
        raise ValueError("The shared_memory name is too long, it is limited to 14.")

    CHARSET = "0123456789abcdefpsmnwABCDEF"
    name = name.replace(SHM_NAME_PREFIX, "")

    name_len = len(name)
    # 58表示除符号位之后用来表示长度的5位的最低位的偏移，63 - 5
    final_value = (name_len & 0b11111) << 58

    idx_sum = 0
    # 5位表示长度，3位表示校验，其余每5位表示一个char
    for i, char in enumerate(name):
        shift = 53 - 5 * i  # 53表示除符号位、表示长度的5位之后用于表示每一位字符的最低位的偏移，63 - 5 - 5
        if char not in CHARSET:
            raise ValueError(f"Invalid character '{char}' in shared_memory name.")
        char_idx = CHARSET.index(char)
        idx_sum += char_idx
        final_value |= (char_idx & 0b11111) << shift 

    checksum = idx_sum % 16

    final_value |= (checksum & 0b111)

    if final_value.bit_length() > 64:
        raise ValueError("Encoded value exceeds 64 bits")
    final_value = -final_value
    return final_value


def encode_shape_to_int64(shape):
    """
    Encodes a shape into a 64-bit integer.

    Parameters:
        shape (tuple): A tuple containing 2 to 5 dimensions.
    Returns:
        int: The encoded 64-bit integer, represented as a negative value.
    Raises:
        ValueError: If the number of dimensions is not between 3 and 5 or if any dimension size exceeds limits.
    Encoding Logic:
        - The first 3 bits represent the number of dimensions.
        - Each dimension size is encoded using a specific number of bits.
        - The last 3 bits are used for checksum, based on the number of 1s in the encoded value.
    """

    split_plan_set = {
        2:[25, 15],  # [N * D]
        3:[18, 18, 18],  # [N * D * S]
        4:[14, 3, 14, 14], # [N * C * H * W ]
        5:[13, 14, 3, 14, 14]  # [N * L * C * H * W]
    }

    num_dims = len(shape)
    if not (2 <= num_dims <= 5):
        raise ValueError("Shape must have 3 to 5 dimensions")
    
    shift = 60
    encoded_value = (num_dims & 0b111) << shift 
    split_plan = split_plan_set.get(num_dims)
    for i, dim in enumerate(shape):
        max_dimension_size = 1 << split_plan[i] - 1
        if dim > max_dimension_size:
            raise ValueError(f"Dimension {dim} is too large, max allowed is {max_dimension_size}")
        shift -= split_plan[i]
        encoded_value |= (dim & ((1 << split_plan[i]) - 1)) << shift

    checksum = bin(encoded_value).count('1') % 3
    
    final_value = encoded_value | checksum
    final_value = -final_value
    return final_value


def decode_shm_name_from_int64(encoded_value):
    """
    Decodes a 64-bit integer back into a shared memory name.

    Parameters:
        encoded_value (int or torch.Tensor): The encoded 64-bit integer representing the shared memory name.
    Returns:
        str: The decoded shared memory name.
    Raises:
        ValueError: If the checksum verification fails.
    Decoding Logic:
        - Negate the encoded value to retrieve the original representation.
        - Extract the length of the name from the highest 5 bits.
        - Decode each character using the specified character set.
        - Validate the checksum to ensure data integrity.
    """

    if isinstance(encoded_value, torch.Tensor):  
        encoded_value = encoded_value.item()
    encoded_value = -encoded_value

    CHARSET = "0123456789abcdefpsmnwABCDEF"

    name_len = (encoded_value >> 58) & 0b11111

    decode_name = SHM_NAME_PREFIX
    idx_sum = 0
    for i in range(name_len):
        shift = 53 - 5 * i
        char_idx = (encoded_value >> shift) & 0b11111
        idx_sum += char_idx
        if char_idx >= len(CHARSET):
            raise ValueError("Decode shared_memory name error, invalid character index")
        decode_name += CHARSET[char_idx]

    checksum = encoded_value & 0b111
    expected_checksum = (idx_sum % 16) & 0b111

    if checksum != expected_checksum:
        raise ValueError("Checksum verification failed.")
    return decode_name


def decode_shape_from_int64(encoded_value):
    """
    Decodes a 64-bit integer back into a shape tuple.

    Parameters:
        encoded_value (int or torch.Tensor): The encoded 64-bit integer representing the shape.
    Returns:
        list: The decoded shape as a list of dimensions.
    Raises:
        ValueError: If the checksum does not match or the number of dimensions is invalid.
    Decoding Logic:
        - Negate the encoded value to retrieve the original representation.
        - Validate the checksum to ensure data integrity.
        - Extract the number of dimensions from the highest 3 bits.
        - Decode each dimension size using the specified split plan.
    """
     
    if isinstance(encoded_value, torch.Tensor):  
        encoded_value = encoded_value.item()
    encoded_value = -encoded_value

    checksum = encoded_value & 0b111

    calculated_checksum = bin(encoded_value >> 3).count('1') % 3
    if checksum != calculated_checksum:
        raise ValueError("Checksum does not match, the encoded value might be corrupted")

    # 提取维度数量（最高3位）
    shift = 60
    num_dims = (encoded_value >> shift) & 0b111
    if not (2 <= num_dims <= 5):
        raise ValueError(f"Invalid number of dimensions in encoded value: {num_dims}")

    split_plan_set = {
        2:[25, 15],  # [N * D]
        3:[18, 18, 18],  # [N * D * S]
        4:[14, 3, 14, 14], # [N * C * H * W ]
        5:[13, 14, 3, 14, 14]  # [N * L * C * H * W]
    }
    split_plan = split_plan_set.get(num_dims)

    shape = []
    for split_size in split_plan:
        shift -= split_size
        dim = (encoded_value >> shift) & ((1 << split_size) - 1)
        shape.append(dim)

    return shape


def get_data_from_shm(shm_name_value, shape_value, dtype=np.float32, device="cpu"):
    '''
    Get the numpy data which stored in sharememory.

    Parameters:
        shm_name_value: str, sharememory name.
        shape_value: list || tuple, the shape of data which stored in sharememory.
        dtype: numpy.dtype, the dtype of data which stored in sharememory.
        device: str, the device of data which stored in sharememory.
    Returns:
        torch.tensor
    '''
    pixel_values = None
    try:
        shm_name = decode_shm_name_from_int64(shm_name_value)
        shape = decode_shape_from_int64(shape_value)
        shm = shared_memory.SharedMemory(name=shm_name)
        pixel_values = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        pixel_values = torch.from_numpy(pixel_values).to(device)
    except Exception as e:
        raise ValueError(f"Get data from share memory error: {e}") from e
    return pixel_values