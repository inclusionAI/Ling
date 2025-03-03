# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import os
from enum import Enum

import types
from dataclasses import dataclass
from typing import List, Callable, Any
import warnings
from PIL import Image

from atb_llm.utils import file_utils

MAX_PATH_LENGTH = 4096
MAX_IMAGE_PIXELS = 10000 * 10000
MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024
MAX_VIDEO_FILE_SIZE = 512 * 1024 * 1024
MAX_AUDIO_FILE_SIZE = 20 * 1024 * 1024

MAX_FILE_SIZE_KEY = "max_file_size"
MAX_PATH_LENGTH_KEY = "max_path_length"
CHECK_LINK_KEY = "check_link"



@dataclass
class MultimodalRequestOut:
    req_list: List
    batch: int
    image_file_list: List | None
    video_file_list: List | None
    audio_file_list: List | None
    input_texts: List | None


@dataclass
class MultimodalInput:
    input_texts: List | None
    image_path: List | None
    video_path: List | None
    audio_path: List | None


@dataclass
class RunReturns:
    req_list: List | None
    batch: int | None
    image_file_list: List | None
    video_file_list: List | None
    audio_file_list: List | None
    all_input_texts: List | None
    all_generate_text_list: List | None
    all_token_num_list: List | None
    e2e_time_all: float | None


class MultimodalSourceType(Enum):
    @classmethod
    def has_extension(cls, ext: str) -> bool:
        """
        Check if the extension is in the enum class
        :param ext: File extension
        :return: Whether the extension is in the enum class
        """
        return ext.lower() in cls._value2member_map_


class ImageType(MultimodalSourceType):
    JPG = ".jpg"
    PNG = ".png"
    JPEG = ".jpeg"
    BMP = ".bmp"


class VideoType(MultimodalSourceType):
    MP4 = ".mp4"
    WMV = ".wmv"
    AVI = ".avi"


class AudioType(MultimodalSourceType):
    MP3 = ".mp3"
    WAV = ".wav"


def is_image(file_name: str) -> bool:
    """
    Check if the file is of image type
    :param file_name: File name
    :return: Whether the file is of image type
    """
    ext = os.path.splitext(file_name)[1]
    return ImageType.has_extension(ext)


def is_video(file_name: str) -> bool:
    """
    Check if the file is of video type
    :param file_name: File name
    :return: Whether the file is of video type
    """
    ext = os.path.splitext(file_name)[1]
    return VideoType.has_extension(ext)


def is_audio(file_name: str) -> bool:
    """
    Check if the file is of audio type
    :param file_name: File name
    :return: Whether the file is of audio type
    """
    ext = os.path.splitext(file_name)[1]
    return AudioType.has_extension(ext)


def is_multimodal_source_path(path: str, check_function: Callable[[str], bool]) -> bool:
    """
    Check if the files in the path are of the specified multimodal source type
    :param path: File path
    :param check_function: Check function
    :return: Whether the files are of the specified multimodal source type
    """
    if not os.path.exists(path):
        raise RuntimeError(f"{path} does not exist, please check")
    return check_function(os.listdir(path)[0])


def is_image_path(path: str) -> bool:
    """
    Check if the files in the path are of image type
    :param path: File path
    :return: Whether the files are of image type
    """
    return is_multimodal_source_path(path, is_image)


def is_video_path(path: str) -> bool:
    """
    Check if the files in the path are of video type
    :param path: File path
    :return: Whether the files are of video type
    """
    return is_multimodal_source_path(path, is_video)


def is_audio_path(path: str) -> bool:
    """
    Check if the files in the path are of audio type
    :param path: File path
    :return: Whether the files are of audio type
    """
    return is_multimodal_source_path(path, is_audio)


def safe_open_audio(audio_cls, audio_path: str, mode='r', is_exist_ok=True, **kwargs):
    """
    :param audio_path: 文件路径
    :param mode: 文件打开模式
    :param check_link: 是否校验软链接
    :param kwargs:
    :return:
    """
    max_path_length = kwargs.get('max_path_length', MAX_PATH_LENGTH)
    max_file_size = kwargs.get('max_file_size', MAX_AUDIO_FILE_SIZE)
    check_link = kwargs.get('check_link', True)

    audio_path = file_utils.standardize_path(audio_path, max_path_length, check_link)
    file_utils.check_file_safety(audio_path, mode, is_exist_ok, max_file_size)

    return audio_cls.load(audio_path)


def safe_open_image(image_cls, file_path: str, mode='r', is_exist_ok=True, **kwargs):
    """
    :image_cls: 图像类
    :param file_path: 文件路径
    :param mode: 文件打开模式
    :param check_link: 是否校验软链接
    :param kwargs:
    :return:
    """
    if not (isinstance(image_cls, types.ModuleType) and image_cls.__name__ == "PIL.Image"):
        raise ValueError("Unsupported image loader type."
                         " Please use PIL.Image or implement a similar class with size validation to ensure security.")
    Image.MAX_IMAGE_PIXELS = kwargs.get("max_image_pixels", MAX_IMAGE_PIXELS)
    warnings.simplefilter("error", Image.DecompressionBombWarning)

    max_path_length = kwargs.get('max_path_length', MAX_PATH_LENGTH)
    max_file_size = kwargs.get('max_file_size', MAX_IMAGE_FILE_SIZE)
    check_link = kwargs.get('check_link', True)

    file_path = file_utils.standardize_path(file_path, max_path_length, check_link)
    file_utils.check_file_safety(file_path, mode, is_exist_ok, max_file_size)

    try:
        return image_cls.open(file_path)
    except Image.DecompressionBombWarning as e:
        raise RuntimeError(f'{e}') from e
    except Exception as e:
        raise RuntimeError("Open image failed.") from e


def check_video_path(file_path: str, mode='r', is_exist_ok=True, **kwargs):
    """
    :param file_path: 文件路径
    :param mode: 文件打开模式
    :param check_link: 是否校验软链接
    :param kwargs:
    :return:
    """
    max_path_length = kwargs.get('max_path_length', MAX_PATH_LENGTH)
    max_file_size = kwargs.get('max_file_size', MAX_VIDEO_FILE_SIZE)
    check_link = kwargs.get('check_link', True)

    file_path = file_utils.standardize_path(file_path, max_path_length, check_link)
    file_utils.check_file_safety(file_path, mode, is_exist_ok, max_file_size)

    return file_path


def validate_image_loader(target_func: Callable, kwargs: dict) -> None:
    """
    Validate the image loader function and set image-specific parameters.
    :param target_func: The target function to load the image
    :param kwargs: Additional keyword arguments
    :raises ValueError: If the target function is not a valid image loader
    """
    # Temporarily only support PIL.Image
    if not (isinstance(target_func, (types.FunctionType, types.MethodType)) and target_func.__module__ == "PIL.Image"):
        raise ValueError(
            "Unsupported image loader type. "
            "Please use PIL.Image or implement a similar class with size validation to ensure security.")
    # Dealing with Decompression Bomb by setting MAX_IMAGE_PIXELS of Image
    Image.MAX_IMAGE_PIXELS = kwargs.pop("max_image_pixels", MAX_IMAGE_PIXELS)
    # Upgrade the level of DecompressionBombWarning from warning to error
    warnings.simplefilter("error", Image.DecompressionBombWarning)


def safe_load_multimodal_source(target_func: Callable, file_path: str, mode: str = 'r', is_exist_ok: bool = True,
                                **kwargs: Any) -> Any:
    """
    Safely load a multimodal source (image, video, or audio) using the specified target function.
    :param target_func: The target function to load the source
    :param file_path: The file path
    :param mode: The file open mode
    :param is_exist_ok: Whether it is okay if the file exists
    :param kwargs: Additional keyword arguments
    :return: The content loaded by the target function
    :raises ValueError: If the file type is not supported
    """
    # Determine the multimodal source type and set corresponding maximum file size
    if is_image(file_path):
        validate_image_loader(target_func, kwargs)
        max_file_size = kwargs.pop(MAX_FILE_SIZE_KEY, MAX_IMAGE_FILE_SIZE)
    elif is_video(file_path):
        max_file_size = kwargs.pop(MAX_FILE_SIZE_KEY, MAX_VIDEO_FILE_SIZE)
    elif is_audio(file_path):
        max_file_size = kwargs.pop(MAX_FILE_SIZE_KEY, MAX_AUDIO_FILE_SIZE)
    else:
        raise ValueError("Multimodal source type should be among image, video and audio. "
                         "Or the format of this modality is temporarily not supported.")

    max_path_length = kwargs.pop(MAX_PATH_LENGTH_KEY, MAX_PATH_LENGTH)
    check_link = kwargs.pop(CHECK_LINK_KEY, True)

    file_path = file_utils.standardize_path(file_path, max_path_length, check_link)
    file_utils.check_file_safety(file_path, mode, is_exist_ok, max_file_size)

    return target_func(file_path, **kwargs)
