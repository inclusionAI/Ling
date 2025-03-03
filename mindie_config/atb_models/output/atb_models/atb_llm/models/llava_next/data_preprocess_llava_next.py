# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from PIL import Image
import av
import numpy as np

from atb_llm.utils.multimodal_utils import safe_open_image, check_video_path

_VIDEO_TOKEN_ID = -2
_IMAGE_FEATURE_WIDTH = 144


def is_video(file_name):
    ext = os.path.splitext(file_name)[1]
    ext = ext.lower()
    if ext in [".mp4", ".wmv", ".avi"]:
        return True
    return False


def is_image(file_name):
    ext = os.path.splitext(file_name)[1]
    ext = ext.lower()
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return True
    return False


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """
    Calculate the number of patches after the preprocessing for images of any resolution.

    Args:
        image_size (`Union[torch.LongTensor, np.ndarray, Tuple[int, int]):
            The size of the input image in the format (height, width). ?
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        int: the number of patches
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise ValueError(f"image_size invalid type {type(image_size)} with value {image_size}")
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution
    num_patches = 0
    # consider change to ceil(height/patch_size)*ceil(width/patch_size) + 1
    for _ in range(0, height, patch_size):
        for _ in range(0, width, patch_size):
            num_patches += 1
    # add the base patch
    num_patches += 1
    return num_patches


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise ValueError(
                f"image_size invalid type: {type(image_size)} not valid, \
                should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def calculate_unpadded_shape(input_shape, original_size):
    """
    根据输入图像的形状和原始大小计算去除填充后的形状。

    Args:
        input_shape (`tuple`): 当前图像的形状 (num_channels, height, width)。
        original_size (`tuple`): 原始图像的大小 (height, width)。

    Returns:
        `tuple`: 去除填充后的形状 (num_channels, new_height, new_width)。
    """
    current_height, current_width = input_shape
    original_height, original_width = original_size

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        new_height = current_height - 2 * padding
        new_width = current_width  # 宽度保持不变
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        new_width = current_width - 2 * padding
        new_height = current_height  # 高度保持不变

    return (new_height, new_width)


def pack_image_features(image_feature_shapes, image_sizes, config, image_newline=False):
    """
    Reshape, unpad and then pack each image_feature into a 
    single image_features tensor containing all visual vectors.

    Args:
        image_feature_shapes: List[torch.Tensor], each of shape 
            `(num_patches, image_length, embed_dim)`)
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Actual image size of each images (H, W).
        image_newline: bool, is there a new line embedding vector.
    Returns:
        feature_lens (`List[int]`)
            token length of each image in image_features
    """
    feature_lens = []
    for image_idx, single_feature_shape in enumerate(image_feature_shapes):
        if single_feature_shape[0] > 1:
            image_length = single_feature_shape[1]
            height = width = config.vision_config.image_size // config.vision_config.patch_size
            if height * width != image_length:
                raise ValueError("The number of patches is not consistent with the image size.")
            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                image_sizes[image_idx],
                config.image_grid_pinpoints,
                config.vision_config.image_size,
            )
            new_height, new_width = calculate_unpadded_shape([num_patch_height * height, num_patch_width * width], 
                image_sizes[image_idx])
            if image_newline:
                new_width += 1
            cur_length = new_height * new_width + image_length
        else:
            cur_length = single_feature_shape[1]
            if image_newline:
                cur_length += 1
        feature_lens.append(cur_length)
    feature_lens = torch.tensor(feature_lens)
    return feature_lens


def data_prerocess_llava_next(processor, config, image_or_video_path: str):
    image_sizes = None
    if is_image(image_or_video_path):
        image = safe_open_image(Image, image_or_video_path)
        inputs = processor.image_processor(images=image, return_tensors="pt")
        image.close()
        pixel_values = inputs["pixel_values"].half().numpy()
        image_sizes = inputs["image_sizes"]
        image_num_patches = image_size_to_num_patches(
                image_size=image_sizes[0],
                grid_pinpoints=config.image_grid_pinpoints,
                patch_size=config.vision_config.image_size,
            )
        
        feature_height = feature_width = config.vision_config.image_size // config.vision_config.patch_size
        image_hidden_dims = config.text_config.hidden_size
        dummy_features_shapes = [
            (
                image_num_patches,
                feature_height * feature_width,
                image_hidden_dims,
            ),
        ]
        image_newline = True if config.text_config.hidden_size else False
        feature_lens = pack_image_features(
            dummy_features_shapes,
            image_sizes,
            config,
            image_newline,
        )
        image_sizes = image_sizes[0]
        image_or_video_token_id = config.image_token_index
    elif is_video(image_or_video_path):
        image_or_video_path = check_video_path(image_or_video_path)
        container = av.open(image_or_video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        container.close()
        pixel_values = processor.video_processor(
            images=clip, 
            return_tensors="pt",
        )["pixel_values_videos"]
        pixel_values = pixel_values.half().numpy()
        frames = len(indices)
        feature_lens = frames * _IMAGE_FEATURE_WIDTH
        if not config.video_token_index:
            config.video_token_index = _VIDEO_TOKEN_ID
        image_or_video_token_id = config.video_token_index
    else:
        raise ValueError("Unsupport extension type!")
    result = (pixel_values, feature_lens, image_or_video_token_id, image_sizes)
    return result