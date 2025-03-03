# coding=utf-8
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# This file was referenced from OpenGVLab
import os

import numpy as np
from PIL import Image
import av
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from atb_llm.utils import multimodal_utils
from atb_llm.utils import shm_utils
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.log.logging import logger


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VISION_MODEL = 'vision_model'
MLP_PROJECTOR = 'mlp1'

# Weight suffix and split dimension for InternVisionModel
WEIGHT_KEYS_MAPPING = {
    'attn.qkv.weight': 0,
    'attn.proj.weight': 0,
    'mlp.fc1.weight': 0,
    'mlp.fc2.weight': 1,
}
BIAS_KEYS = ['attn.qkv.bias', 'attn.proj.bias', 'mlp.fc1.bias']


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def build_transform_no_norm(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
    ])
    return transform


def create_standardization_params(mean, std, scale, input_channel):
    if isinstance(std, (list, tuple)):
        std = torch.Tensor(std)
    if isinstance(mean, (list, tuple)):
        mean = torch.Tensor(mean)
    
    if torch.any(std == 0):
        raise ZeroDivisionError("Value of std is zero.")
    weight = (scale / std).view(input_channel, 1, 1, 1)
    bias = -mean.view(input_channel) / std.view(input_channel)

    return weight, bias


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        if ratio[1] == 0:
            raise ZeroDivisionError("Provided ratio is zero")
        target_aspect_ratio = np.divide(ratio[0], ratio[1])
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    if orig_height == 0:
        raise ZeroDivisionError("Original height of image is zero")
    aspect_ratio = np.divide(orig_width, orig_height)

    # calculate the existing image aspect ratio
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if len(processed_images) != blocks:
        raise ValueError("Number of processed images does not the match number of blocks")
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12, use_dynamic_prepro=True):
    image = multimodal_utils.safe_load_multimodal_source(Image.open, image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if use_dynamic_prepro:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image]
    pixel_values = []
    for image in images:
        pixel_values.append(transform(image))
        image.close()
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_and_preprocess_image(image_file, input_size=448, max_num=12, normalizer=None, use_dynamic_prepro=True):
    image = multimodal_utils.safe_load_multimodal_source(Image.open, image_file).convert('RGB')
    transform = build_transform_no_norm(input_size=input_size)  # No normalization and rescaling

    if use_dynamic_prepro:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image]

    pixel_values = []
    for image in images:
        # Convert to tensor and permute dimensions (C x H x W format)
        pixel_value = torch.from_numpy(np.array(transform(image))).permute(2, 0, 1)
        pixel_values.append(pixel_value)
        image.close()
    pixel_values = torch.stack(pixel_values)

    # Apply preprocessing layer (normalization based on convolution operation)
    if normalizer is not None:
        pixel_values = normalizer(pixel_values.npu().float())

    return pixel_values


def split(tensor: torch.Tensor, tp_size: int, tp_rank: int, dim=0):
    if tp_size == 1:
        return tensor
    if not (len(tensor.shape) > 1 or dim == 0):
        logger.error("Invalid dimension for splitting. Expected len(tensor.shape) > 1 or dim == 0.",
                     ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError("Invalid dimension for splitting. Expected len(tensor.shape) > 1 or dim == 0.")
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(np.split(tensor, tp_size, axis=dim)[tp_rank].copy())
    if tensor.shape[dim] % tp_size != 0:
        logger.error(f"Unable to split: shape={tensor.shape} (dim={dim}) tp_size={tp_size}.",
                     ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError(f"Unable to split: shape={tensor.shape} (dim={dim}) tp_size={tp_size}.")
    split_size = tensor.shape[dim] // tp_size
    return tensor.split(split_size, dim=dim)[tp_rank].clone().detach()


def internvl_tensor_parallel_split(
    key: str, 
    prefix: str, 
    tp_rank: int, 
    tp_size: int, 
    saved_weight: torch.nn.Parameter
):
    if prefix == VISION_MODEL:
        for k, dim in WEIGHT_KEYS_MAPPING.items():
            if k in key:
                saved_weight.data = split(saved_weight.data, tp_size, tp_rank, dim=dim)
                return saved_weight
            
        if any(k in key for k in BIAS_KEYS):
            saved_weight.data = torch.chunk(saved_weight.data, tp_size)[tp_rank]
    
    if prefix == MLP_PROJECTOR:
        if '1.weight' in key:
            saved_weight.data = split(saved_weight.data, tp_size, tp_rank, dim=0)
        elif '1.bias' in key:
            saved_weight.data = torch.chunk(saved_weight.data, tp_size)[tp_rank]
        elif '3.weight' in key:
            saved_weight.data = split(saved_weight.data, tp_size, tp_rank, dim=1)

    return saved_weight


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        if len(bound) < 2:
            logger.error(f"The  size of parameter bound should be 2, but get {len(bound)}",
                         ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
            raise ValueError(f"The  size of parameter bound should be 2, but get {len(bound)}")
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    if num_segments == 0:
        logger.error('The parameter num_segments can not be 0.',
                     ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError('The parameter num_segments can not be 0.')
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, use_dynamic_prepro=True):
    try:
        # 打开视频并获取视频流
        container = multimodal_utils.safe_load_multimodal_source(av.open, video_path)
        video_stream = next(s for s in container.streams if s.type == 'video')
    except Exception as e:
        logger.error(f'Read video error:{e}',
                     ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
        raise RuntimeError(f'Read video error:{e}') from e
    # 获取视频的总帧数和 FPS
    max_frame = video_stream.frames - 1
    fps = float(video_stream.average_rate)
    # 初始化存储张量的列表
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    # 获取指定的帧索引
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=32)
    # 逐帧读取指定索引的帧
    frame_count = 0
    for frame in container.decode(video=0):
        if frame_count > max(frame_indices):
            break
        if frame_count in frame_indices:
            # 转换帧到图像
            img = Image.fromarray(frame.to_rgb().to_ndarray()).convert('RGB')
            if use_dynamic_prepro:
                img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            else:
                img = [img]
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        frame_count += 1
    # 合并所有帧的像素值
    pixel_values = torch.cat(pixel_values_list)
    container.close()
    return pixel_values, num_patches_list


def process_image_input(single_input, input_num, image_index, use_dynamic_prepro, num_image_token, shm_name_save_path):
    pixel_value = load_and_preprocess_image(single_input.get("image"), use_dynamic_prepro=use_dynamic_prepro)
    pixel_value = pixel_value.numpy()
    if input_num == 1:
        current_query = (f'<img>{"<IMG_CONTEXT>" * pixel_value.shape[0] * num_image_token}</img>\n')
    else:
        current_query = (f'Image-{image_index}: '
                    f'<img>{"<IMG_CONTEXT>" * pixel_value.shape[0] * num_image_token}</img>\n')
        image_index += 1
    if shm_name_save_path is None:
        shm_name_save_dir = os.path.dirname(os.path.dirname(single_input.get("image")))
        shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
    shm = shm_utils.create_shm(pixel_value.nbytes, shm_name_save_path)
    shared_array = np.ndarray(pixel_value.shape, dtype=pixel_value.dtype, buffer=shm.buf)
    shared_array[:] = pixel_value
    shm_name_value = shm_utils.encode_shm_name_to_int64(shm.name)
    shape_value = shm_utils.encode_shape_to_int64(pixel_value.shape)
    return current_query, shm_name_value, shape_value


def process_video_input(single_input, use_dynamic_prepro, num_image_token, shm_name_save_path):
    pixel_value, num_patches_list = load_video(single_input.get("video"),
        use_dynamic_prepro=use_dynamic_prepro)
    pixel_value = pixel_value.numpy().astype(np.float16)

    pre_index = 0
    current_query = ""
    shm_name_value = []
    shape_value = []
    if shm_name_save_path is None:
        shm_name_save_dir = os.path.dirname(os.path.dirname(single_input.get("video")))
        shm_name_save_path = os.path.join(shm_name_save_dir, "shm_name.txt")
    for i, num_patch in enumerate(num_patches_list):
        current_query += (f'Frame{i+1}: '
                    f'<img>{"<IMG_CONTEXT>" * num_patch * num_image_token}</img>\n')
        single_frame = pixel_value[pre_index : pre_index + num_patch]
        pre_index += num_patch
        shm = shm_utils.create_shm(single_frame.nbytes, shm_name_save_path)
        shared_array = np.ndarray(single_frame.shape, dtype=np.float16, buffer=shm.buf)
        shared_array[:] = single_frame
        shm_name_value.append(shm_utils.encode_shm_name_to_int64(shm.name))
        shape_value.append(shm_utils.encode_shape_to_int64(pixel_value.shape))
    
    return current_query, shm_name_value, shape_value