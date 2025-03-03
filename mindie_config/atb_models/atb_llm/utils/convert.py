# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
# Part of this file was copied from project 'text-generation-inference/0.9.2'.
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import torch
from safetensors.torch import save_file, load_file, _find_shared_tensors, _is_complete

from .log import logger
from . import file_utils


def _remove_duplicate_names(
        state_dict: Dict[str, torch.Tensor],
        *,
        preferred_names: List[str] = None,
        discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = list(set(preferred_names))
    if discard_names is None:
        discard_names = []
    discard_names = list(set(discard_names))

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])]
        )
        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict,"
            )
        keep_name = sorted(list(complete_names))[0]

        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def convert_file(pt_file: Path, sf_file: Path, discard_names: List[str]):
    pt_file = file_utils.standardize_path(str(pt_file), check_link=False)
    file_utils.check_file_safety(pt_file, 'r', is_check_file_size=False)
    loaded_state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
    if "state_dict" in loaded_state_dict:
        loaded_state_dict = loaded_state_dict["state_dict"]
    to_remove_dict = _remove_duplicate_names(loaded_state_dict, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_list in to_remove_dict.items():
        for to_remove in to_remove_list:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded_state_dict[to_remove]

    loaded_state_dict = {k: v.contiguous() for k, v in loaded_state_dict.items()}

    os.makedirs(os.path.dirname(sf_file), exist_ok=True)
    sf_file = file_utils.standardize_path(str(sf_file), check_link=False)
    file_utils.check_file_safety(sf_file, 'w', is_check_file_size=False)
    save_file(loaded_state_dict, sf_file, metadata=metadata)

    reloaded_state_dict = load_file(sf_file)
    for k, pt_tensor in loaded_state_dict.items():
        sf_tensor = reloaded_state_dict[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_files(pt_files: List[Path], sf_files: List[Path], discard_names: List[str]):
    num_pt_files = len(pt_files)

    for i, (pt_file, sf_file) in enumerate(zip(pt_files, sf_files)):
        blacklisted_keywords = ["arguments", "args", "training"]
        if any(substring in pt_file.name for substring in blacklisted_keywords):
            continue

        start_time = datetime.now(tz=timezone.utc)
        convert_file(pt_file, sf_file, discard_names)
        elapsed_time = datetime.now(tz=timezone.utc) - start_time
        try:
            logger.info(f"Convert: [{i + 1}/{num_pt_files}] -- Took: {elapsed_time}")
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
