# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
from pathlib import Path
from typing import List


def weight_files(
        model_id: str, extension: str = ".safetensors"
) -> List[Path]:
    """Get the local files"""
    # Local model
    if Path(model_id).exists() and Path(model_id).is_dir():
        local_files = list(Path(model_id).glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(
                f"No local weights found in {model_id} with extension {extension};"
                f"当前仅支持加载safetensor格式的权重，请参考模型Readme获取safetensor格式的权重"
            )
        return local_files
    
    raise FileNotFoundError("The input model id is not exists or not a directory")
