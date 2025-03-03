# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import copy
import logging as logger
import json
import numpy as np
from transformers import BertTokenizer
import torch
import torch_npu
from vlmo.config import ex
from vlmo.modules.vlmo_module import VLMo
from vlmo.datasets import CocoCaptionKarpathyDataset
from atb_llm.utils.file_utils import safe_open
import vlmo_ascend_utils
from vlmo.modules.vlmo_file_check import file_check, ErrorCode

CLS_FEATS = "cls_feats"





@ex.automain
def main(_config):

    device_id = 4
    coco_arrow_dir = "./vlmo/cocoarrow/"
    bert_vocab_coco = "./vocab.txt"
    _config = copy.deepcopy(_config)
    database = CocoCaptionKarpathyDataset(
        image_size=_config["image_size"],
        data_dir=coco_arrow_dir,
        transform_keys=_config["train_transform_keys"],
        split="test",
    )
    file_check(bert_vocab_coco)
    try:
        database.tokenizer = BertTokenizer.from_pretrained(bert_vocab_coco)
    except EnvironmentError:
        logger.error("Get vocab from BertTokenizer failed, "
                               "please check vocab files in the coco model path.",
                               extra={'error_code': ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise EnvironmentError("Get vocab from BertTokenizer failed, "
                               "please check vocab files in the model path.") from None

    torch_npu.npu.set_device(device_id)
    model = VLMo(_config).npu().half()
    model.eval()

    textlables = torch.full((database.max_text_len,), -101, dtype=torch.int)
    for res in database:

        text, other = res["text"]
        image = res["image"]
        batch = {}
        batch["text_ids"] = (
            torch.tensor(other["input_ids"]).reshape(1, database.max_text_len).npu()
        )
        batch["text_labels"] = textlables.reshape(1, database.max_text_len).npu()
        batch["text_masks"] = (
            torch.tensor(other["attention_mask"])
            .reshape(1, database.max_text_len)
            .npu()
        )

        image_tensor = []

        image_tensor.append(torch.tensor(image[0].reshape(1, 3, 384, 384)).npu().half())
        batch["image"] = image_tensor
        with torch.no_grad():
            infer_org = model.infer_text_ft(batch, mask_text=False)
            infer_asc = model.infer_text_ft_ascend(batch, mask_text=False)
            if np.allclose(
                infer_org[CLS_FEATS].cpu(),
                infer_asc[CLS_FEATS].cpu(),
                rtol=0.02,
                atol=0.02,
            ):
                logger.info("==> text result equal.")
            else:
                logger.info("==>!!!text result not equal.")

        with torch.no_grad():
            infer_org = model.infer_image_ft(batch, mask_image=False)
            infer_asc = model.infer_image_ft_ascend(batch, mask_image=False)
            if np.allclose(
                infer_org[CLS_FEATS].cpu(),
                infer_asc[CLS_FEATS].cpu(),
                rtol=0.02,
                atol=0.02,
            ):
                logger.info("==>image  result equal.")
            else:
                logger.info("==>!!!image result not equal.")
