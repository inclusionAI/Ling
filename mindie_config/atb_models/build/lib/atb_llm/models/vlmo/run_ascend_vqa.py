# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import time
import copy
import logging as logger
import json
from transformers import BertTokenizer
import torch
import torch_npu
import numpy as np
from vlmo.config import ex
from vlmo.modules.vlmo_module import VLMo
from vlmo.datasets import VQAv2Dataset
from atb_llm.utils.file_utils import safe_open
import vlmo_ascend_utils
from vlmo.modules.vlmo_file_check import file_check, ErrorCode





@ex.automain
def main(_config):
    device_id = [2]
    voa_arrow_dir = "./arrow"
    bert_vocab = "./vocab.txt"
    _config = copy.deepcopy(_config)
    _config['device'] = device_id
    database = VQAv2Dataset(
        image_size=_config["image_size"],
        data_dir=voa_arrow_dir,
        transform_keys=_config["val_transform_keys"],
        split="val",
    )
    load_path = _config['load_path']
    _config['load_path'] = load_path + 'vlmo_base_patch16_480_vqa.pt'
    file_check(bert_vocab)
    try:
        database.tokenizer = BertTokenizer.from_pretrained(bert_vocab)
    except EnvironmentError:
        logger.error("Get vocab from BertTokenizer failed, "
                               "please check vocab files in the vqa model path.",
                               extra={'error_code': ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise EnvironmentError("Get vocab from BertTokenizer failed, "
                               "please check vocab files in the model path.") from None

    torch_npu.npu.set_device(device_id)

    model = VLMo(_config).npu().half()
    model.eval()

    textlables = torch.full((database.max_text_len,), -101, dtype=torch.int)
    label2ans = vlmo_ascend_utils.label_2_ans(voa_arrow_dir)
    jl = []
    for res in vlmo_ascend_utils.safe_iter(database):
        qid = res["qid"]
        question, other = res["text"]
        image = res["image"]
        logger.info("qid: %s %s", qid, question)
        
        batch = {}
        batch["text_ids"] = (
            torch.tensor(other["input_ids"]).reshape(1, database.max_text_len).npu()
        )
        batch["text_masks"] = (
            torch.tensor(other["attention_mask"])
            .reshape(1, database.max_text_len)
            .npu()
        )
        batch["text_labels"] = textlables.reshape(1, database.max_text_len).npu()
        image_tensor = []
        image_tensor.append(torch.tensor(image[0].reshape(1, 3, 480, 480)).npu())
        batch["image"] = image_tensor
        answers = res["vqa_answer"]
        with torch.no_grad():
            start_time_org = time.time()
            infer = model.infer_ascend(batch, mask_text=False)
            end_time_org = time.time()
            vqa_logits = model.vqa_classifier(infer["cls_feats"])
            _, preds = vqa_logits.max(-1)
        res = label2ans[preds[0].item()]
        logger.info("cost: %f ms", float(end_time_org - start_time_org) * 1000)
        logger.info("res: %s", res)
        logger.info("answers: %s", answers)
        wd = {
            "qid": qid,
            "question": question,
            "cost": float(end_time_org - start_time_org) * 1000,
            "res": res,
            "answers" : answers
        }
        jl.append(wd)
    vlmo_ascend_utils.get_data(jl)
