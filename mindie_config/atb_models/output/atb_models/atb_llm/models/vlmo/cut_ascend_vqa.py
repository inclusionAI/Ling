# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import os
import copy
import time
import logging as logger
import json
from transformers import BertTokenizer
import torch
import torch_npu
from vlmo.config import ex
from vlmo.modules.vlmo_module import VLMo
from vlmo.datasets import VQAv2Dataset
import numpy as np
from atb_llm.utils.file_utils import safe_open
import vlmo_ascend_utils
from vlmo.modules.vlmo_file_check import file_check, ErrorCode



LOAD_PATH_CONST = "load_path"


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    device_id = [0, 1]
    _config['device'] = device_id
    voa_arrow_dir = "./arrow"
    bert_vocab = "./vocab.txt"
    load_path = _config[LOAD_PATH_CONST]
    pt_name = "cut_VQA_weights.pt"
    local_rank = 0
    database = VQAv2Dataset(
        image_size=_config["image_size"],
        data_dir=voa_arrow_dir,
        transform_keys=_config["val_transform_keys"],
        split="val",
    )
    file_check(bert_vocab)
    try:
        database.tokenizer = BertTokenizer.from_pretrained(bert_vocab)
    except EnvironmentError:
        logger.error("Get vocab from BertTokenizer failed, "
                               "please check vocab files in the model path.",
                               extra={'error_code': ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID})
        raise EnvironmentError("Get vocab from BertTokenizer failed, "
                               "please check vocab files in the model path.") from None
    if len(device_id) == 1:
        torch_npu.npu.set_device(device_id[0])
        _config[LOAD_PATH_CONST] = os.path.join(load_path, pt_name)
        model = VLMo(_config).half()
        model.eval()
    else:
        torch.distributed.init_process_group("hccl")
        local_rank = torch.distributed.get_rank()
        part_model_path = os.path.join(load_path, 'part_model', str(local_rank), pt_name)
        _config[LOAD_PATH_CONST] = part_model_path
        _config["test_only"] = True
        torch_npu.npu.set_device(device_id[local_rank])
        model = VLMo(_config)
        model = model.npu().half()
        model.eval()
    textlables = torch.full((database.max_text_len,), -101, dtype=torch.int)
    label2ans = vlmo_ascend_utils.label_2_ans(voa_arrow_dir)
    jl = []
    for res in vlmo_ascend_utils.safe_iter(database):
        qid = res["qid"]
        question, other = res["text"]
        image = res["image"]
        
        batch = {}
        batch["text_masks"] = (
            torch.tensor(other["attention_mask"])
            .reshape(1, database.max_text_len)
            .npu()
        )
        batch["text_ids"] = (
            torch.tensor(other["input_ids"]).reshape(1, database.max_text_len).npu()
        )

        batch["text_labels"] = textlables.reshape(1, database.max_text_len).npu()
        image_tensor_cut = []
        image_tensor_cut.append(torch.tensor(image[0].reshape(1, 3, 480, 480)).npu())
        batch["image"] = image_tensor_cut
        answers = res["vqa_answer"]
        with torch.no_grad():
            start_time_org = time.time()
            infer = model.infer_ascend(batch, mask_text=False)
            end_time_org = time.time()
            vqa_logits = model.vqa_classifier(infer["cls_feats"])
            _, preds = vqa_logits.max(-1)
        res = label2ans[preds[0].item()]

        if local_rank != 0:
            logger.info("qid: %s %s", qid, question)
            logger.info("cost: %f ms", float(end_time_org - start_time_org) * 1000)
            logger.info("res: %s", res)
            logger.info("answers: %s", answers)
            results = {
                "qid": qid,
                "question": question,
                "cost": float(end_time_org - start_time_org) * 1000,
                "res": res,
                "answers" : answers
            }
            jl.append(results)
    if local_rank != 0:
        vlmo_ascend_utils.get_data(jl)
