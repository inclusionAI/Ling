# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import argparse
import json
import os
import stat
import torch
from transformers import StoppingCriteriaList

from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log import logger

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, StoppingCriteriaSub


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="Specify the gpu to load the model.")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml",
                        help="Path to configuration file.")
    parser.add_argument("--image-path", required=True,
                        help="Image path for inference.")
    parser.add_argument("--output-path", required=True,
                        help="Output path of inference.")
    parser.add_argument("--options", nargs="+",
                        help="override some settings in the used config, the key-value pair "
                             "in xxx=yyy format will be merged into config file (deprecate), "
                             "change to --cfg-options instead.",
                        )
    return parser.parse_args()


def traverse_img_dir(img_dir, res_file_dir):
    # 判断目标目录是否存在
    if not os.path.exists(img_dir):
        logger.info("目标目录不存在！")
        return
    if not os.path.exists(res_file_dir):
        os.mkdir(res_file_dir)

    input_text = "Describe this image in detail."

    for root, _, files in os.walk(img_dir):
        if not files:
            continue
        res_dict = {}
        for file in files:
            image_path = os.path.join(root, file)
            logger.info("文件路径：", image_path)
            chat_state = CONV_VISION_Vicuna0.copy()
            img_list = []
            llm_message = chat.upload_img(image_path, chat_state, img_list)
            logger.info(f"{llm_message=}")
            chat.encode_img(img_list)
            logger.info(f"===== image_list: {img_list}")
            logger.info(f"===== chat_state: {chat_state.messages}")
            chat.ask(input_text, chat_state)
            llm_message = chat.answer(conv=chat_state,
                                      img_list=img_list,
                                      num_beams=1,
                                      temperature=0.1,
                                      max_new_tokens=300,
                                      max_length=2000)[0]
            logger.info(f"MiniGPT4 Answer: {llm_message}")
            res_dict[image_path] = llm_message
            logger.info(f"已生成 {len(res_dict)} 条记录 from {root}")

            flags = os.O_WRONLY | os.O_CREAT
            mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            with safe_open(args.results_save_path, "w", permission_mode=mode, override_flags=flags) as f:
                json.dump(res_dict, f)
    logger.info("-----ALL DONE-----")


if __name__ == '__main__':
    # Model Initialization
    logger.info('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f'cuda:{args.gpu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}', stopping_criteria=stopping_criteria)
    logger.info('Initialization Finished')

    # Model Inference
    traverse_img_dir(img_dir=args.image_path, res_file_dir=args.output_path)

