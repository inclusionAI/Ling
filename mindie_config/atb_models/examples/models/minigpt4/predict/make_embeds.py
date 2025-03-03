# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import argparse
import os
import torch
import torch_npu
from transformers import StoppingCriteriaList

from atb_llm.utils.log import logger
from atb_llm.utils import file_utils
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, StoppingCriteriaSub


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--npu_id", type=int, default=0,
                        help="Specify the npu to work on.")
    parser.add_argument("--cfg_path", type=str, required=True,
                        help="Path to configuration file.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Image path(directory or file) for inference.")
    parser.add_argument("--inputs_embeds_dir", type=str, required=True,
                        help="Directory of .pt files containing inputs_embeds.")
    parser.add_argument("--options", nargs="+",
                        help="override some settings in the used config, the key-value pair "
                             "in xxx=yyy format will be merged into config file (deprecate), "
                             "change to --cfg-options instead.",
                        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args)

    torch_npu.npu.set_device(args.npu_id)
    torch.npu.set_compile_mode(jit_compile=False)

    logger.info('----- Chat Initialization Begins ... -----')
    model_config = cfg.model_cfg
    model_config.device_8bit = args.npu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'npu:{args.npu_id}')

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]  # 835:###, 2277:##, 29937:#
    stop_words_ids = [torch.tensor(ids).to(device=f'npu:{args.npu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, stopping_criteria=stopping_criteria, device=f'npu:{args.npu_id}')
    logger.info('----- Chat Initialization Finished! -----')

    logger.info('----- inputs_embeds Making Begins ... -----')
    if not os.path.isdir(args.inputs_embeds_dir):
        os.mkdir(args.inputs_embeds_dir)

    if not os.path.isdir(args.image_path):
        image_path_list = [args.image_path]
    else:
        image_path_list = [os.path.join(args.image_path, _) for _ in os.listdir(args.image_path)]

    input_text = "Describe this image in detail."

    for image_path in sorted(image_path_list):
        chat_state = CONV_VISION_Vicuna0.copy()
        img_list = []
        chat.upload_img(image_path, chat_state, img_list)
        chat.encode_img(img_list)
        chat.ask(input_text, chat_state)
        inputs_embeds = chat.answer_prepare(conv=chat_state, img_list=img_list)["inputs_embeds"]
        logger.info(f"{inputs_embeds=}")
        inputs_embeds_file_path = os.path.join(args.inputs_embeds_dir, f"{os.path.basename(image_path)}.pt")
        inputs_embeds_file_path = file_utils.standardize_path(inputs_embeds_file_path)
        file_utils.check_file_safety(inputs_embeds_file_path, 'w')
        torch.save(inputs_embeds, inputs_embeds_file_path)
        logger.info('----- inputs_embeds .pt file Saved! -----')
        logger.info(f"{inputs_embeds_file_path=}")

    logger.info(f'----- inputs_embeds Making All Finished! Total: {len(image_path_list)} -----')


