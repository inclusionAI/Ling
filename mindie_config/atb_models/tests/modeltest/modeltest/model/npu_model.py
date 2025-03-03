# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import importlib
import os
from modeltest.api.model import Model, ResultMetadata
from atb_llm.utils.env import ENV


class NPUModel(Model):
    def __init__(self, batch_size, *args) -> None:
        super().__init__('NPU', *args)
        input_dict = {
            'rank': int(os.getenv("RANK", "0")),
            'local_rank': int(os.getenv("LOCAL_RANK", "0")),
            'world_size': int(os.getenv("WORLD_SIZE", "1")),
            'max_prefill_tokens': -1,
            'block_size': self.model_config.block_size,
            'model_path': self.model_config.model_path,
            'max_position_embeddings': (self.model_config.max_position_embedding
                                        if self.model_config.max_position_embedding != -1
                                        else self.task_config.requested_max_input_length +
                                            self.task_config.requested_max_output_length),
            'max_batch_size': batch_size,
            'max_input_length': self.task_config.requested_max_input_length,
            'max_output_length': self.task_config.requested_max_output_length,
            'trust_remote_code': self.model_config.trust_remote_code
        }
        if self.model_config.mm_model:
            input_dict.update(input_texts=self.model_config.mm_model.get('input_texts'))
            input_dict.update(input_texts=self.model_config.mm_model.get('warm_up_input_texts'))
            input_dict.update(image_path=self.model_config.mm_model.get('warm_up_image_path'))
            input_dict.update(video_path=self.model_config.mm_model.get('warm_up_video_path'))
            input_dict.update(audio_path=self.model_config.mm_model.get('warm_up_audio_path'))
            input_dict.update(
                shm_name_save_path=self.model_config.mm_model.get('infer_params').get('shm_name_save_path'))

            self.model = self.get_model_cls()(**input_dict)
        else:
            self.model = self.get_model_cls()(**input_dict)
        self.model.warm_up()

    def get_model_cls(self):
        ENV.update()
        model_name = self.model_config.model_name
        model_type = self.model_config.model_type
        if self.model_config.mm_model:
            path = self.model_config.mm_model["path"]
            module_path = f"examples.models.{model_name}.{path}"
            module = importlib.import_module(module_path)
            runner_name = self.model_config.mm_model["classname"]
        else:
            module_path = f"examples.run_{str.lower(model_type)}"
            module = importlib.import_module(module_path)
            runner_name = f"{str.upper(model_type)}Runner"
        return getattr(module, runner_name)

    def inference(self, infer_input, batch_size, max_output_length, ignore_eos_token, **extra_args):

        if self.model_config.mm_model:
            infer_params = self.model_config.mm_model["infer_params"]
            generate_text_list, token_num_list, e2e_time = self.model.infer(infer_input, **infer_params)
        else:
            batch_size = batch_size if len(infer_input) != 1 else 1
            generate_text_list, token_num_list, e2e_time = self.model.infer(
                infer_input,
                batch_size,
                max_output_length,
                ignore_eos_token,
                self.model_config.use_chat_template,
                **extra_args
            )

        return ResultMetadata(
            generate_text=generate_text_list,
            generate_id=[],
            logits=[],
            input_id=[],
            token_num=token_num_list,
            e2e_time=e2e_time)
