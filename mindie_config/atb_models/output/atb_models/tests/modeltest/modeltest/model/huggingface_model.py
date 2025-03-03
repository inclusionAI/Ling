# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from modeltest.model.gpu_model import GPUModel
from modeltest.api.model import ResultMetadata
from atb_llm.utils.log.logging import logger


dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}


class HuggingfaceModel(GPUModel):
    def __init__(self, _, *args) -> None:
        super().__init__('huggingface', *args)
        
    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_path,
            device_map="auto", 
            torch_dtype=dtype_map.get(self.model_config.data_type, torch.float16),
            trust_remote_code=self.model_config.trust_remote_code)
        model.generation_config = self.__remove_part_of_generation_config(model.generation_config)
        return model
        
    def inference(self, infer_input, output_token_num):
        inputs = self.construct_inputids(infer_input, self.model_config.use_chat_template)
        if self.task_config.need_logits:
            return self.__generate_token(inputs)
        else:
            return self.__generate_text(inputs, output_token_num)
    
    def __generate_text(self, inputs, output_token_num):
        tokenizer_out_ids = inputs.input_ids.to(0)
        attention_mask = inputs.attention_mask.to(0)
        outputs_ids_tensor = self.model.generate(
            inputs=tokenizer_out_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=output_token_num
        )
        outputs_ids_list = [outputs_id[len(inputs["input_ids"][idx]):]
                            for idx, outputs_id in enumerate(outputs_ids_tensor.tolist())]
        generate_texts = [self.tokenizer.decode(output_ids) for output_ids in outputs_ids_list]
        return ResultMetadata(
            generate_text=generate_texts,
            generate_id=outputs_ids_list,
            logits=[],
            input_id=inputs.input_ids.tolist(),
            token_num=[],
            e2e_time=0)
    
    def __generate_token(self, inputs):
        tokenizer_out_ids = inputs.input_ids.to(0)
        attention_mask = inputs.attention_mask.to(0)
        outputs = self.model(
            input_ids=tokenizer_out_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        output_token_ids = logits[:, -1, :].argmax(dim=-1)
        output_text = self.tokenizer.decode(output_token_ids)
        return ResultMetadata(
            generate_text=output_text,
            generate_id=output_token_ids.tolist(),
            logits=logits,
            input_id=inputs.input_ids.tolist(),
            token_num=[],
            e2e_time=0)

    def __remove_part_of_generation_config(self, generation_config):
        logger.info("Original generation config: %s", generation_config)
        ori_gen = GenerationConfig()
        diff_dict = generation_config.to_diff_dict()
        logger.debug(diff_dict)
        for key in diff_dict:
            if key.endswith("_id"):
                continue
            ori_value = getattr(ori_gen, key, None)
            if ori_value is not None:
                setattr(generation_config, key, getattr(ori_gen, key))
                logger.info("replace %s", key)
        logger.info("Generation config after remove: %s", generation_config)
        return generation_config
        