# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
from vllm import LLM, SamplingParams
from modeltest.model.gpu_model import GPUModel
from modeltest.api.model import ResultMetadata


class VllmModel(GPUModel):
    def __init__(self, tp, *args) -> None:
        self.tp = tp
        super().__init__('vllm', *args)

    def get_model(self):
        model = LLM(
            model=self.model_config.model_path,
            tensor_parallel_size=self.tp, 
            dtype="auto",
            enforce_eager=True)
        model.set_tokenizer(self.tokenizer)
        return model

    def get_vllm_sampling_params(self, output_token_num):
        return SamplingParams(temperature=0, max_tokens=output_token_num)
        
    def inference(self, infer_input, output_token_num):
        inputs = self.construct_inputids(infer_input, self.model_config.use_chat_template)
        prompts = [self.tokenizer.decode(ids) for ids in inputs]
        outputs = self.model.generate(prompts, self.get_vllm_sampling_params(output_token_num))
        generate_texts = [output.outputs[0].text for output in outputs]
        outputs_ids_list = [output.outputs[0].token_ids for output in outputs]
        input_ids_list = [output.prompt_token_ids for output in outputs]
        return ResultMetadata(
            generate_text=generate_texts,
            generate_id=outputs_ids_list,
            logits=[],
            input_id=input_ids_list,
            token_num=[],
            e2e_time=0)
        