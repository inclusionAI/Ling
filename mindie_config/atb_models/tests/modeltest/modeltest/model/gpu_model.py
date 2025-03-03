# Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.
import torch
from modeltest.api.model import Model
from atb_llm.runner.tokenizer_wrapper import TokenizerWrapper


class GPUModel(Model):
    def __init__(self, backend, *args) -> None:
        super().__init__('GPU', *args)
        self.backend = backend
        self.rank = torch.cuda.current_device()
        self.tokenizer_wrapper = TokenizerWrapper(model_name_or_path=self.model_config.model_path)
        self.tokenizer = self.tokenizer_wrapper.tokenizer
        self.model = self.get_model()
    
    def get_model(self):
        raise NotImplementedError("Subclasses should implement get_model.")
    
    def inference(self, infer_input, output_token_num):
        raise NotImplementedError("Subclasses should implement inference.")
    
    def construct_inputids(self, input_texts, use_chat_template=False, is_truncation=True):
        def build_inputs(input_builder, conversations, **kwargs):
            return [input_builder.make_context(self.rank, conversation, **kwargs) for conversation in conversations]
        if not (isinstance(input_texts, list) and input_texts and isinstance(input_texts[0], str)):
            raise ValueError(f"The input_texts must be as List[str]."
                            f" Now the inputs ({input_texts}) is not acceptable or is empty")
        if use_chat_template:
            input_conversations = [[{"role":"user", "content": t}] for t in input_texts]
            input_ids = build_inputs(self.tokenizer_wrapper.input_builder, input_conversations)
            input_texts = [self.tokenizer.decode(ids) for ids in input_ids]
        return self.tokenizer(input_texts, padding=True, return_tensors="pt", truncation=is_truncation)
        