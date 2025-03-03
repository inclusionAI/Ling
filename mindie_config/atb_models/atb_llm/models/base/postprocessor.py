# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
from typing import List, Union

from atb_llm.utils.env import ENV
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.argument_utils import MAX_KEY_LENGTH


class Postprocessor:
    """A stop checker only used in run_pa.py"""
    def __init__(self, tokenizer, generation_config):
        self.max_length: int = generation_config.get('max_length', None)
        self.max_new_tokens: int = generation_config.get('max_new_tokens', None)
        self.min_length: int = generation_config.get('min_length', None)
        self.min_new_tokens: int = generation_config.get('min_new_tokens', None)

        self.do_sample: bool = generation_config.get('do_sample', None)
        self.num_beams: int = generation_config.get('num_beams', None)
        self.num_beam_groups: int = generation_config.get('num_beam_groups', None)
        self.penalty_alpha: float = generation_config.get('penalty_alpha', None)

        self.temperature: float = generation_config.get('temperature', None)
        self.top_k: int = generation_config.get('top_k', None)
        self.top_p: float = generation_config.get('top_p', None)
        self.typical_p: float = generation_config.get('typical_p', None)
        self.epsilon_cutoff: float = generation_config.get('epsilon_cutoff', None)
        self.eta_cutoff: float = generation_config.get('eta_cutoff', None)
        self.diversity_penalty: float = generation_config.get('diversity_penalty', None)
        self.repetition_penalty: float = generation_config.get('repetition_penalty', None)
        self.encoder_repetition_penalty: float = generation_config.get('encoder_repetition_penalty', None)
        self.length_penalty: float = generation_config.get('length_penalty', None)

        self.tokenizer = tokenizer
        self.pad_token_id: int = generation_config.get('pad_token_id', None)
        self.bos_token_id: int = generation_config.get('bos_token_id', None)
        self.eos_token_id: Union[int, List[Union[int, List[int]]]] = generation_config.get('eos_token_id', None)
        if not self.eos_token_id:
            self.eos_token_id = tokenizer.eos_token_id

    def stopping_criteria(self, output_ids):
        """Check if the output ids encounter eos token id."""
        ret = False
        rank = ENV.rank
        if isinstance(self.eos_token_id, int):
            ret = output_ids[-1] == self.eos_token_id
        elif isinstance(self.eos_token_id, list):
            is_end_list = []
            for eos in self.eos_token_id:
                if isinstance(eos, int):
                    is_end_list.append(output_ids[-1] == eos)
                elif isinstance(eos, list):
                    is_end_list.append(len(output_ids) >= len(eos) and output_ids[-len(eos):] == eos)
                else:
                    print_log(rank, logger.warning, f"unsupport type of eos_token_id: "
                              f"{self.eos_token_id}.\nPlease check the type of your eos_token_id. It must be "
                              f"Union[int, List[Union[int, List[int]]]].")
            ret = any(is_end_list)
        else:
            print_log(rank, logger.warning, f"unsupport type of eos_token_id: "
                      f"{self.eos_token_id}.\nPlease check the type of your eos_token_id. It must be "
                      f"Union[int, List[Union[int, List[int]]]].")
        
        if ENV.modeltest_dataset_specified:
            ENV.update()
            if isinstance(ENV.modeltest_dataset_specified, str):
                if len(ENV.modeltest_dataset_specified) > MAX_KEY_LENGTH:
                    raise ValueError("The length of environment variable `MODELTEST_DATASET_SPECIFIED` "
                        f"should be no larger than {MAX_KEY_LENGTH}.")
                split_parts = ENV.modeltest_dataset_specified.split('_')
                if len(split_parts) >= 3 and "HumanEval" in split_parts[0]:
                    from tests.modeltest.modeltest.task.humanevalx import is_code_generation_finished
                    text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
                    ret = is_code_generation_finished(
                        text,
                        language_type=split_parts[2],
                        dataset=split_parts[0],
                    )
        return ret