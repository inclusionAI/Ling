# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os
import time
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from atb_llm.runner.model_runner import ModelRunner
from atb_llm.utils.log import logger, print_log
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.env import ENV
from atb_llm.utils import file_utils
from atb_llm.utils.argument_utils import ArgumentParser, StringArgumentValidator, MAX_JSON_LENGTH


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_path', help="model and tokenizer path")
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default="What's deep learning?")
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument('--max_position_embeddings', type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--enable_atb_torch', action='store_true')
    parser.add_argument('--trust_remote_code', action='store_true')

    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)

    parser.add_argument('--inputs_embeds_dir', type=str, default=None,
                        help='Directory of .pt files containing inputs_embeds.')
    parser.add_argument('--min_length', type=int, default=10)
    parser.add_argument('--stop_words_ids', type=str, default=None,
                        validator=StringArgumentValidator(min_length=2, max_length=MAX_JSON_LENGTH, allow_none=True))
    parser.add_argument('--do_sample', type=cmd_bool, default=False)
    parser.add_argument('--results_save_path', type=str, default=None,
                        help='File path to save inference results.')

    return parser.parse_args()


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be a boolean")


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=None):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all(torch.eq(input_ids[:, -len(stop):], stop)).item():
                return True
        return False


class FARunner:
    def __init__(self, **kwargs):
        self.rank = kwargs.get('rank', '0')
        self.local_rank = kwargs.get('local_rank', self.rank)
        self.world_size = kwargs.get('world_size', '1')

        self.model_path = kwargs.get('model_path', None)
        self.max_input_length = kwargs.get('max_input_length', None)
        self.max_output_length = kwargs.get('max_output_length', None)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)
        self.batch_size = kwargs.get('batch_size', None)
        self.enable_atb_torch = kwargs.get('enable_atb_torch', False)
        self.num_beams = kwargs.get('num_beams', None)
        self.temperature = kwargs.get('temperature', None)
        self.top_p = kwargs.get('top_p', None)
        self.length_penalty = kwargs.get('length_penalty', None)
        self.repetition_penalty = kwargs.get('repetition_penalty', None)
        self.min_length = kwargs.get('min_length', None)
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        
        self.check_limits() 

        self.model = ModelRunner(
            self.model_path, rank=self.rank, world_size=self.world_size,
            local_rank=self.local_rank,
            is_flash_causal_lm=False,
            max_position_embeddings=self.max_position_embeddings,
            enable_atb_torch=self.enable_atb_torch,
            trust_remote_code=self.trust_remote_code
        )
        self.tokenizer = self.model.tokenizer
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.quantize = self.model.quantize
        self.model.load_weights()

        self.skip_word_embedding = False
        if hasattr(self.model.model, 'skip_word_embedding'):
            self.skip_word_embedding = self.model.model.skip_word_embedding
    
    def check_limits(self):
        if self.max_input_length <= 0:
            print_log(self.rank, logger.warning, 
                    f"'max_input_length' must be greater than 0, but is {self.max_input_length}. "
                    "Please increase the value of 'max_input_length'. ")
        if self.max_output_length <= 0:
            print_log(self.rank, logger.warning, 
                    f"'max_output_length' must be greater than 0, but is {self.max_output_length}. "
                    "Please increase the value of 'max_output_length'. ")
        
        if self.max_position_embeddings is not None and  \
           self.max_position_embeddings < self.max_input_length + self.max_output_length:
            print_log(self.rank, logger.warning, 
                    f"'max_position_embeddings' must be greater than " 
                    f"{self.max_input_length + self.max_output_length}, but is {self.max_position_embeddings}. "
                    "Please increase the value of 'max_position_embeddings'. ")

        if self.batch_size <= 0:
            print_log(self.rank, logger.warning, 
                    f"'batch_size' must be greater than 0, but is {self.batch_size}. "
                    "Please increase the value of 'batch_size'. ")  
        
        if self.num_beams is not None and self.num_beams <= 0:
            print_log(self.rank, logger.warning, 
                    f"'num_beams' must be greater than 0, but is {self.num_beams}. "
                    "Please increase the value of 'num_beams'. ")  
        
        if self.temperature is not None and self.temperature <= 0:
            print_log(self.rank, logger.warning, 
                    f"'temperature' must be greater than 0.0, but is {self.temperature}. "
                    "Please increase the value of 'temperature'. ")   
         
        if self.top_p is not None and \
            (self.top_p < 0 or self.top_p > 1):
            print_log(self.rank, logger.warning, 
                    f"The value of 'top_p' must be in range of [0, 1], but is {self.top_p}. "
                    "Please change the value of 'top_p'. ")    
        
        if self.length_penalty is not None and \
            (self.length_penalty < 0 or self.length_penalty > 2):
            print_log(self.rank, logger.warning, 
                    f"'length_penalty' must be in range of [0.0, 2.0], but is {self.length_penalty}. "
                    "Please change the value of 'length_penalty'. ")   
        
        if self.repetition_penalty is not None and \
            (self.repetition_penalty <= 0 or self.repetition_penalty > 2):
            print_log(self.rank, logger.warning, 
                    f"'repetition_penalty' must be in range of (0.0, 2.0], but is {self.repetition_penalty}. "
                    "Please change the value of 'repetition_penalty'. ")  
        
        if self.min_length is not None and \
            (self.min_length < 0 or self.min_length > self.max_input_length + self.max_output_length):
            print_log(self.rank, logger.warning, 
                    f"'min_length' must be in range of [0.0, `max_input_length` + `max_output_length`], "
                    f"but is {self.min_length}. "
                    "Please increase the value of 'min_length'. ")
               
    def warm_up(self):
        print_log(self.rank, logger.info, "---------------begin warm_up---------------")
        dummy_input_ids_full = torch.randint(
            0, 32000, [self.batch_size, self.max_input_length], dtype=torch.long).npu()
        self.model.generate(inputs=dummy_input_ids_full, do_sample=False, max_new_tokens=10)
        print_log(self.rank, logger.info, "---------------end warm_up---------------")

    def infer(self, input_text, attention_mask=None):
        print_log(self.rank, logger.info, "---------------begin inference---------------")
        if isinstance(input_text, str):
            input_text = [input_text] * self.batch_size
                            
        if isinstance(input_text[0], str):

            inputs = self.tokenizer(input_text, return_tensors="pt", padding='max_length',
                                    max_length=self.max_input_length,
                                    truncation=True)
            for item in inputs:
                if len(item) > self.max_input_length:
                    print_log(self.rank, logger.warning,
                                "Num of tokens in input_text is larger than max_input_length. "
                                "Please shorten input_text to avoid out of memory. ")                        

            prefill_start_time = time.time()
            with torch.no_grad():
                self.model.generate(
                    inputs=inputs.input_ids.npu(),
                    attention_mask=inputs.attention_mask.npu(),
                    max_new_tokens=1
                )
            prefill_end_time = time.time()

            decode_start_time = time.time()
            with torch.no_grad():
                generate_ids = self.model.generate(
                    inputs=inputs.input_ids.npu(),
                    attention_mask=inputs.attention_mask.npu(),
                    max_new_tokens=self.max_output_length
                )
            decode_end_time = time.time()

            generate_text = self.tokenizer.batch_decode(
                generate_ids[:, self.max_input_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            input_tokens_num = len(inputs.input_ids[0])
            generate_tokens_num = len(generate_ids[0]) - len(inputs.input_ids[0])

            logger.info(f'{inputs.input_ids.shape=}')
        else:
            inputs = input_text
            for item in inputs:
                if len(item) > self.max_input_length:
                    print_log(self.rank, logger.warning,
                                "Num of tokens in input_text is larger than max_input_length. "
                                "Please shorten input_text to avoid out of memory. ")   

            prefill_start_time = time.time()
            with torch.no_grad():
                generate_ids = self.model.generate(
                    inputs=inputs.npu(),
                    attention_mask=attention_mask.npu(),
                    max_new_tokens=1
                )
            prefill_end_time = time.time()

            decode_start_time = time.time()
            with torch.no_grad():
                generate_ids = self.model.generate(
                    inputs=inputs.npu(),
                    attention_mask=attention_mask.npu(),
                    max_new_tokens=self.max_output_length
                )
            decode_end_time = time.time()
            generate_text = self.tokenizer.batch_decode(
                generate_ids[:, self.max_input_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            input_tokens_num = len(inputs)
            generate_tokens_num = len(generate_ids) - len(inputs)

        print_log(self.rank, logger.info, f'Input token num: {input_tokens_num}')
        print_log(self.rank, logger.info, f'Generate token num: {generate_tokens_num}')
        print_log(self.rank, logger.info, f'Generate text: {generate_text}')
        print_log(self.rank, logger.info, "---------------end inference---------------")

        prefill_time = (prefill_end_time - prefill_start_time)
        e2e_time = (decode_end_time - prefill_start_time)
        try:
            decode_average_time = (decode_end_time - decode_start_time) / (self.max_output_length - 1)
        except ZeroDivisionError as e:
            raise ZeroDivisionError from e
        print_log(self.rank, logger.info,
            f"Prefill time: {prefill_time * 1000}ms, "
            f"Decode average time: {decode_average_time * 1000}ms, "
            f"E2E time: {e2e_time}s"
        )
        return generate_text, generate_tokens_num, e2e_time

    def infer_from_embeds(self, args):
        if rank == 0:
            logger.info("---------------begin inference---------------")

        stop_words_ids = [torch.tensor(ids).npu() for ids in json.loads(args.stop_words_ids)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        generation_args = {
            "inputs_embeds": None,
            'min_length': args.min_length,
            'max_new_tokens': args.max_output_length,
            "stopping_criteria": stopping_criteria,
            'do_sample': args.do_sample,
            'num_beams': args.num_beams,
            'top_p': args.top_p,
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
            'length_penalty': args.length_penalty,
        }

        image_answer_pairs = {}
        inputs_embeds_dir = file_utils.standardize_path(args.inputs_embeds_dir)
        file_utils.check_path_permission(inputs_embeds_dir)
        for inputs_embeds_file_path in sorted([os.path.join(inputs_embeds_dir, _)
                                               for _ in file_utils.safe_listdir(inputs_embeds_dir)]):

            if not inputs_embeds_file_path.endswith(".pt"):
                continue

            if rank == 0:
                logger.info(f'NO.{len(image_answer_pairs) + 1}')

            inputs_embeds_file_path = file_utils.standardize_path(inputs_embeds_file_path)
            file_utils.check_file_safety(inputs_embeds_file_path, 'r', is_check_file_size=False)
            inputs_embeds = torch.load(inputs_embeds_file_path, weights_only=True).npu()
            inputs_embeds = inputs_embeds.to(self.dtype)
            generation_args["inputs_embeds"] = inputs_embeds

            with torch.no_grad():
                generate_ids = self.model.generate(**generation_args)

            output_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            image_answer_pairs[inputs_embeds_file_path] = output_text

            if rank == 0:
                logger.info(f'Answer: {output_text}')
                with safe_open(args.results_save_path, "w", encoding='utf-8') as f:
                    json.dump(image_answer_pairs, f)
                logger.info('json dump finished')

        if rank == 0:
            logger.info("---------------end inference---------------")


if __name__ == '__main__':
    arguments = parse_arguments()

    rank = ENV.rank
    local_rank = ENV.local_rank
    world_size = ENV.world_size
    input_dict = {
        'rank': rank,
        'world_size': world_size,
        'local_rank': local_rank,
        **vars(arguments)
    }

    fa_runner = FARunner(**input_dict)

    if fa_runner.skip_word_embedding:
        fa_runner.infer_from_embeds(arguments)
    else:
        fa_runner.warm_up()
        fa_runner.infer(arguments.input_text)
