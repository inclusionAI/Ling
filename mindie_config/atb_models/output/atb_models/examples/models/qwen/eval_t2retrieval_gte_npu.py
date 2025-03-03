# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from typing import List, Any, Union
from collections import defaultdict
import importlib
import json
import numpy as np
import torch
from atb_llm.utils.log.logging import print_log
import transformers.tokenization_utils_base
from mteb import MTEB, AbsTaskRetrieval
from datasets import load_dataset, DatasetDict
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm as progressbar
from atb_llm.utils.env import ENV
from atb_llm.utils.file_utils import safe_open
from atb_llm.utils.log.logging import logger
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate LLM.')
    parser.add_argument(
        '--model_type_or_path',
        type=str,
        required=True,
        help='Specipy model type to load default model or path to the directory containing model file.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size of dataset for computing.'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=7,
        choices=list(range(8)),
        help='Adapt model on device id x.'
    )
    return parser.parse_args()


def load_retrieval_data(hf_hub_name, eval_splits):
    key_of_text = 'text'
    eval_split = eval_splits[0]
    dataset = load_dataset(
        'parquet',
        data_files={
            'corpus': './dataset/T2Retrieval/data/corpus-00000-of-00001-8afe7b7a7eca49e3.parquet',
            'queries': './dataset/T2Retrieval/data/queries-00000-of-00001-930bf3b805a80dd9.parquet'
    })
    qrels = load_dataset(
        'parquet',
        data_files={eval_split: './dataset/T2Retrieval-qrels/data/dev-00000-of-00001-92ed0416056ff7e1.parquet'}
    )[eval_split]

    corpus = {e['id']: {key_of_text: e[key_of_text]} for e in dataset['corpus']}
    queries = {e['id']: e[key_of_text] for e in dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['qid']][e['pid']] = e['score']

    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    return corpus, queries, relevant_docs


class T2RetrievalLocal(AbsTaskRetrieval):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.data_loaded = None
        self.corpus = None
        self.queries = None
        self.relevant_docs = None

    @property
    def description(self) -> dict:
        return {
            'name': 'T2RetrievalLocal',
            'description': 'T2Ranking: A large-scale Chinese Benchmark for Passage Ranking',
            'hf_hub_name': 'C-MTEB/T2Retrieval',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        try:
            self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
                self.description['hf_hub_name'],
                self.description['eval_splits']
            )
        except KeyError as e:
            raise RuntimeError('load dataset failed') from e
        else:
            self.data_loaded = True


class Model:
    def __init__(self, tokenizer_path: str, batch_size: int) -> None:
        self.tokenizer = safe_get_tokenizer_from_pretrained(tokenizer_path)
        self.batch_size = batch_size

    def encode(self, sentences: List[str], **kwargs: Any) -> torch.Tensor:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode

        Returns:
            `torch.Tensor`: Tensor of embeddings for the given sentences
        """
        pass



class PyTorchModel(Model):
    is_flash_causal_lm: bool = True
    dtype = 'float16'
    
    def __init__(self, tokenizer_path: str, model_path: str, batch_size: int, device_id:int):
        self.model_name_or_path = model_path
        self.npu_id = device_id
        super(PyTorchModel, self).__init__(tokenizer_path, batch_size)
        
        # init model runtime
        try:
            import torch_npu
        except ImportError:
            self.device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'npu:{}'.format(device_id)
            torch_npu.npu.set_device(device_id)
            torch_npu.npu.set_compile_mode(jit_compile=False)

        self.model_cls = self.get_model_cls()
        
        rank = ENV.rank
        local_rank = ENV.local_rank
        world_size = ENV.world_size
        self.model_path = self.model_name_or_path
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.max_position_embeddings = None
        from examples.run_pa import PARunner
        self.input_dict = {
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'model_path': self.model_path,
            'input_texts':["who are you?"],
            'input_ids': None,
            'input_file':None,
            'input_dict':None,
            'max_batch_size':50,
            'max_input_length':1536,
            'max_output_length':1,
            'max_position_embeddings':None,
            'max_prefill_tokens':-1,
            'block_size':128,
            'chat_template':None,
            'ignore_eos':False,
            'is_chat_model':False,
            'is_embedding_model':True,
            'load_tokenizer':True,
            'enable_atb_torch':False,
            'kw_args':''
            }                
        self.pa_runner = PARunner(**self.input_dict)
        print_log(rank, logger.info, f'pa_runner: {self.pa_runner}')
        self.pa_runner.warm_up()
      
    def get_model_cls(self):
        """
        get_model_cls
        """
        
        model_file_dir_name = f"atb_llm.models.qwen2."
        model_file_name = 'flash_causal' if self.is_flash_causal_lm else 'causal'
        module_path = f"{model_file_dir_name}{model_file_name}_qwen2_gte"
        module = importlib.import_module(module_path)
        model_cls_name = f"Qwen2ForCausalLM"
        if self.is_flash_causal_lm:
            model_cls_name = "Flash" + model_cls_name
        return getattr(module, model_cls_name)
  
    
    number = 0    
    
    def encode(self, sentences: List[str], **kwargs: Any) -> Union[np.ndarray, torch.Tensor]:
        all_embs = []

        for start_index in progressbar(range(0, len(sentences), self.batch_size)):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            
            sen = [sentences_batch[0][:512]]
            sen = sentences_batch
            with torch.no_grad():
                
                
                infer_params = {
                    "inputs": sen,
                    "batch_size": 1,
                    "max_output_length": 1,
                    "ignore_eos": False,
                    "is_chat_model": False
                }
                generate_texts, token_nums, _ = self.pa_runner.infer(**infer_params)
                file_path = '../../embedding_tensor/gte-qwen2/embedding_tensor_0.pth'
                embs = torch.load(file_path)
                sentence_embeddings = embs
            sentence_embeddings = sentence_embeddings.float()
            all_embs.append(sentence_embeddings.cpu())

        if all_embs:
            if isinstance(all_embs, np.ndarray):
                all_embs = torch.from_numpy(all_embs)
            else:
                all_embs = torch.stack(all_embs)
        else:
            all_embs = torch.Tensor()

        return all_embs


def load_model(model_args: argparse.Namespace) -> Model:
    model_path = args.model_type_or_path
    tokenizer_path = args.model_type_or_path
    model_for_eval = PyTorchModel(
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        batch_size=model_args.batch_size,
        device_id=model_args.device
    )
    return model_for_eval


if __name__ == '__main__':
    args = get_args()
    model = load_model(args)
    task = ['T2RetrievalLocal']
    evaluation = MTEB(tasks=task, task_langs=['zh'])
    results = evaluation.run(model)
    logger.info(results)
