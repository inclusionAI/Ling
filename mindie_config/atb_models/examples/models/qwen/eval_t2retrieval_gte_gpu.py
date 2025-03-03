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
from typing import List, Any, Union
from collections import defaultdict
from venv import logger
import numpy as np
import torch
import transformers.tokenization_utils_base
from mteb import MTEB, AbsTaskRetrieval
from datasets import load_dataset, DatasetDict
from optimum.onnxruntime import ORTModelForFeatureExtraction
from tqdm import tqdm as progressbar
from atb_llm.models.base.model_utils import safe_get_tokenizer_from_pretrained
from atb_llm.models.base.model_utils import safe_get_auto_model_from_pretrained


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


class Model:
    def __init__(self, tokenizer_path: str, in_batch_size: int) -> None:
        self.tokenizer = safe_get_tokenizer_from_pretrained(
            tokenizer_path, 
            trust_remote_code=False
        )
        self.batch_size = in_batch_size

    def encode(self, sentences: List[str], **kwargs: Any) -> torch.Tensor:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode

        Returns:
            `torch.Tensor`: Tensor of embeddings for the given sentences
        """
        pass

    def _tokenize_sentences(self, sentences: List[str]) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=512
        )


class PyTorchModel(Model):
    def __init__(self, tokenizer_path: str, model_path: str, in_batch_size: int, device_id: int):
        super(PyTorchModel, self).__init__(tokenizer_path, in_batch_size)

        try:
            import torch_npu
        except ImportError:
            self.device = 'cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'npu:{}'.format(device_id)
            torch_npu.npu.set_device(device_id)
            torch_npu.npu.set_compile_mode(jit_compile=False)
        self.model = safe_get_auto_model_from_pretrained(
            model_path,
            trust_remote_code=False
        ).to(self.device)
        self.model.eval()

    def encode(self, sentences: List[str], **kwargs: Any) -> Union[np.ndarray, torch.Tensor]:
        all_embs = []

        for start_index in progressbar(range(0, len(sentences), self.batch_size)):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            # Tokenize sentences
            encoded_inputs = self._tokenize_sentences(sentences_batch).to(self.device)
            # Compute token embeddings
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state, encoded_inputs['attention_mask'])
                sentence_embeddings = embeddings.float()
            all_embs.extend(sentence_embeddings.cpu())

        if all_embs:
            if isinstance(all_embs, np.ndarray):
                all_embs = torch.from_numpy(all_embs)
            else:
                all_embs = torch.stack(all_embs)
        else:
            all_embs = torch.Tensor()
        return all_embs
    
    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            in_batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(in_batch_size, device=last_hidden_states.device), sequence_lengths]


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

    corpus = {e['id']: {key_of_text: e[key_of_text][:512]} for e in dataset['corpus']}
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
        self.metadata = None

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



if __name__ == '__main__':
    args = get_args()
    model_type_or_path = args.model_type_or_path
    batch_size = args.batch_size
    device = args.device
    model = PyTorchModel(model_type_or_path, model_type_or_path, batch_size, device)
    task = ['T2RetrievalLocal']
    evaluation = MTEB(tasks=task, task_langs=['zh'])
    results = evaluation.run(model)