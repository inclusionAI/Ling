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

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import C_MTEB
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from mteb import MTEB, AbsTaskRetrieval, AbsTaskReranking
from tqdm import tqdm
from transformers import BatchEncoding

from atb_llm.utils.file_utils import safe_open, standardize_path
from atb_llm.utils.log.logging import logger, message_filter
from atb_llm.utils.log.error_code import ErrorCode

from model_runner import ModelRunner, Arguments


AbsTaskReranking.evaluate = C_MTEB.tasks.evaluate


class T2RetrievalLocal(AbsTaskRetrieval):

    data_loaded = False
    dataset_path = None
    corpus = None
    queries = None
    relevant_docs = None

    @property
    def description(self) -> Dict:
        return {
            "name": "T2RetrievalLocal",
            "description": "T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
            "dataset": {
                "path": self.dataset_path,
                "revision": "8731a845f1bf500a4f111cf1070785c793d10e64",
                "qrel_revision": "1c83b8d1544e529875e3f6930f3a1fcf749a8e97",
            },
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
        }

    @staticmethod
    def load_retrieval_data(
            dataset_path: str,
            dataset_revision: str,
            qrel_revision: str,
            eval_splits: str
    ) -> Tuple[DatasetDict, DatasetDict, DatasetDict]:
        eval_split = eval_splits[0]
        dataset = load_dataset(dataset_path, revision=dataset_revision)
        qrels = load_dataset(dataset_path + "-qrels", revision=qrel_revision)[eval_split]

        key_of_text = "text"
        corpus = {e["id"]: {key_of_text: e[key_of_text]} for e in dataset["corpus"]}
        queries = {e["id"]: e[key_of_text] for e in dataset["queries"]}
        relevant_docs = defaultdict(dict)
        for e in qrels:
            relevant_docs[e["qid"]][e["pid"]] = e["score"]

        corpus = DatasetDict({eval_split: corpus})
        queries = DatasetDict({eval_split: queries})
        relevant_docs = DatasetDict({eval_split: relevant_docs})

        return corpus, queries, relevant_docs

    def load_data(self, **kwargs) -> None:
        key_of_dataset = "dataset"
        if self.data_loaded:
            return
        try:
            self.corpus, self.queries, self.relevant_docs = self.load_retrieval_data(
                self.description[key_of_dataset]["path"],
                self.description[key_of_dataset]["revision"],
                self.description[key_of_dataset]["qrel_revision"],
                self.description["eval_splits"]
            )
        except KeyError as e:
            error_msg = "load dataset failed"
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise RuntimeError(error_msg) from e
        else:
            self.data_loaded = True


class T2RerankingLocal(AbsTaskReranking):

    data_loaded = False
    dataset_path = None
    dataset = None

    @property
    def description(self) -> dict:
        return {
            "name": "T2RerankingLocal",
            "description": "T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
            "dataset": {
                "path": self.dataset_path,
                "revision": "76631901a18387f85eaa53e5450019b87ad58ef9",
            },
            "type": "Reranking",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh"],
            "main_score": "map",
        }

    def load_data(self, **kwargs) -> None:
        if self.data_loaded:
            return
        try:
            self.dataset = load_dataset(**self.description["dataset"])
        except KeyError as e:
            error_msg = "load dataset failed"
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise RuntimeError(error_msg) from e
        else:
            self.data_loaded = True


class PrecisionModelTest:
    def __init__(self, model: ModelRunner, batch_size: int, **kwargs: Any) -> None:
        self.model = model
        self.batch_size = batch_size
        self.tokenizer_args = kwargs

    def encode(self, sentences: List[str], **kwargs: Any) -> torch.Tensor:
        all_embs = []

        for start_index in tqdm(range(0, len(sentences), self.batch_size)):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            encoded_inputs = self.model.tokenize(sentences_batch, **self.tokenizer_args)
            sentence_embeddings = self.model.embed(encoded_inputs)
            all_embs.extend(sentence_embeddings.float())

        if all_embs:
            if isinstance(all_embs, np.ndarray):
                all_embs = torch.from_numpy(all_embs)
            else:
                all_embs = torch.stack(all_embs)
        else:
            all_embs = torch.Tensor()

        return all_embs

    def compute_score(self, sentence_pairs: Union[List[List[str]], Tuple[str, str]], **kwargs: Any) -> List[float]:
        if not isinstance(sentence_pairs, list):
            error_msg = "type of `sentence_pairs` is not `list`"
            logger.error(error_msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise TypeError(error_msg)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []

        for start_index in tqdm(range(0, len(sentence_pairs), self.batch_size), "Computing"):
            pairs_batch = sentence_pairs[start_index:start_index + self.batch_size]
            encoded_inputs = self.model.tokenize(pairs_batch, **self.tokenizer_args)
            scores = self.model.rerank(encoded_inputs)
            all_scores.extend(scores.float().numpy().tolist())

        return all_scores[0] if len(all_scores) == 1 else all_scores


class Tester:
    def __init__(self, **kwargs: Any) -> None:
        self.model = ModelRunner(
            str(kwargs.pop("model_name_or_path", None)),
            bool(kwargs.pop("trust_remote_code", False)),
            getattr(torch, kwargs.pop("torch_dtype", "float16")),
            torch.device(f"{kwargs.pop('device_type', 'cpu')}:{str(kwargs.pop('device_id', 0))}"),
            str(kwargs.pop("model_type", None))
        )

        self.model_name_or_path = self.model.model_name_or_path
        self.torch_dtype = self.model.torch_dtype
        self.device = self.model.device
        self.model_type = self.model.model_type
        self.config = self.model.config
        self.tokenizer = self.model.tokenizer

        self.batch_size = kwargs.pop("batch_size", 20)
        self.loop = kwargs.pop("loop", 100)
        self.return_tensors = kwargs.pop("return_tensors", "pt")
        self.max_seq_len = kwargs.pop("max_seq_len", self.tokenizer.model_max_length)

        self.tokenizer_args = {
            "padding": kwargs.pop("padding", "max_length"),
            "truncation": kwargs.pop("truncation", True),
            "return_tensors": self.return_tensors,
            "max_length": self.max_seq_len,
        }

    def performance_test(self, output_folder: Optional[str]) -> Dict:
        inputs = self.model.generate_inputs(
            self.tokenizer.model_input_names,
            self.tokenizer.vocab_size,
            self.batch_size,
            self.max_seq_len,
            self.device
        )
        encoded_inputs = BatchEncoding(inputs, tensor_type=self.return_tensors)

        total_time = []
        inference_time = []
        transfer_time = []

        logger.info("---------------begin performance_test---------------")
        for _ in tqdm(range(self.loop + 1)):
            tick = time.perf_counter()
            model_outputs = self.model.forward(encoded_inputs)
            midpoint = time.perf_counter()
            torch.nn.functional.normalize(model_outputs[0]).view(-1, ).float().cpu()
            tock = time.perf_counter()
            total_time.append(1000 * (tock - tick))
            inference_time.append(1000 * (midpoint - tick))
            transfer_time.append(1000 * (tock - midpoint))
        logger.info("---------------end performance_test---------------")

        try:
            first_token_throughput = 1000 * self.batch_size / total_time[0]
            non_first_token_throughput = 1000 * self.batch_size / np.mean(total_time[1:])
        except IndexError as e:
            error_msg = f"performance_test uncompleted, {len(total_time)} results recorded"
            logger.error(error_msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError(error_msg) from e
        except ZeroDivisionError as e:
            error_msg = f"first token time {total_time[0]} ms or non-first token time {np.mean(total_time[1:])} ms is 0"
            logger.error(error_msg, ErrorCode.ATB_MODELS_EXECUTION_FAILURE)
            raise RuntimeError(error_msg) from e
        performance_test_results = {
            "infer_shape": {
                "batch_size": self.batch_size,
                "seq_len": self.max_seq_len
            },
            "first_token": {
                "time (ms)": {
                    "total_time": total_time[0],
                    "inference_time": inference_time[0],
                    "transfer_time": transfer_time[0],
                },
                "throughput (tokens/s)": first_token_throughput
            },
            "non_first_token": {
                "time (ms)": {
                    "total_time": np.mean(total_time[1:]),
                    "inference_time": np.mean(inference_time[1:]),
                    "transfer_time": np.mean(transfer_time[1:]),
                },
                "throughput (tokens/s)": non_first_token_throughput
            }
        }
        if output_folder:
            filename = f"performance_batch{self.batch_size}_seqlen{self.max_seq_len}_result.json"
            with safe_open(os.path.join(output_folder, filename), "w") as f:
                json.dump(performance_test_results, f, indent=2, ensure_ascii=True)

        return performance_test_results

    def precision_test(self, task: str, output_folder: Optional[str]) -> Dict:
        task_map = {
            "retrieval": "T2RetrievalLocal",
            "reranking": "T2RerankingLocal"
        }
        if task not in task_map:
            error_msg = f"task {message_filter(task)} not supported"
            logger.error(error_msg, ErrorCode.ATB_MODELS_MODEL_PARAM_JSON_INVALID)
            raise ValueError(error_msg)

        evaluation = MTEB(tasks=[task_map.get(task)], task_langs=["zh"])
        precision_tester = PrecisionModelTest(self.model, self.batch_size, **self.tokenizer_args)

        logger.info("---------------begin precision_test---------------")
        precision_test_results = evaluation.run(precision_tester, output_folder=output_folder)
        logger.info("---------------end precision_test---------------")

        return precision_test_results


if __name__ == "__main__":
    args = Arguments.set_tester_args()
    tester_args = {**{k: v for k, v in vars(args).items() if v}}
    logger.info(f"tester_args: {tester_args}")

    dataset_path = standardize_path(args.dataset_path, check_link=False)
    T2RetrievalLocal.dataset_path = dataset_path
    T2RerankingLocal.dataset_path = dataset_path
    tester = Tester(**tester_args)

    if args.task == "performance":
        results = tester.performance_test(output_folder=None)
        logger.info(f"performance results: {results}")
    elif args.task == "retrieval" or args.task == "reranking":
        outputs = f"{args.outputs}/precision/{args.model_name_or_path.split('/')[-1]}"
        results = tester.precision_test(args.task, outputs)
        logger.info(f"{message_filter(args.task)} results: {results}")
    else:
        task_error_msg = f"Task {message_filter(args.task)} not supported."
        logger.error(task_error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError(task_error_msg)
