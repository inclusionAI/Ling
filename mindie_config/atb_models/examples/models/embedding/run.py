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

import time
from typing import Any, List, Union

import torch
from transformers.tokenization_utils_base import BatchEncoding

from atb_llm.utils.log.logging import logger, message_filter
from atb_llm.utils.log.error_code import ErrorCode

from model_runner import ModelRunner, Arguments


class Runner:
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

        self.max_batch_size = kwargs.pop("max_batch_size", 1)
        self.max_seq_len = kwargs.pop("max_seq_len", self.tokenizer.model_max_length)
        self.return_tensors = kwargs.pop("return_tensors", "pt")

        self.tokenizer_args = {
            "padding": kwargs.pop("padding", "max_length"),
            "truncation": kwargs.pop("truncation", True),
            "return_tensors": self.return_tensors,
            "max_length": self.max_seq_len,
        }

    def warm_up(self) -> None:
        inputs = self.model.generate_inputs(
            self.tokenizer.model_input_names,
            self.tokenizer.vocab_size,
            self.max_batch_size,
            self.max_seq_len,
            self.device
        )
        encoded_inputs = BatchEncoding(inputs, tensor_type=self.return_tensors)

        logger.info("---------------begin warm_up---------------")
        tick = time.perf_counter()
        model_outputs = self.model.forward(encoded_inputs)
        torch.nn.functional.normalize(model_outputs[0]).view(-1, ).float().cpu()
        tock = time.perf_counter()
        logger.info("---------------end warm_up---------------")
        logger.info(f"warm time: {(tock - tick) * 1000:.2f} ms")

    def embed(self, texts: Union[str, List[str], List[List[str]]]) -> torch.Tensor:
        encoded_inputs = self.model.tokenize(texts, **self.tokenizer_args)
        logger.info("---------------begin embed---------------")
        tick = time.perf_counter()
        embeddings = self.model.embed(encoded_inputs)
        tock = time.perf_counter()
        logger.info("---------------end embed---------------")
        logger.info(f"embed time: {(tock - tick) * 1000:.2f} ms")
        return embeddings

    def rerank(self, texts: Union[str, List[str], List[List[str]]]) -> torch.Tensor:
        encoded_inputs = self.model.tokenize(texts, **self.tokenizer_args)
        logger.info("---------------begin rerank---------------")
        tick = time.perf_counter()
        scores = self.model.rerank(encoded_inputs)
        tock = time.perf_counter()
        logger.info("---------------end rerank---------------")
        logger.info(f"rerank time: {(tock - tick) * 1000:.2f} ms")
        return scores


if __name__ == "__main__":
    args = Arguments.set_runner_args()
    runner_args = {**{k: v for k, v in vars(args).items() if v}}
    logger.info(f"runner_args: {runner_args}")

    runner = Runner(**runner_args)
    runner.warm_up()

    if args.request == "embed":
        results = runner.embed(args.texts)
        for text, result in zip(args.texts, results):
            logger.info(f"Text: {message_filter(text)}")
            logger.info(f"Embeddings: {result}")
    elif args.request == "rerank":
        results = runner.rerank([text.split(",") if "," in text else text for text in args.texts])
        for text, result in zip(args.texts, results):
            logger.info(f"Text: {message_filter(text)}")
            logger.info(f"Embeddings: {result}")
    else:
        request_error_msg = f"Request {message_filter(args.request)} not supported."
        logger.error(request_error_msg, ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE)
        raise ValueError(request_error_msg)
