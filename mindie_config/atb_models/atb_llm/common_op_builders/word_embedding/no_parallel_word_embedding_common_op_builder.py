# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import _libatb_torch as atb

from atb_llm.common_op_builders.word_embedding.base_word_embedding_common_op_builder import \
    BaseWordEmbeddingCommonOpBuilder
from atb_llm.utils.singleton import Singleton


class NoParallelWordEmbeddingCommonOpBuilder(BaseWordEmbeddingCommonOpBuilder, Singleton):
    def __init__(self):
        super().__init__()
        super(Singleton, self).__init__()

    def is_match(self, param: dict):
        if not super().is_match(param):
            return False
        if self.param.enable_parallel and self.param.parallel_info.world_size > 1:
            return False
        return True

    def build(self, graph: atb._GraphOperation, tensor_map: dict = None) -> None:
        super().build(graph, tensor_map)

        # gather 算子
        super().add_gather(graph, self.out_tensor_key.word_embedding_out)