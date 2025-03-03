# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""PyTorch BERT flash attention model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.models.bert.configuration_bert import BertConfig

from atb_llm.utils.initial import load_atb_speed
from atb_llm.models.embedding.modeling_embedding_utils import (
    BaseEmbeddings,
    BaseSelfAttention,
    BaseSelfOutput,
    BaseAttention,
    BaseIntermediate,
    BaseOutput,
    BaseLayer,
    BaseEncoder,
    BasePooler,
    BasePreTrainedModel,
    BaseModel,
    BaseForSequenceClassification,
    BASE_START_DOCSTRING,
    BASE_INPUTS_DOCSTRING
)


_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01


load_atb_speed()


class BertEmbeddings(BaseEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)


class BertSelfAttention(BaseSelfAttention):
    def __init__(self, config: BertConfig, position_embedding_type: Optional[str] = None) -> None:
        super().__init__(config, position_embedding_type)


class BertSelfOutput(BaseSelfOutput):
    def __init__(self, config: BertConfig):
        super().__init__(config)


class BertAttention(BaseAttention):
    def __init__(self, config: BertConfig, position_embedding_type: Optional[str] = None) -> None:
        super().__init__(config, position_embedding_type)


class BertIntermediate(BaseIntermediate):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)


class BertOutput(BaseOutput):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)


class BertLayer(BaseLayer):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)


class BertEncoder(BaseEncoder):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)


class BertPooler(BasePooler):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)


class BertPreTrainedModel(BasePreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True


BERT_START_DOCSTRING = BASE_START_DOCSTRING.replace("BaseConfig", _CONFIG_FOR_DOC)

BERT_INPUTS_DOCSTRING = BASE_INPUTS_DOCSTRING


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel, BaseModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need] by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as a decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: torch.Tensor) -> None:
        self.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**. past_key_values (`tuple(
            tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(
            batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and value
            hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        return self._forward(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            inputs_embeds,
            past_key_values,
            return_dict
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification(BertPreTrainedModel, BaseForSequenceClassification):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)

        loss = self._calc_loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
