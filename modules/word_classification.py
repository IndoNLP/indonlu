import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.modeling_xlm import XLMPreTrainedModel
from transformers import AlbertPreTrainedModel, BertPreTrainedModel, AlbertModel, BertModel, BertConfig, XLMModel, XLMConfig, XLMRobertaModel, XLMRobertaConfig
from transformers import AutoTokenizer, AutoConfig

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-pytorch_model.bin",
}

XLM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-mlm-en-2048": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-pytorch_model.bin",
    "xlm-mlm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-pytorch_model.bin",
    "xlm-mlm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-pytorch_model.bin",
    "xlm-mlm-enro-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-pytorch_model.bin",
    "xlm-mlm-tlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-pytorch_model.bin",
    "xlm-mlm-xnli15-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-pytorch_model.bin",
    "xlm-clm-enfr-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-pytorch_model.bin",
    "xlm-clm-ende-1024": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-pytorch_model.bin",
    "xlm-mlm-17-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-pytorch_model.bin",
    "xlm-mlm-100-1280": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-pytorch_model.bin",
}

class BertForWordClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subword_to_word_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # average the token-level outputs to compute word-level representations
        max_seq_len = subword_to_word_ids.max() + 1
        word_latents = []
        for i in range(max_seq_len):
            mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
            word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
        word_batch = torch.stack(word_latents, dim=1)

        sequence_output = self.dropout(word_batch)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class AlbertForWordClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subword_to_word_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        # average the token-level outputs to compute word-level representations
        max_seq_len = subword_to_word_ids.max() + 1
        word_latents = []
        for i in range(max_seq_len):
            mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
            word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
        word_batch = torch.stack(word_latents, dim=1)

        sequence_output = self.dropout(word_batch)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class XLMForWordClassification(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLMModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subword_to_word_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.XLMConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import XLMTokenizer, XLMForTokenClassification
        import torch
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
        model = XLMForTokenClassification.from_pretrained('xlm-mlm-100-1280')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
        """
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        # average the token-level outputs to compute word-level representations
        max_seq_len = subword_to_word_ids.max() + 1
        word_latents = []
        for i in range(max_seq_len):
            mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
            word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
        word_batch = torch.stack(word_latents, dim=1)

        sequence_output = self.dropout(word_batch)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class XLMRobertaForWordClassification(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        subword_to_word_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # average the token-level outputs to compute word-level representations
        max_seq_len = subword_to_word_ids.max() + 1
        word_latents = []
        for i in range(max_seq_len):
            mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
            word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
        word_batch = torch.stack(word_latents, dim=1)

        sequence_output = self.dropout(word_batch)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

if __name__ == '__main__':
    print("BertForWordClassification")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = BertForWordClassification.from_pretrained("bert-base-uncased", config=config)

    print("AlbertForWordClassification")
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    config = AutoConfig.from_pretrained("albert-base-v2")
    model = AlbertForWordClassification.from_pretrained("albert-base-v2", config=config)

    print("XLMForWordClassification")
    tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-100-1280")
    config = AutoConfig.from_pretrained("xlm-mlm-100-1280")
    model = XLMForWordClassification.from_pretrained("xlm-mlm-100-1280", config=config)

    print("XLMRobertaForWordClassification")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    config = AutoConfig.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForWordClassification.from_pretrained("xlm-roberta-base", config=config)