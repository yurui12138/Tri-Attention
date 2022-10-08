# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: 31906
"""
import math
import os
import torch
from torch import nn, unsqueeze, squeeze
# import SentencePiece
from torch.nn import AvgPool1d, CrossEntropyLoss, AdaptiveMaxPool1d, AdaptiveAvgPool1d

from pytorch_pretrained_bert import modeling_mutil
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

'''dot simi'''
class Multil_Atten(nn.Module):
    def __init__(self): #config
        super(Multil_Atten, self).__init__()
        self.num_attention_heads = 12#config.num_attention_heads
        self.attention_head_size = int(768/12) #int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(768, self.all_head_size).cuda()    #config.hidden_size
        self.key = nn.Linear(768, self.all_head_size).cuda()
        self.value = nn.Linear(768, self.all_head_size).cuda()
        self.context = nn.Linear(768, self.all_head_size).cuda()

        self.add_dense = nn.Linear(64, 1).cuda()

        self.dropout = nn.Dropout(0.1)     #config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, sen1, sen2, context_sens=None, attention_mask=None):
        mixed_query_layer = self.query(sen1)
        mixed_key_layer = self.key(sen2)
        mixed_value_layer = self.value(sen2)
        mixed_context_layer = self.context(context_sens)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))       # bs, 12, 30, 30*64
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)       # bs, 12, 30, 30*64
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer = context_layer + mixed_context_layer
        context_layer = unsqueeze(context_layer, dim=2) + unsqueeze(mixed_context_layer, dim=1)
        context_layer = squeeze(self.add_dense(torch.transpose(context_layer, dim0=-1, dim1=-2)), dim=-1)
        return context_layer



class BertModel(nn.Module):
    def __init__(self, requires_grad=True, num_labels=2, depth=3):
        super(BertModel, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForSequenceClassification.from_pretrained('/mnt/public/home/lvwp/process/SPM/pytorch_pretrained_bert/bert-base-chinese',
                                                                  num_labels=self.num_labels)
        self.bert_one = modeling_mutil.BertForSequenceClassification.from_pretrained('/mnt/public/home/lvwp/process/SPM/pytorch_pretrained_bert/bert-base-chinese',
                                                                  num_labels=self.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained('/mnt/public/home/lvwp/process/SPM/pytorch_pretrained_bert/bert-base-chinese',
                                                       do_lower_case=True)
        self.multi_atten = []
        self.context_feed = []
        for i in range(depth):
            self.multi_atten.append(Multil_Atten())
            self.context_feed.append(nn.Linear(768, 768).cuda())
        self.drop = nn.Dropout(0.1)
        self.pool = AvgPool1d(kernel_size=30)
        self.context_pool = AdaptiveAvgPool1d(30)

        self.classifier = nn.Linear(768*4, num_labels).cuda()

        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # 每个参数都要求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, batch_seqs_s1, batch_seq_masks_s1, batch_seqs_s2, batch_seq_masks_s2, labels, t, depth=3):
        context_sentences, pooled_output = self.bert(input_ids=batch_seqs, attention_mask=batch_seq_masks,
                                 token_type_ids=batch_seq_segments, labels=labels, output_all_encoded_layers=False)

        sen1_embed, _ = self.bert_one(input_ids=batch_seqs_s1, attention_mask=batch_seq_masks_s1, labels=labels, embed_is=False)
        sen2_embed, _ = self.bert_one(input_ids=batch_seqs_s2, attention_mask=batch_seq_masks_s2, labels=labels, embed_is=False)

        # context_sentences_drop = torch.transpose(self.context_pool(torch.transpose(context_sentences, dim0=1, dim1=2)), dim0=1, dim1=2)
        # context_depth = [3,7,11]        #
        for i in range(depth):
            sen1_embed_ = self.multi_atten[i](sen1_embed, sen2_embed, context_sentences)
            sen2_embed_ = self.multi_atten[i](sen2_embed, sen1_embed, context_sentences)
            # sen1_embed_can = sen1_embed + sen1_embed_
            # sen2_embed_can = sen2_embed + sen2_embed_
            sen1_embed, sen2_embed = sen1_embed_, sen2_embed_
            context_sentences = self.drop(nn.GELU()(self.context_feed[i](context_sentences)))           #nn.GELU()(self.context_feed[i](

        sen1_embed = torch.transpose(sen1_embed, dim0=1, dim1=2)
        sen2_embed = torch.transpose(sen2_embed, dim0=1, dim1=2)
        sen1_embed = torch.squeeze(self.pool(sen1_embed), dim=-1)
        sen2_embed = torch.squeeze(self.pool(sen2_embed), dim=-1)

        sen1_sen2 = torch.abs(sen1_embed - sen2_embed)

        feature = torch.cat([sen1_embed, sen2_embed, sen1_sen2, pooled_output], dim=-1)

        pooled_output = self.drop(feature)
        logits = self.classifier(pooled_output)

        probabilities = nn.functional.softmax(logits, dim=-1)

        logits = logits / t

        # if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits, probabilities