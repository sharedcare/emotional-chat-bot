#!/usr/bin/python3

'''
Seq2Seq in Transformer, implemented by Pytorch's nn.Transformer
'''
import math
import random
import numpy as np
import pickle
import ipdb
import sys
import types
import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .layers import *


class Decoder(nn.Module):
    '''
    Add the multi-head attention for GRU
    '''

    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=2, dropout=0.5, nhead=8):
        super(Decoder, self).__init__()
        self.embed_size, self.hidden_size = embed_size, hidden_size
        self.output_size = output_size

        self.embed = nn.Embedding(output_size, embed_size)
        self.multi_head_attention = nn.ModuleList([Attention(hidden_size) for _ in range(nhead)])
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size + embed_size,
                          hidden_size,
                          num_layers=n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)
        self.ffn = nn.Linear(nhead * hidden_size, hidden_size)

        self.init_weight()

    def init_weight(self):
        # orthogonal inittor
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch]
        # last_hidden: [2, batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)  # [1, batch, embed_size]

        # attn_weights: [batch, 1, timestep of encoder_outputs]
        key = last_hidden.sum(axis=0)
        # calculate the attention
        context_collector = []
        for attention_head in self.multi_head_attention:
            attn_weights = attention_head(key, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            context = context.squeeze(1).transpose(0, 1)  # [hidden, batch]
            context_collector.append(context)  # [N, hidden, batch]
        context = torch.stack(context_collector).view(-1, context.shape[-1]).transpose(0, 1)  # [N, hidden, batch]
        # context = context.view(-1, context.shape[-1]).transpose(0, 1)    # [batch, N*hidden]
        context = torch.tanh(self.ffn(context)).unsqueeze(0)  # [1, batch, hidden]

        # context: [batch, 1, hidden_size]
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context = context.transpose(0, 1)

        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)
        # context = context.squeeze(0)
        # [batch, hidden * 2]
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output)  # [batch, output_size]
        output = F.log_softmax(output, dim=1)

        # output: [batch, output_size]
        # hidden: [2, batch, hidden_size]
        # hidden = hidden.squeeze(0)
        return output, hidden


class Transformer(nn.Module):
    '''
    Transformer encoder and GRU decoder

    Multi-head attention for GRU
    '''

    def __init__(self, input_vocab_size, opt_vocab_size, d_model, nhead,
                 num_encoder_layers, dim_feedforward, position_embed_size=300,
                 utter_n_layer=2, dropout=0.3, sos=0, pad=0, teach_force=1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.hidden_size = d_model
        self.embed_src = nn.Embedding(input_vocab_size, d_model)
        # position maxlen is 5000
        self.pos_enc = PositionEmbedding(d_model, dropout=dropout,
                                         max_len=position_embed_size)
        self.input_vocab_size = input_vocab_size
        self.utter_n_layer = utter_n_layer
        self.opt_vocab_size = opt_vocab_size
        self.pad, self.sos = pad, sos
        self.teach_force = teach_force

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_encoder_layers)

        self.decoder = Decoder(d_model, d_model, opt_vocab_size,
                               n_layers=utter_n_layer, dropout=dropout, nhead=nhead)

    def generate_key_mask(self, x, lengths):
        # x: [seq, batch]
        # return: key mask [batch, seq]
        seq_length = x.shape[0]
        masks = []
        for sentence_l in lengths:
            masks.append([False for _ in range(sentence_l)] + [True for _ in range(seq_length - sentence_l)])
        masks = torch.tensor(masks)
        if torch.cuda.is_available():
            masks = masks.cuda()
        return masks

    def forward(self, src, tgt, lengths):
        # src: [seq, batch], tgt: [seq, batch], lengths: [batch]
        batch_size, max_len = src.shape[1], tgt.shape[0]
        src_key_padding_mask = self.generate_key_mask(src, lengths)

        outputs = torch.zeros(max_len, batch_size, self.opt_vocab_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        # src: [seq, batch, d_model]
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        # memory: [seq, batch, d_model]
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # hidden: [2, batch, d_model]
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        output = tgt[0, :]

        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, memory)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, memory)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()

        # [max_len, batch, output_size]
        return outputs

    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            src_key_padding_mask = self.generate_key_mask(src, lengths)

            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.opt_vocab_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()

            # src: [seq, batch, d_model]
            src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
            # memory: [seq, batch, d_model]
            memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

            # hidden: [2, batch, d_model]
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()

            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, memory)
                floss[t] = output
                # output = torch.max(output, 1)[1]    # [1]
                output = output.topk(1)[1].squeeze()
                outputs[t] = output  # output: [1, output_size]

            if loss:
                return outputs, floss
            else:
                return outputs 
