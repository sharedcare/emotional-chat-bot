#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.2.24

'''
Functional Layers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import random
import numpy as np
from collections import Counter
import pickle
import ipdb


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        # self.v = nn.Parameter(torch.rand(hidden_size * 2))
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: from decoder, [batch, decoder_hidden_size]
        timestep = encoder_outputs.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [batch, timestep, decoder_hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch, timestep, encoder_hidden_size]

        # [batch, timestep]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch, 1, timestep]

    def score(self, hidden, encoder_outputs):
        # hidden: [batch, timestep, decoder_hidden_size]
        # encoder_outputs: [batch, timestep, encoder_hidden_size]
        # energy: [batch, timestep, hidden_size]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [batch, 2 * hidden_size, timestep]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [batch, 1, 2 * hidden_size]
        energy = torch.bmm(v, energy)  # [batch, 1, timestep]
        return energy.squeeze(1)  # [batch, timestep]


class Multi_head_attention(nn.Module):
    '''
    Multi head attention for RNN, Layernorm and residual connection are used.
    By the way, Transformer sucks.
    '''

    def __init__(self, hidden_size, nhead=4):
        super(Multi_head_attention, self).__init__()
        self.nhead = nhead
        self.hidden_size = hidden_size
        self.multi_head_attention = nn.ModuleList([Attention(hidden_size) for _ in range(nhead)])
        self.ffn = nn.Linear(self.nhead * self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden]
        # encoder_outputs: [seq, batch, hidden]
        # return: context [1, batch, seq]
        context_collector = []  # [N, hidden, batch]
        for attention_head in self.multi_head_attention:
            attn_weights = attention_head(hidden, encoder_outputs)  # [batch, 1 seq]
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [batch, 1, hidden]
            context = context.squeeze(1).transpose(0, 1)  # [hidden, batch]
            context_collector.append(context)
        # [batch, n*hidden]
        context = torch.stack(context_collector).view(-1, context.shape[-1]).transpose(0, 1)
        context = torch.tanh(self.ffn(context)).unsqueeze(0)  # [1, batch, hidden]


class Multi_head_attention_trs(nn.Module):
    '''
    make sure the hidden_size can be divisible by nhead
    Recommand: 512, 8

    1. Multi head attention for encoder hidden state
    2. Use the hidden state to query the context encoder
    '''

    def __init__(self, hidden_size, nhead=8, dropout=0.3):
        super(Multi_head_attention_trs, self).__init__()
        self.nhead = nhead
        self.hidden_size = hidden_size

        if hidden_size % nhead != 0:
            raise Exception(f'hidden_size must be divisble by nhead, but got {hidden_size}/{nhead}.')

        self.multi_head_attention = nn.MultiheadAttention(hidden_size, nhead)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.final_attn = Attention(hidden_size)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden]
        # encoder_outputs: [seq, batch, hidden]
        # return: context [1, batch, seq]

        # context: [seq, batch, hidden]
        context, _ = self.multi_head_attention(encoder_outputs,
                                               encoder_outputs,
                                               encoder_outputs)
        context = context + encoder_outputs
        context = torch.tanh(self.layer_norm(context))
        attn_weights = self.final_attn(hidden.unsqueeze(0), context)
        context = attn_weights.bmm(context.transpose(0, 1))
        context = context.transpose(0, 1)
        return context


class WSeq_attention(nn.Module):
    '''
    Cosine similarity defined in ACL 2017 paper:
    How to Make Context More Useful?
    An Empirical Study on context-Aware Neural Conversational Models

    mode: sum, concat is very hard to be implemented
    '''

    def __init__(self, mode='sum'):
        super(WSeq_attention, self).__init__()

    def forward(self, query, utterances):
        # query: [batch, hidden], utterances: [seq_len, batch, hidden]
        # cosine similarity
        utterances = utterances.permute(1, 2, 0)  # [batch, hidden, seq_len]
        query = query.reshape(query.shape[0], 1, query.shape[1])  # [batch, 1, hidden]
        p = torch.bmm(query, utterances).squeeze(1)  # [batch, seq_len]
        query_norm = query.squeeze(1).norm(dim=1)  # [batch]
        utterances_norm = utterances.norm(dim=1)  # [batch, seq_len]
        p = p / query_norm.reshape(-1, 1)
        p = p / utterances_norm  # [batch, seq_len]

        # softmax
        sq = torch.ones(p.shape[0], 1)
        if torch.cuda.is_available():
            sq = sq.cuda()
        p = torch.cat([p, sq], 1)  # [batch, seq_len + 1]
        p = F.softmax(p, dim=1)  # [batch, seq_len + 1]

        # mode for getting vector
        utterances = utterances.permute(0, 2, 1)  # [batch, seq_len, hidden]
        vector = torch.cat([utterances, query], 1)  # [batch, seq_len + 1, hidden]
        p = p.unsqueeze(1)  # [batch, 1, seq_len + 1]

        # p: [batch, 1, seq_len + 1], vector: [batch, seq_len + 1, hidden]
        vector = torch.bmm(p, vector).squeeze(1)  # [batch, hidden]

        # [batch, hidden]
        return vector


class PositionEmbedding(nn.Module):
    '''
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # not the parameters of the Module

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PretrainedEmbedding(nn.Module):
    '''
    Pretrained English BERT contextual word embeddings
    make sure the embedding size is the same as the embed_size setted in the model
    or the error will be thrown.
    '''

    def __init__(self, vocab_size, embed_size, path):
        super(PretrainedEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)

        # load pretrained embedding
        with open(path, 'rb') as f:
            emb = pickle.load(f)

        self.emb.weight.data.copy_(torch.from_numpy(emb))

    def forward(self, x):
        return self.emb(x)


def gen_nopeek_mask(length):
    # for transformer masking
    mask = torch.triu(torch.ones(length, length)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    if torch.cuda.is_available():
        mask = mask.cuda()

    return mask


# ========= BOS Loss ========== #
def to_bow(sentence, vocab_size, pad, sos, eos, unk):
    '''  Convert a sentence into a bag of words representation
    Args
        - sentence: a list of token ids
        - vocab_size: V
    Returns
        - bow: a integer vector of size V, numpy ndarray
    '''
    sentence = sentence.cpu().numpy()
    bow = Counter(sentence)
    # Remove special tokens
    bow[pad], bow[eos], bow[sos], bow[unk] = 0, 0, 0, 0
    x = np.zeros(vocab_size, dtype=np.int64)
    x[list(bow.keys())] = list(bow.values())
    x = torch.tensor(x, dtype=torch.long)
    return x


def bag_of_words_loss(bow_logits, target_bow, weight=None):
    ''' Calculate bag of words representation loss
    Args
        - bow_logits: [batch_size, vocab_size]
        - target_bow: [batch_size, vocab_size]
    '''
    log_probs = F.log_softmax(bow_logits, dim=1)  # [batch, vocab]
    target_distribution = target_bow / (target_bow.sum(1).view(-1, 1) + 1e-23) + 1e-23
    entropy = -(torch.log(target_distribution) * target_bow).sum()
    loss = -(log_probs * target_bow).sum() - entropy  # too big, affect original NLLLoss
    loss = loss / target_bow.sum()
    return loss


# ========== torch_geometric, optinonal =========
try:
    from torch_geometric.nn import GCNConv, GATConv, TopKPooling
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import MessagePassing


    class My_GatedGCN(MessagePassing):

        '''
        GCN with Gated mechanism
        Help with the tutorial of the pytorch_geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

        x_i^k = x_i^{k-1} + \eta \sum_{j\in N(i)} e_{ij} * GRU(x_i^{k-1}, x_j^{k-1})

        aggregation method use the `mean` (`add` is not good?)
        '''

        def __init__(self, in_channels, out_channels, kernel):
            super(My_GatedGCN, self).__init__(aggr='mean')

            # kernel is a Gated GRUCell
            self.rnn = kernel
            self.linear = nn.Linear(in_channels, out_channels)
            self.in_channels = in_channels
            self.out_channels = out_channels

        def forward(self, x, edge_index, edge_weight=None):
            # x: [N, in_channels], edge_index: [2, E]
            return self.propagate(edge_index, size=(x.size(0), x.size(0)),
                                  x=x, edge_weight=edge_weight)

        def message(self, x_i, x_j, edge_weight):
            # x_i has shape [E, in_channels]
            # x_j has shape [E, in_channels]
            # edge_weight has shape [E]
            x = self.rnn(x_i, x_j)  # [E, in_channels]
            return edge_weight.view(-1, 1) * x

        def update(self, aggr_out, x):
            # aggr_out has shape [N, in_channels]
            # x has shape [N, in_channels]
            aggr_out = aggr_out + x
            aggr_out = self.linear(aggr_out)  # [N, out_channels]
            return aggr_out

        def __repr__(self):
            return '{}(in_channels={}, out_channels={})'.format(
                self.__class__.__name__, self.in_channels, self.out_channels)


    class My_DoubleGatedGCN(MessagePassing):

        '''
        GCN with Gated mechanism
        Help with the tutorial of the pytorch_geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

        x_i^k = GRU(\sum_{j\in N(i)} e_{ij} * GRU(x_i^{k-1}, x_j^{k-1}), x_i^{k-1})

        aggregation method use the `mean` (`add` is not good?)
        '''

        def __init__(self, in_channels, out_channels, kernel1, kernel2):
            super(My_DoubleGatedGCN, self).__init__(aggr='mean')

            # kernel is a Gated GRUCell
            self.rnn1 = kernel1
            self.rnn2 = kernel2
            self.linear = nn.Linear(in_channels, out_channels)
            self.in_channels = in_channels
            self.out_channels = out_channels

        def forward(self, x, edge_index, edge_weight=None):
            # x: [N, in_channels], edge_index: [2, E]
            return self.propagate(edge_index, size=(x.size(0), x.size(0)),
                                  x=x, edge_weight=edge_weight)

        def message(self, x_i, x_j, edge_weight):
            # x_i has shape [E, in_channels]
            # x_j has shape [E, in_channels]
            # edge_weight has shape [E]
            x = self.rnn1(x_i, x_j)  # [E, in_channels]
            return edge_weight.view(-1, 1) * x

        def update(self, aggr_out, x):
            # aggr_out has shape [N, in_channels]
            # x has shape [N, in_channels]
            aggr_out = self.rnn2(aggr_out, x)
            aggr_out = self.linear(aggr_out)  # [N, out_channels]
            return aggr_out

        def __repr__(self):
            return '{}(in_channels={}, out_channels={})'.format(
                self.__class__.__name__, self.in_channels, self.out_channels)


    class My_GATRNNConv(nn.Module):

        '''
        GAT with Gated mechanism
        Help with the tutorial of the pytorch_geometric:
        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

        x_i^k = GRU(GAT(x_i^{k-1}, x_j^{k-1}), x_{i}^{k-1})
        '''

        def __init__(self, in_channels, out_channels, kernel, head=8, dropout=0.5):
            super(My_GATRNNConv, self).__init__()

            # kernel is a Gated GRUCell
            self.rnn = kernel  # [in_channel, out_channel]
            self.conv = GATConv(in_channels, in_channels, heads=head, dropout=dropout)
            self.compress = nn.Linear(in_channels * head, in_channels)
            self.in_channels = in_channels
            self.opt = nn.Linear(in_channels, out_channels)

        def forward(self, x, edge_index):
            # x: [node, in_channels]
            m = F.dropout(x, p=0.6)
            m = F.relu(self.conv(m, edge_index))  # [node, 8 * in_channels]
            m = F.relu(self.compress(m))  # [node, in_channels]
            x = torch.tanh(self.rnn(m, x))  # [node, in_channels]
            return self.opt(x)  # [node, out_channels]

        def __repr__(self):
            return '{}(in_channels={})'.format(
                self.__class__.__name__, self.in_channels)
except:
    print(f'[!] cannot find the module "torch_geometric", ignore it')
