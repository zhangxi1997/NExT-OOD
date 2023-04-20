import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# from dataset import *
import torch.nn.init as init
from .PSAC_model_utils import *
from torch.nn.utils.weight_norm import weight_norm
import random

Lv = 100
Lq = 41 # 41 for single bert, 37 for base or original bert
Lc = 16 # 36
ctx_dim = 2048
ctx_dim_m=512

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, ntoken_c, emb_dim, c_emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken, emb_dim, padding_idx=0)
        self.c_emb = nn.Embedding(ntoken_c, c_emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.ntoken_c = ntoken_c
        self.emb_dim = emb_dim
        self.c_emb_dim = c_emb_dim


    def init_embedding(self,dict, glove_file, task):
        if not os.path.exists('./data/%s_glove6b_init_300d.npy'%task):
            print('Construct initial embedding...')
            weight_init, word2emb = create_glove_embedding_init(dict.idx2word, glove_file)
            np.save(os.path.join('./data','%s_glove6b_init_300d.npy'% task), weight_init)
            weight_init = torch.from_numpy(weight_init)
            weight_init_char = torch.from_numpy(np.random.normal(loc=0.0, scale=1, size=(self.ntoken_c, self.c_emb_dim)))
            np.save(os.path.join('./data','%s_char_glove6b_init_300d.npy'% task), weight_init_char)
        else:
            print('loading glove from ./data/%s_glove6b_init_300d.npy'%task)
            weight_init = torch.from_numpy(np.load('./data/%s_glove6b_init_300d.npy'%task))
            weight_init_char = torch.from_numpy(np.load('./data/%s_char_glove6b_init_300d.npy'%task))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        assert weight_init_char.shape == (self.ntoken_c, self.c_emb_dim)
        # self.emb.weight.data[:self.ntoken] = weight_init
        # self.c_emb.weight.data[:self.ntoken_c] = weight_init_char
        return weight_init, weight_init_char


    def forward(self, x, x_c):
        emb = self.emb(x)
        emb = self.dropout(emb)
        emb_c = self.c_emb(x_c)
        emb_c = self.dropout(emb_c)
        return emb, emb_c


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, out = 'last_layer', rnn_type='LSTM'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.out = out
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1 and self.out == 'last_layer':
            return output[:, -1]
        else:
            return output

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 8, 512, 64, 64
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask) # mb x len_v x d_model
        enc_output = self.pos_ffn(enc_output) # batch_size x v_len x ctx_dim
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, n_layer=6, n_head=8, d_k=64, d_v=64, v_len=36, v_emb_dim=300,
                 d_model=2048, d_inner_hid=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.d_model= d_model
        self.position_enc = nn.Embedding(v_len, v_emb_dim)
        self.position_enc.weight.data = position_encoding_init(v_len, v_emb_dim)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout)
                                          for _ in range(n_layer)])
        self.pos_linear = nn.Linear(300, 2048)

    def forward(self, src_seq, return_attns=False): # src_seq: batch_size x steps x ctx_dim
        # visual info
        # step 1: position embedding
        seq_batch_size, seq_len, v_feat_dim = src_seq.size() # batch_size:128  steps:35  ctx_dim:2048
        seq_mask = get_v_mask(seq_batch_size, seq_len).cuda() # batch_size x steps : position mask

        pos_emb = self.position_enc(seq_mask) # batch_size x v_len x v_emb_dim
        # print('ok')
        # print(pos_emb)
        # print(pos_emb.shape)
        pos_emb = self.pos_linear(pos_emb)
        enc_input = src_seq + pos_emb  # position embedding error
        # enc_input = src_seq   # no position embedding
        if return_attns:
            enc_slf_attns = []

        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq) # batch_size x v_len
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(   # batch_size x v_len x d_v
                enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output

class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_word, out_word, k, dim=1, bias=True):
        super(DepthwiseSeperableConv, self).__init__()
        if dim ==1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_word, out_channels=in_word, kernel_size=k, groups=in_word, padding=k//2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_word, out_channels=out_word, kernel_size=1, padding=0, bias=bias)
        elif dim ==2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_word, out_channels=in_word, kernel_size=k, groups=in_word, padding=k//2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_word, out_channels=out_word, kernel_size=1, padding=0,
                                            bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))

class VQAttention(nn.Module):
    def __init__(self):
        super(VQAttention, self).__init__()
        w4V = torch.empty(ctx_dim_m, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, ctx_dim_m)
        nn.init.xavier_uniform_(w4V)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4V = nn.Parameter(w4V)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        self.trans = weight_norm(nn.Linear(ctx_dim, ctx_dim_m))
        # self.trans = Initialized_Conv1d(ctx_dim, ctx_dim_m)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, Vid_enc, Ques_enc, V_mask, Q_mask):
        # Vid_enc = self.trans(Vid_enc.transpose(1, 2))
        Vid_enc = self.trans(Vid_enc)
        Ques_enc = Ques_enc.transpose(1, 2)
        batch_size = Vid_enc.size()[0]
        # Vid_enc = Vid_enc.transpose(1,2)
        S = self.trilinear_for_attention(Vid_enc, Ques_enc)
        V_mask = V_mask.view(batch_size, Lc, 1)
        Q_mask = Q_mask.view(batch_size, 1, Lq)
        S1 = F.softmax(mask_logits(S, Q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, V_mask), dim=1)
        A = torch.bmm(S1, Ques_enc)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), Vid_enc)
        out = torch.cat([Vid_enc, A, torch.mul(Vid_enc, A), torch.mul(Vid_enc, B)], dim=2)
        return out.transpose(1, 2), Vid_enc


    def trilinear_for_attention(self, Vid_enc, Ques_enc):
        V = F.dropout(Vid_enc, p=dropout, training=self.training)
        Q = F.dropout(Ques_enc, p=dropout, training=self.training)
        subres0 = torch.matmul(V, self.w4V).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1 ])
        subres2 = torch.matmul(V * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class VQAttention_counter(nn.Module):
    def __init__(self):
        super(VQAttention_counter, self).__init__()
        w4V = torch.empty(ctx_dim_m, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, ctx_dim_m)
        nn.init.xavier_uniform_(w4V)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4V = nn.Parameter(w4V)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        self.trans = weight_norm(nn.Linear(ctx_dim, ctx_dim_m))
        # self.trans = Initialized_Conv1d(ctx_dim, ctx_dim_m)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, Vid_enc, Ques_enc, V_mask, Q_mask):
        # Vid_enc = self.trans(Vid_enc.transpose(1, 2))
        Vid_enc = self.trans(Vid_enc)
        Ques_enc = Ques_enc.transpose(1, 2)  # [bs, 37, 512]
        batch_size = Vid_enc.size()[0]  # [bs, 16, 512]
        # Vid_enc = Vid_enc.transpose(1,2)
        S = self.trilinear_for_attention(Vid_enc, Ques_enc) # [bs, 16, 37]

        V_mask = V_mask.view(batch_size, Lc, 1)
        Q_mask = Q_mask.view(batch_size, 1, Lq)
        S1 = F.softmax(mask_logits(S, Q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, V_mask), dim=1)

        A = torch.bmm(S1, Ques_enc)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), Vid_enc)
        out = torch.cat([Vid_enc, A, torch.mul(Vid_enc, A), torch.mul(Vid_enc, B)], dim=2)

        # ===== counterfactual =====
        S_counter = torch.zeros_like(S).uniform_(0, 1)  # random
        S1_counter = F.softmax(mask_logits(S_counter, Q_mask), dim=2)
        S2_counter = F.softmax(mask_logits(S_counter, V_mask), dim=1)

        A_counter = torch.bmm(S1_counter, Ques_enc)
        B_counter = torch.bmm(torch.bmm(S1_counter, S2_counter.transpose(1,2)), Vid_enc)
        # the residual
        out_counter = torch.cat([Vid_enc, A-A_counter, torch.mul(Vid_enc, A-A_counter), torch.mul(Vid_enc, B-B_counter)], dim=2)

        return out.transpose(1, 2), Vid_enc, out_counter.transpose(1,2)


    def trilinear_for_attention(self, Vid_enc, Ques_enc):
        V = F.dropout(Vid_enc, p=dropout, training=self.training)
        Q = F.dropout(Ques_enc, p=dropout, training=self.training)
        subres0 = torch.matmul(V, self.w4V).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1 ])
        subres2 = torch.matmul(V * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class VQAttention_counter_concat(nn.Module):
    def __init__(self):
        super(VQAttention_counter_concat, self).__init__()
        w4V = torch.empty(ctx_dim_m, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, ctx_dim_m)
        nn.init.xavier_uniform_(w4V)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4V = nn.Parameter(w4V)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        self.trans = weight_norm(nn.Linear(ctx_dim, ctx_dim_m))
        # self.trans = Initialized_Conv1d(ctx_dim, ctx_dim_m)

        self.fcA = nn.Linear(3*ctx_dim_m, ctx_dim_m)
        self.fcB = nn.Linear(3 * ctx_dim_m, ctx_dim_m)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, Vid_enc, Ques_enc, V_mask, Q_mask):
        # Vid_enc = self.trans(Vid_enc.transpose(1, 2))
        Vid_enc = self.trans(Vid_enc)
        Ques_enc = Ques_enc.transpose(1, 2)  # [bs, 37, 512]
        batch_size = Vid_enc.size()[0]  # [bs, 16, 512]
        # Vid_enc = Vid_enc.transpose(1,2)
        S = self.trilinear_for_attention(Vid_enc, Ques_enc) # [bs, 16, 37]

        V_mask = V_mask.view(batch_size, Lc, 1)
        Q_mask = Q_mask.view(batch_size, 1, Lq)
        S1 = F.softmax(mask_logits(S, Q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, V_mask), dim=1)

        A = torch.bmm(S1, Ques_enc)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), Vid_enc)
        out = torch.cat([Vid_enc, A, torch.mul(Vid_enc, A), torch.mul(Vid_enc, B)], dim=2)


        # ===== counterfactual =====
        S_counter = torch.zeros_like(S).uniform_(0, 1)  # random
        S1_counter = F.softmax(mask_logits(S_counter, Q_mask), dim=2)
        S2_counter = F.softmax(mask_logits(S_counter, V_mask), dim=1)

        A_counter = torch.bmm(S1_counter, Ques_enc)
        B_counter = torch.bmm(torch.bmm(S1_counter, S2_counter.transpose(1,2)), Vid_enc)

        A_counter_ = torch.cat([A - A_counter, A, A_counter], dim=-1)
        A_counter_ = self.fcA(A_counter_)
        B_counter_ = torch.cat([B - B_counter, B, B_counter], dim=-1)
        B_counter_ = self.fcB(B_counter_)

        out_counter = torch.cat([Vid_enc, A_counter_, torch.mul(Vid_enc, A_counter_), torch.mul(Vid_enc, B_counter_)], dim=2)

        return out.transpose(1, 2), Vid_enc, out_counter.transpose(1,2)


    def trilinear_for_attention(self, Vid_enc, Ques_enc):
        V = F.dropout(Vid_enc, p=dropout, training=self.training)
        Q = F.dropout(Ques_enc, p=dropout, training=self.training)
        subres0 = torch.matmul(V, self.w4V).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1 ])
        subres2 = torch.matmul(V * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class VQAttention_counter2(nn.Module):
    def __init__(self):
        super(VQAttention_counter2, self).__init__()
        w4V = torch.empty(ctx_dim_m, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, ctx_dim_m)
        nn.init.xavier_uniform_(w4V)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4V = nn.Parameter(w4V)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        self.trans = weight_norm(nn.Linear(ctx_dim, ctx_dim_m))
        # self.trans = Initialized_Conv1d(ctx_dim, ctx_dim_m)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, Vid_enc, Ques_enc, V_mask, Q_mask):
        # Vid_enc = self.trans(Vid_enc.transpose(1, 2))
        Vid_enc = self.trans(Vid_enc)
        Ques_enc = Ques_enc.transpose(1, 2)  # [bs, 37, 512]
        batch_size = Vid_enc.size()[0]  # [bs, 16, 512]
        # Vid_enc = Vid_enc.transpose(1,2)
        S = self.trilinear_for_attention(Vid_enc, Ques_enc) # [bs, 16, 37]

        V_mask = V_mask.view(batch_size, Lc, 1)
        Q_mask = Q_mask.view(batch_size, 1, Lq)
        S1 = F.softmax(mask_logits(S, Q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, V_mask), dim=1)

        A = torch.bmm(S1, Ques_enc)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), Vid_enc)
        out = torch.cat([Vid_enc, A, torch.mul(Vid_enc, A), torch.mul(Vid_enc, B)], dim=2)

        # ===== counterfactual =====
        S1_counter = torch.zeros_like(S1).uniform_(0, 1)  # random
        # S1_counter = mask_logits(S1_counter, Q_mask)
        S1_counter = S1_counter * Q_mask
        S2_counter = torch.zeros_like(S2).uniform_(0, 1)
        # S2_counter = mask_logits(S2_counter, V_mask)
        S2_counter = S2_counter * V_mask
        # S1_counter = F.softmax(mask_logits(S_counter, Q_mask), dim=2)
        # S2_counter = F.softmax(mask_logits(S_counter, V_mask), dim=1)

        A_counter = torch.bmm(S1_counter, Ques_enc)
        B_counter = torch.bmm(torch.bmm(S1_counter, S2_counter.transpose(1,2)), Vid_enc)
        # the residual
        out_counter = torch.cat([Vid_enc, A-A_counter, torch.mul(Vid_enc, A-A_counter), torch.mul(Vid_enc, B-B_counter)], dim=2)

        return out.transpose(1, 2), Vid_enc, out_counter.transpose(1,2)


    def trilinear_for_attention(self, Vid_enc, Ques_enc):
        V = F.dropout(Vid_enc, p=dropout, training=self.training)
        Q = F.dropout(Ques_enc, p=dropout, training=self.training)
        subres0 = torch.matmul(V, self.w4V).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1 ])
        subres2 = torch.matmul(V * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class VQAttention_counter3(nn.Module):
    def __init__(self):
        super(VQAttention_counter3, self).__init__()
        w4V = torch.empty(ctx_dim_m, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, ctx_dim_m)
        nn.init.xavier_uniform_(w4V)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4V = nn.Parameter(w4V)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        self.trans = weight_norm(nn.Linear(ctx_dim, ctx_dim_m))
        # self.trans = Initialized_Conv1d(ctx_dim, ctx_dim_m)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, Vid_enc, Ques_enc, V_mask, Q_mask):
        # Vid_enc = self.trans(Vid_enc.transpose(1, 2))
        Vid_enc = self.trans(Vid_enc)
        Ques_enc = Ques_enc.transpose(1, 2)
        batch_size = Vid_enc.size()[0]
        # Vid_enc = Vid_enc.transpose(1,2)
        S = self.trilinear_for_attention(Vid_enc, Ques_enc)
        V_mask = V_mask.view(batch_size, Lc, 1)
        Q_mask = Q_mask.view(batch_size, 1, Lq)

        S_shuffle = torch.zeros_like(S)
        for i in range(S_shuffle.shape[0]):
            rand_idx = random.randint(0, S_shuffle.shape[0] - 1)
            while rand_idx == i:
                rand_idx = random.randint(0, S.shape[0] - 1)
            S_shuffle[i] = S[rand_idx]

        S1 = F.softmax(mask_logits(S, Q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, V_mask), dim=1)

        A = torch.bmm(S1, Ques_enc)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), Vid_enc)

        S1_counter = F.softmax(mask_logits(S_shuffle, Q_mask), dim=2)
        S2_counter = F.softmax(mask_logits(S_shuffle, V_mask), dim=1)

        A_counter = torch.bmm(S1_counter, Ques_enc)
        B_counter = torch.bmm(torch.bmm(S1_counter, S2_counter.transpose(1, 2)), Vid_enc)

        # maybe concat
        out = torch.cat([Vid_enc, A, torch.mul(Vid_enc, A), torch.mul(Vid_enc, B)], dim=2)
        out_counter = torch.cat([Vid_enc, A-A_counter, torch.mul(Vid_enc, A-A_counter), torch.mul(Vid_enc, B-B_counter)], dim=2)
        return out.transpose(1, 2), Vid_enc, out_counter.transpose(1,2)


    def trilinear_for_attention(self, Vid_enc, Ques_enc):
        V = F.dropout(Vid_enc, p=dropout, training=self.training)
        Q = F.dropout(Ques_enc, p=dropout, training=self.training)
        subres0 = torch.matmul(V, self.w4V).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1 ])
        subres2 = torch.matmul(V * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class VQAttention_counter3a(nn.Module):
    def __init__(self):
        super(VQAttention_counter3a, self).__init__()
        w4V = torch.empty(ctx_dim_m, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, ctx_dim_m)
        nn.init.xavier_uniform_(w4V)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4V = nn.Parameter(w4V)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        self.trans = weight_norm(nn.Linear(ctx_dim, ctx_dim_m))
        # self.trans = Initialized_Conv1d(ctx_dim, ctx_dim_m)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, Vid_enc, Ques_enc, V_mask, Q_mask):
        # Vid_enc = self.trans(Vid_enc.transpose(1, 2))
        Vid_enc = self.trans(Vid_enc)
        Ques_enc = Ques_enc.transpose(1, 2)
        batch_size = Vid_enc.size()[0]
        # Vid_enc = Vid_enc.transpose(1,2)
        S = self.trilinear_for_attention(Vid_enc, Ques_enc)
        V_mask = V_mask.view(batch_size, Lc, 1)
        Q_mask = Q_mask.view(batch_size, 1, Lq)

        # S_shuffle = torch.zeros_like(S)
        # for i in range(S_shuffle.shape[0]):
        #     rand_idx = random.randint(0, S_shuffle.shape[0] - 1)
        #     while rand_idx == i:
        #         rand_idx = random.randint(0, S.shape[0] - 1)
        #     S_shuffle[i] = S[rand_idx]

        S_avg = torch.zeros_like(S)
        for i in range(S_avg.shape[0]):
            S_avg[i] = torch.mean(S, dim=0)

        S1 = F.softmax(mask_logits(S, Q_mask), dim=2)
        S2 = F.softmax(mask_logits(S, V_mask), dim=1)

        A = torch.bmm(S1, Ques_enc)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), Vid_enc)

        S1_counter = F.softmax(mask_logits(S_avg, Q_mask), dim=2)
        S2_counter = F.softmax(mask_logits(S_avg, V_mask), dim=1)

        A_counter = torch.bmm(S1_counter, Ques_enc)
        B_counter = torch.bmm(torch.bmm(S1_counter, S2_counter.transpose(1, 2)), Vid_enc)

        # maybe concat
        out = torch.cat([Vid_enc, A, torch.mul(Vid_enc, A), torch.mul(Vid_enc, B)], dim=2)
        out_counter = torch.cat([Vid_enc, A-A_counter, torch.mul(Vid_enc, A-A_counter), torch.mul(Vid_enc, B-B_counter)], dim=2)
        return out.transpose(1, 2), Vid_enc, out_counter.transpose(1,2)


    def trilinear_for_attention(self, Vid_enc, Ques_enc):
        V = F.dropout(Vid_enc, p=dropout, training=self.training)
        Q = F.dropout(Ques_enc, p=dropout, training=self.training)
        subres0 = torch.matmul(V, self.w4V).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1 ])
        subres2 = torch.matmul(V * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Initialized_Conv1d(D*3, 1)


    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        return Y1


class EncoderQns_PSAC(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)
        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        if self.use_bert:
            mask = ((torch.ones_like(qns)* 0)!=qns).float().mean(dim=-1) # generating mask
        else:
            mask = ((torch.ones_like(qns)* 0)!=qns).float() # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed)

        # qns_embed = self.fc(qns_embed)

        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        X, _ = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)
        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        return out, vid_enc, qns_embed_enc

class EncoderQns_PSAC2(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC2, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)
        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        mask = ((torch.ones_like(qns)* 0)!=qns).float() # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed)

        # qns_embed = self.fc(qns_embed)

        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        X, vid_enc_trans = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)
        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        return out, vid_enc_trans, qns_embed_enc

class EncoderQns_PSAC_counter(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC_counter, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention_counter()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)

        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        if self.use_bert:  # use bert
            mask = ((torch.ones_like(qns) * 0) != qns).float().mean(dim=-1)  # [320,37]
        else:
            mask = ((torch.ones_like(qns) * 0) != qns).float()  # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed)
        # qns_embed = self.fc(qns_embed)
        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        # ============= VQAttention ==========
        X, _, X_counter = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)
        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        M0_counter = self.vq_resizer(X_counter)
        out_counter = M0_counter.mean(-1)

        return out, out_counter, vid_enc, qns_embed_enc

class EncoderQns_PSAC_counter_concat(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC_counter_concat, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention_counter_concat()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)
        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        if self.use_bert:# use bert
            mask = ((torch.ones_like(qns)* 0)!=qns).float().mean(dim=-1) #[320,37]
        else:
            mask = ((torch.ones_like(qns)* 0)!=qns).float() # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed) #[bs*5, 512, 37]
        # qns_embed = self.fc(qns_embed)
        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)  #[bs*5, 512, 37]

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        # ============= VQAttention ==========
        X, _, X_counter = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)
        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        M0_counter = self.vq_resizer(X_counter)
        out_counter = M0_counter.mean(-1)

        return out, out_counter, vid_enc, qns_embed_enc

class EncoderQns_PSAC_counter2(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC_counter2, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention_counter2()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)
        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        mask = ((torch.ones_like(qns)* 0)!=qns).float() # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed)
        # qns_embed = self.fc(qns_embed)
        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        # ============= VQAttention ==========
        X, _, X_counter = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)
        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        M0_counter = self.vq_resizer(X_counter)
        out_counter = M0_counter.mean(-1)

        return out, out_counter, vid_enc, qns_embed_enc

class EncoderQns_PSAC_counter3(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC_counter3, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention_counter3()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)
        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        mask = ((torch.ones_like(qns)* 0)!=qns).float() # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed)

        # qns_embed = self.fc(qns_embed)

        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        X, _, X_counter = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)

        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        M0_counter = self.vq_resizer(X_counter)
        out_counter = M0_counter.mean(-1)

        return out, out_counter, vid_enc, qns_embed_enc

class EncoderQns_PSAC_counter3a(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns_PSAC_counter3a, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        # self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.vqatt = VQAttention_counter3a()
        self.vq_resizer = Initialized_Conv1d(D * 4, D)

        self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
        self.out = Pointer()
        self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
        # self.fc = nn.Linear(300, 512)
        self.conv1d = Initialized_Conv1d(Dword, D, bias=False)
        self.high = Highway(2)


    def forward(self, vid_enc, qns, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        mask = ((torch.ones_like(qns)* 0)!=qns).float() # generating mask
        qns_embed = self.embedding(qns) # qns = [bs, 37], qns_embed = [bs, 37, 300]
        qns_embed = self.input_dropout(qns_embed)
        # packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        # packed_output, hidden = self.rnn(packed, hidden)
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)

        qns_embed = qns_embed.transpose(1, 2)
        qns_embed = self.conv1d(qns_embed)  # batch_size x D x seq_q
        qns_embed = self.high(qns_embed)

        # qns_embed = self.fc(qns_embed)

        qns_embed_enc = self.emb_enc(qns_embed, mask, 1, 1)

        maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
        X, _, X_counter = self.vqatt(vid_enc, qns_embed_enc, maskV, mask)

        M0 = self.vq_resizer(X)
        out = M0.mean(-1)

        M0_counter = self.vq_resizer(X_counter)
        out_counter = M0_counter.mean(-1)

        return out, out_counter, vid_enc, qns_embed_enc

# class Ques_Encoder(nn.Module):
#     def __init__(self, word_mat, pretrained_char=False):
#         super(Ques_Encoder, self).__init__()
#         # add embedding matric for word and char
#         # if pretrained_char:
#         #     self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat))
#         # else:
#         #     char_mat = char_mat.float()
#         #     char_mat = torch.Tensor(char_mat)
#         #     self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
#         # self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat), freeze=True)
#         # self.emb = Embedding()
#         # self.emb_enc = EncoderBlock(conv_num=4, ch_num=D, k=7)
#         self.vqatt = VQAttention()
#         self.vq_resizer = Initialized_Conv1d(D*4, D)
#
#         self.model_enc_blks = nn.ModuleList(EncoderBlock(conv_num=2, ch_num=D, k=5) for _ in range(7))
#         self.out = Pointer()
#
#     def forward(self, vid_enc, q_w):#, q_c):
#         mask = ((torch.ones_like(q_w)* 0)!=q_w).float() # generating mask
#         # mask_c = (torch.ones_like(q_c)*0 != q_c).float()
#
#         # q_w_emb = self.word_emb(q_w) # batch_size x q_len x w_dim
#         # q_c_emb = self.char_emb(q_c) # batch_size x q_len x c_len x c_dim
#         # Q = self.emb(q_c_emb, q_w_emb, Lq) # batch_size x D x q_len
#         # Q = self.emb(q_w_emb, Lq) # batch_size x D x q_len
#         # Cq = self.emb_enc(Q, mask, 1, 1)
#         maskV = torch.ones(vid_enc.shape[0], vid_enc.shape[1]).cuda()
#         X = self.vqatt(vid_enc, q_w, maskV, mask)
#         M0 = self.vq_resizer(X)
#         out = M0.mean(-1)
#         # for i, blk in enumerate(self.model_enc_blks):
#         #      M0 = blk(M0, mask, i*(2+2)+1, 7)
#         # M1 = M0
#         # for i, blk in enumerate(self.model_enc_blks):
#         #      M0 = blk(M0, mask, i*(2+2)+1, 7)
#         # M2 = M0
#         # M0 = F.dropout(M0, p=dropout, training=self.training)
#         # for i, blk in enumerate(self.model_enc_blks):
#         #      M0 = blk(M0, mask, i*(2+2)+1, 7)
#         # M3 = M0
#         # out = self.out(M1, M2, M3, mask)
#         return out





