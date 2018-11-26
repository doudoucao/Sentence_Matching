import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

def masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)

    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


def weight_sum(matrix, attention):
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)


def max_pooling(tensor, mask, dim=0):
    masks = mask.view(mask.size(0), mask.size(1), -1)
    masks = masks.expand(-1, -1, tensor.size(2)).float()
    return torch.max(tensor + (-999999*masks.le(0.5).float()), dim=dim)


def weighted_sum_1(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of tensor, and mask the vectors in the result with 'mask'
    :param tensor:
    :param weights:
    :param mask:
    :return:
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)

    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

def replace_masked(tensor, mask, value):
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def sort_by_seq_lens(batch, sequences_lenghts, descending=True):
    sorted_seq_lens, sorting_index = sequences_lenghts.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = sequences_lenghts.new_tensor(torch.arange(0, len(sequences_lenghts)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


class Gate(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super(Gate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return input*self.gate(input)

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequence_length, embedding_dim)
    """
    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequence.
        :param sequences_batch: batch, seq_len, embedding_Dim
        :return:
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1)*sequences_batch

'''
class RNNDropout(nn.Module):
    def __init__(self, p, batch_first=True):
        super(RNNDropout, self).__init__()
        self.dropout = nn.Dropout(p)
        self.batch_first = batch_first

    def forward(self, input):
        if not self.training:
            return input
        if self.batch_first:
            mask = input.new_ones(input.size(0), 1, input.size(2), requires_grad=False)
        else:
            mask = input.new_ones(1, input.size(1), input.size(2), requires_grad=False)

        return self.dropout(mask)*input
'''

class StaticAddAttention(nn.Module):
    def __init__(self, memory_size, input_size, attention_size, dropout=0.2):
        super(StaticAddAttention, self).__init__()
        self.attention_w = nn.Sequential(
            nn.Linear(input_size + memory_size, attention_size, bias=False),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, input, memory, memory_mask):
        T = input.size(1)
        memory_mask = memory_mask.unsqueeze(0)
        memory_key = memory.unsqueeze(0).expand(T, -1, -1, -1)
        input_key = input.unsqueeze(1).expand(-1, T, -1, -1)
        attention_logits = self.attention_w(torch.cat([input_key, memory_key], -1)).squeeze(-1)
        score = masked_softmax(attention_logits, memory_mask)
        context = torch.sum(score.unsqueeze(-1)*input.unsqueeze(0), dim=1)
        new_input = torch.cat([context, input], dim=-1)
        return new_input

class StaticDotAttention(nn.Module):
    def __init__(self, memory_size, input_size, attention_size, batch_first=True, dropout=0.2):
        super(StaticDotAttention, self).__init__()
        self.input_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(input_size, attention_size, bias=False),
            nn.ReLU()
        )
        self.attention_size = attention_size
        self.batch_first = batch_first

    def forward(self, input, memory, memory_mask):
        if not self.batch_first:
            input = input.transpose(0, 1)
            memory = memory.transpose(0, 1)
            mask = memory_mask.transpose(0, 1)

        input_ = self.input_linear(input)
        memory_ = self.input_linear(memory)
        logits = torch.bmm(input_, memory_.transpose(2, 1))/self.attention_size**0.5
        mask = memory_mask.unsqueeze(1).expand(-1, input.size(1), -1)

        score = masked_softmax(logits, mask)
        context = torch.bmm(score, memory)
        new_input = torch.cat([context, input], dim=-1)

        if not self.batch_first:
            return new_input.transpose(0, 1)

        return new_input

class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size, hidden_size, num_layers=num_layers,
                                 bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths, batch_first=True)
        outputs, _ = self._encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs

class ResEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, bias=True):
        super(ResEncoder, self).__init__()
        self.encoder = Seq2SeqEncoder(rnn_type=nn.LSTM, input_size=input_size, hidden_size=hidden_size,
                                      dropout=dropout, bidirectional=True)
        self.encoder_1 = Seq2SeqEncoder(rnn_type=nn.LSTM, input_size=(input_size+hidden_size*2), hidden_size=hidden_size,
                                        num_layers=1, bidirectional=True)
        self.encoder_2 = Seq2SeqEncoder(rnn_type=nn.LSTM, input_size=(input_size+hidden_size*2), hidden_size=hidden_size,
                                        num_layers=1, bidirectional=True)

    def forward(self, sequences_batch, sequences_length):
        # batch, seq_len, word_embed_dim
        # batch, seq_len, hidden*2
        layer1_out = self.encoder(sequences_batch, sequences_length)
        len = layer1_out.size(1)
        sequences_batch = sequences_batch[:, :len, :]
        layer2_in = torch.cat([sequences_batch, layer1_out], dim=2)
        layer2_out = self.encoder_1(layer2_in, sequences_length)
        layer3_in = torch.cat([sequences_batch, layer2_out], dim=2)
        layer3_out = self.encoder_2(layer3_in, sequences_length)
        return layer3_out


class PairEncoder(nn.Module):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, attention_factory=StaticDotAttention):
        super(PairEncoder, self).__init__()
        self.attention = attention_factory(memory_size, input_size, attention_size, dropout)
        self.gate = Gate(input_size + memory_size, dropout=dropout)
        self.rnn = nn.GRU(input_size=memory_size+input_size, hidden_size=hidden_size, bidirectional=bidirectional)

    def forward(self, input, memory, memory_mask):
        """+``````````````
        Memory: T B H
        input: T B H
        """
        output, _ = self.rnn(self.gate(self.attention(input, memory, memory_mask)))
        return output


class SelfMatchEncoder(nn.Module):
    def __init__(self, memory_size, input_size, hidden_size, attention_size, bidirectional, dropout,
                 attention_factory=StaticDotAttention):
        super(SelfMatchEncoder, self).__init__()
        self.attention = attention_factory(memory_size, input_size, attention_size, dropout=dropout)
        self.gate = Gate(input_size + memory_size, dropout=dropout)
        self.rnn = nn.GRU(input_size=memory_size + input_size, hidden_size=hidden_size, bidirectional=bidirectional)

    def forward(self, input, memory, memory_mask):
        output, _ = self.rnn(self.gate(self.attention(input, memory, memory_mask)))

        return output

'''
class SelfAttention(nn.Module):
    """
    Self Attention Module
    input size: the size for the input vector
    dim: the width of weight matrix
    num_vec: the number of encoded vector
    """
    def __init__(self, args, input_size, attention_unit=350, attention_hops=10, drop=0.5, initial_method=None):
        super(SelfAttention, self).__init__()

        self.attention_hops = attention_hops
        self.ws1 = nn.Linear(input_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.I = Variable(torch.eye(attention_hops)).to(args.device)
        self.I_origin = self.I
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()

    def penalization(self, attention):
        """
        compute the penalization term for attention module
        :param attention:
        :return:
        """
        baz = attention.size(0)
        size = self.I.size()
        if len(size) != 3 or size[0] != baz:
            self.I = self.I_origin.expand(baz, -1, -1)
        attentionT = torch.transpose(attention, 1, 2).contiguous()
        mat = torch.bmm(attention, attentionT) - self.I[:attention.size(0)]
        ret = (torch.sum(torch.sum((mat**2), 2), 1).squeeze() + 1e-10)**0.5
        return torch.sum(ret) / size[0]

    def forward(self, input, input_origin):
        input = input.contiguous()
        size = input.size()

        input_origin = input_origin.expand(self.attention_hops, -1, -1)
        input_origin = input_origin.transpose(0, 1).contiguous() # batch, hops, seq_len

        y1 = self.tanh(self.ws1(self.drop(input))) # batch, seq_len, attention_size
        attention = self.ws2(y1).transpose(1, 2).contiguous() # [batch_size, hops, len]

        attention = attention + (-999999 * (input_origin == 0).float()) # remove the weight on padding token
        attention = F.softmax(attention, 2)
        return attention.bmm(attention, input), self.penalization(attention)
'''

class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input and computing the soft attention
    between their elements.

    The dot product of the encoded vectors in the premises and hypotheses is first computed. The softmax of the result
    is then used in a weighted sum of the vectors of the premises for each element of the hypotheses, and conversely for
    the element of the premises.
    """
    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        p_seq_len = premise_mask.size(1)
        h_seq_len = hypothesis_mask.size(1)
        F.max_pool1d()
        # batch, h_seq_len
        hypothesis_mask = hypothesis_mask.unsqueeze(1).expand(-1, p_seq_len, -1)  # batch, p_seq_len, h_seq_len
        premise_mask = premise_mask.unsqueeze(2).expand(-1, -1, h_seq_len)  # batch, p_seq_len, h_seq_len

        similarity_matrix_hyp = similarity_matrix + (-999999 * (hypothesis_mask == 0).float())
        similarity_matrix_pre = similarity_matrix + (-999999 * (premise_mask == 0).float())
        # softmax attention weight

        attention_a = F.softmax(similarity_matrix_pre, dim=2) # batch, p_seq_len, h_seq_len
        attention_b = F.softmax(similarity_matrix_hyp, dim=1) # batch,

        attended_premises = torch.bmm(attention_a, hypothesis_batch)  # batch, p_seq_len, hidden_size
        attended_hypothesis = torch.bmm(attention_b.transpose(1, 2), premise_batch)  # batch, q_seq_len, hidden_size

        return attended_premises, attended_hypothesis


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=2, n_filters=200, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_filters = n_filters
        self.n_heads = n_heads

        self.key_dim = n_filters // n_heads
        self.value_dim = n_filters // n_heads

        self.fc_query = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_key = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_value = nn.ModuleList([nn.Linear(n_filters, self.value_dim) for i in range(n_heads)])
        self.fc_out = nn.Linear(n_heads * self.value_dim, n_filters)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        l = x.shape[1]

        mask = mask.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[1]).permute(0, 2, 1)

        heads = torch.zeros(self.n_heads, batch_size, l, self.value_dim).to(0)

        for i in range(self.n_heads):
            Q = self.fc_query[i](x)
            K = self.fc_key[i](x)
            V = self.fc_value[i](x)

            # scaled dot-product attention
            tmp = torch.bmm(Q, K.permute(0, 2, 1))
            tmp = tmp / np.sqrt(self.key_dim)
            tmp = F.softmax(tmp - 1e30*(1-mask), dim=-1)

            tmp = F.dropout(tmp, p=0.5, training=self.training)
            heads[i] = torch.bmm(tmp, V)
        # batch, seq_len, n_heads*value_dim
        x = heads.permute(1, 2, 0, 3).contiguous().view(batch_size, l, -1)
        # batch, seq_len, n_filters
        x = self.dropout(self.fc_out(x))

        return x


class PositionwiseFeedForward(nn.Module):
    """Position feed-forward network from attention is all you need"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        A two-layer Feed-Forward-Network with residual layer norm
        :param d_model: the size of input for the first-layer of the FFN.
        :param d_ff: the hidden layer size of the second-layer of the FNN
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output+x


class AverageAttention(nn.Module):
    """Acelerating Neural Transformer via an average attention network"""
    def __init__(self, model_dim, dropout=0.1):
        self.model_dim=model_dim
        super(AverageAttention, self).__init__()
        self.average_layer = PositionwiseFeedForward(model_dim, model_dim, dropout)
        self.gating_layer = nn.Linear(model_dim*2, model_dim*2)
        # self.dropout = nn.Dropout(p=dropout)

    def cumulative_average_mask(self, batch_size, inputs_len):
        """
        builds the mask to compute the cumulative average
        :param batch_size: batch size
        :param inputs_len: length of the inputs
        :return:
        """
        triangle = torch.tril(torch.ones(inputs_len, inputs_len))
        weights = torch.ones(1, inputs_len) / torch.arange(1, inputs_len+1, dtype=torch.float)
        mask = triangle * weights.transpose(0, 1)

        return mask.unsqueeze(0).expand(batch_size, inputs_len, inputs_len)

    def cumulative_average(self, inputs, mask_or_step, layer_cache=None, step=None):
        '''
        computes the cumulative average.
        :param inputs: sequence to average
        '''
        if layer_cache is not None:
            step = mask_or_step
            device = inputs.device
            average_attention = (inputs + step*layer_cache["prev_g"].to(device)) / (step + 1)
            layer_cache['prev_g'] = average_attention
            return average_attention
        else:
            mask = mask_or_step
            return torch.matmul(mask, inputs)

    def forward(self, inputs, mask=None, layer_cache=None, step=None):
        batch_size = inputs.size(0)
        inputs_len = inputs.size(1)
        average_outputs = self.cumulative_average(inputs, self.cumulative_average_mask(batch_size, inputs_len).to(0).float()
                                                  if layer_cache is None else step, layer_cache=layer_cache)
        average_outputs = self.average_layer(average_outputs)
        gating_outputs = self.gating_layer(torch.cat((inputs, average_outputs), -1))
        input_gate, forget_gate = torch.chunk(gating_outputs, 2, dim=2)
        gating_outputs = torch.sigmoid(input_gate)*inputs + torch.sigmoid(forget_gate)*average_outputs

        return gating_outputs, average_outputs


class Initialized_Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, relu=False, bias=False):
        super(Initialized_Conv1D, self).__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                             padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.mem_conv = Initialized_Conv1D(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1D(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """
        dot product attention
        :param q: batch, heads, length_q, depth_k
        :param k: batch, heads, length_kv, depth_k
        :param v: batch, heads, length_kv, depth_v
        :param bias:
        :param mask:
        :return:
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x !=None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = logits + (1-mask)*(-1e30)
        weights = F.softmax(logits, dim=-1)
        F.weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)


    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions"""
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)


    def combine_last_two_dim(self, x):
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a*b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.tf = rf
        self.nf = nf
        if rf == 1:
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)

        else:
            raise NotImplementedError
        return x

class Attention_v2(nn.Module):
    def __init__(self, nx, n_ctx, n_head,  dropout=0.1, scale=False):
        super(Attention_v2, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a

class HighWay(nn.Module):
    def __init__(self, in_size, n_layers=2, act=F.relu):
        super(HighWay, self).__init__()
        self.n_layers = n_layers
        self.act = act

        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = self.act(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate*normal_layer_ret + (1-gate)*x
        return x













