import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import sort_by_seq_lens, masked_softmax, weighted_sum


class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequence_length, embedding_dim)
    """
    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequence.
        :param sequences_batch:
        :return:
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1)*sequences_batch


class Seq2SeqEncoder(nn.Module):
    """
        RNN taking variable length padded sequences of vectors as input and
        encoding them into padded sequences of vectors of the same length.
        This module is useful to handle batches of padded sequences of vectors
        that have different lengths and that need to be passed through a RNN.
        The sequences are sorted in descending order of their lengths, packed,
        passed through the RNN, and the resulting sequences are then padded and
        permuted back to the original order of the input sequences.
        """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


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
        # batch, h_seq_len
        hypothesis_mask = hypothesis_mask.unsqueeze(1).expand(-1, p_seq_len, -1)  # batch, p_seq_len, h_seq_len
        premise_mask = premise_mask.unsqueeze(2).expand(-1, -1, h_seq_len)  # batch, p_seq_len, h_seq_len

        similarity_matrix_hyp = similarity_matrix + (-999999 * (hypothesis_mask == 0).float())
        similarity_matrix_pre = similarity_matrix + (-999999 * (premise_mask == 0).float())
        # softmax attention weight

        attention_a = F.softmax(similarity_matrix_pre, dim=2)  # batch, p_seq_len, h_seq_len
        attention_b = F.softmax(similarity_matrix_hyp, dim=1)  # batch,

        attended_premises = torch.bmm(attention_a, hypothesis_batch)  # batch, p_seq_len, hidden_size
        attended_hypothesis = torch.bmm(attention_b.transpose(1, 2), premise_batch)  # batch, q_seq_len, hidden_size

        return attended_premises, attended_hypothesis

