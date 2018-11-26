import torch
import torch.nn as nn

from layers.esim_layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from utils import get_mask, replace_masked


class ESIM(nn.Module):
    def __init__(self, args, word_embedding, char_embedding):
        super(ESIM, self).__init__()
        self.args = args

        self.vocab_size = len(word_embedding)
        self.embedding_dim = args.embed_dim
        self.hidden_size = args.hidden_size
        self.num_classes = args.class_size
        self.dropout = args.dropout

        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.embed.weight.data.copy_(torch.tensor(word_embedding, dtype=torch.float))
        self.embed.weight.requires_grad=False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size, self.hidden_size), nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size, self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size, self.num_classes))
        self.apply(_init_esim_weights)



    def forward(self, inputs):
        premises_indices = inputs[0]
        hypothesis_indices = inputs[1]
        premises_lengths = torch.sum(premises_indices != 0, dim=-1)
        hypothesis_lengths = torch.sum(hypothesis_indices != 0, dim=-1)
        premise_mask = get_mask(premises_indices, premises_lengths).to(self.args.device)
        hypothesis_mask = get_mask(hypothesis_indices, hypothesis_lengths).to(self.args.device)

        embed_premises = self.embed(premises_indices)
        embed_hypothesis = self.embed(hypothesis_indices)

        if self.dropout:
            embed_premises = self._rnn_dropout(embed_premises)
            embed_hypothesis = self._rnn_dropout(embed_hypothesis)

        encoded_premises = self._encoding(embed_premises, premises_lengths)
        encoded_hypothesis = self._encoding(embed_hypothesis, hypothesis_lengths)

        attended_premises, attended_hypothesis = self._attention(encoded_premises, premise_mask,
                                                                 encoded_hypothesis, hypothesis_mask)
        enhanced_premise = torch.cat([encoded_premises, attended_premises,
                                      encoded_premises - attended_premises,
                                      encoded_premises * attended_premises], dim=-1)
        enhanced_hypothesis = torch.cat([encoded_hypothesis, attended_hypothesis,
                                         encoded_hypothesis - attended_hypothesis,
                                         encoded_hypothesis * attended_hypothesis], dim=-1)

        projected_premises = self._projection(enhanced_premise)
        projected_hypothesis = self._projection(enhanced_hypothesis)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypothesis = self._rnn_dropout(projected_hypothesis)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypothesis, hypothesis_lengths)

        v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(1)\
                            .transpose(2, 1), dim=1) / torch.sum(premise_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypothesis_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) / torch.sum(hypothesis_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premise_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypothesis_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)

        return logits




def _init_esim_weights(module):
    """
        Initialise the weights of the ESIM model.
        """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0




