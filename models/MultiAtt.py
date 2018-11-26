import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.net_layer import Seq2SeqEncoder, SelfAttention, RNNDropout, SoftmaxAttention, MultiHeadAttention, AverageAttention, ResEncoder

from utils import get_mask, replace_masked, masked_softmax, weighted_sum, max_pooling


class MultiAtt(nn.Module):
    def __init__(self, args, word_embeddings, char_embeddings):
        super(MultiAtt, self).__init__()
        self.args = args
        self.vocab_size = len(word_embeddings)
        self.hidden_size = args.hidden_size
        self.num_classes = args.class_size
        self.dropout = args.dropout
        self.seq_len = args.max_seq_len

        self.embed = nn.Embedding(self.vocab_size, self.args.embed_dim, padding_idx=0)
        self.embed.weight.data.copy_(torch.tensor(word_embeddings, dtype=torch.float))
        self.embed.weight.requires_grad = False

        self.cembed = nn.Embedding(len(char_embeddings), self.args.char_dim, padding_idx=0)
        self.cembed.weight.data.copy_(torch.tensor(char_embeddings, dtype=torch.float))
        self.cembed.weight.requires_grad = False

        self._rnn_dropout = RNNDropout(self.dropout)

        # sentence encoder layer
        self.sentence_encoder = Seq2SeqEncoder(nn.GRU,
                                               self.args.embed_dim,
                                               self.args.hidden_size,
                                               bidirectional=True)
        self.char_encoder = Seq2SeqEncoder(nn.GRU,
                                           self.args.char_dim,
                                           self.args.hidden_size,
                                           bidirectional=True)
        '''
        self.sentence_encoder = ResEncoder(self.args.embed_dim,
                                           self.args.hidden_size,
                                           dropout=0.1,
                                           bias=True)
        self.char_encoder = ResEncoder(self.args.embed_dim,
                                       self.args.hidden_size,
                                       dropout=0.1,
                                       bias=True)
        '''
        self.encode_selfAtt = SelfAttention(num_head=4, d_model=2*self.hidden_size, dropout=self.dropout)
        self.cencode_selfAtt = SelfAttention(d_model=2*self.hidden_size,num_head=4, dropout=self.dropout)
        # co-attention layer
        self._trans = nn.Linear(2*4*self.hidden_size, 1, bias=False)
        self.c_trans = nn.Linear(2*4*self.hidden_size, 1, bias=False)
        self._attention = SoftmaxAttention()

        self.mulhead_attention = SelfAttention(d_model=2*self.hidden_size,num_head=4, dropout=0.2)
        self.cmulhead_attention = SelfAttention(d_model=2*self.hidden_size,num_head=4, dropout=0.2)

        self.average_attention = AverageAttention(model_dim=2*self.hidden_size, dropout=self.dropout)
        self.caverage_attention = AverageAttention(model_dim=2*self.hidden_size, dropout=self.dropout)


        # Local inference layer
        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size), nn.ReLU())
        self.char_projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size), nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.GRU,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._char_composition = Seq2SeqEncoder(nn.GRU,
                                                self.hidden_size,
                                                self.hidden_size,
                                                bidirectional=True)


        # Classification Layer
        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(8*2*self.hidden_size, 2*self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(2*self.hidden_size, self.num_classes))

        self.apply(_init_model_weights)


    def forward(self, inputs):
        premises_indices = inputs[0]
        hypothesis_indices = inputs[1]
        # print(premises_indices.size())
        # batch, 1
        premises_lengths = torch.sum(premises_indices != 0, dim=-1)
        hypothesis_lengths = torch.sum(hypothesis_indices != 0, dim=-1)
        # print(premises_lengths.size())
        # batch, seq_len
        premise_mask = get_mask(premises_indices, premises_lengths).to(self.args.device)
        hypothesis_mask = get_mask(hypothesis_indices, hypothesis_lengths).to(self.args.device)
        # print(premise_mask.size())

        embed_premise = self.embed(premises_indices)
        embed_hypothesis = self.embed(hypothesis_indices)
        # batch, seq_len, embed_dim

        embed_premise = self._rnn_dropout(embed_premise)
        embed_hypothesis = self._rnn_dropout(embed_hypothesis)
        # ----Encoder Layer----
        # (batch, seq_len, 2*hidden_size)
        encode_premise = self.sentence_encoder(embed_premise, premises_lengths)
        encode_hypothesis = self.sentence_encoder(embed_hypothesis, hypothesis_lengths)
        # print(encode_premise.size())
        # Co-Attention Layer
        # encode_premise,_ = self.average_attention(encode_premise , premise_mask)
        # encode_hypothesis,_ = self.average_attention(encode_hypothesis, hypothesis_mask)

        # attended_premise, attended_hypothesis = self._attention(encode_premise, premise_mask,
        #                                                         encode_hypothesis, hypothesis_mask)
        seq_len_p = encode_premise.size(1)
        seq_len_h = encode_hypothesis.size(1)

        _hypothesis_mask = hypothesis_mask.unsqueeze(1).expand(-1, seq_len_p, -1)  # batch, p_seq_len, h_seq_len
        _premise_mask = premise_mask.unsqueeze(2).expand(-1, -1, seq_len_h)  # batch, p_seq_len, h_seq_len
        # print(premise_mask.size())

        _encode_premise = encode_premise.unsqueeze(2).expand(-1, -1, seq_len_h, -1)
        _encode_hypothesis = encode_hypothesis.unsqueeze(1).expand(-1, seq_len_p, -1, -1)
        # print(_encode_premise.size())

        p_h = torch.cat([_encode_premise, _encode_hypothesis,
                   _encode_premise - _encode_hypothesis,
                   _encode_premise * _encode_hypothesis], dim=-1)  # batch, seq_len1, seq_len2, 4*2*hidden_size

        p_h = self._trans(p_h).squeeze(-1)  # batch, seq_len1, seq_len2
        # print(p_h.size())

        similarity_matrix_hyp = p_h + (-999999 * (_hypothesis_mask == 0).float())
        similarity_matrix_pre = p_h + (-999999 * (_premise_mask == 0).float())
        # softmax attention weight

        attention_a = F.softmax(similarity_matrix_pre, dim=2)  # batch, p_seq_len, h_seq_len
        attention_b = F.softmax(similarity_matrix_hyp, dim=1)  # batch,

        attended_premise = torch.bmm(attention_a, encode_hypothesis)  # batch, p_seq_len, hidden_size
        attended_hypothesis = torch.bmm(attention_b.transpose(1, 2), encode_premise)  # batch, h_seq_len, hidden_size

        # the enhancement layer
        # (batch, seq_len, 2*4*hidden_size)
        premise_enhanced = torch.cat([encode_premise, attended_premise,
                                      encode_premise - attended_premise,
                                      encode_premise * attended_premise], dim=-1)
        hypothesis_enhanced = torch.cat([encode_hypothesis, attended_hypothesis,
                                         encode_hypothesis - attended_hypothesis,
                                         encode_hypothesis * attended_hypothesis], dim=-1)
        # (batch, seq_len, hidden_size)
        projected_enhanced_premise = self._projection(premise_enhanced)
        projected_enhanced_hypothesis = self._projection(hypothesis_enhanced)

        # (batch, seq_len, 2*hidden_size)
        # premise = self.pair_encoder(projected_enhanced_premise, projected_enhanced_hypothesis, hypothesis_mask)
        # hypothesis = self.pair_encoder(projected_enhanced_hypothesis, projected_enhanced_premise, premise_mask)
        projected_enhanced_premise = self._rnn_dropout(projected_enhanced_premise)
        projected_enhanced_hypothesis = self._rnn_dropout(projected_enhanced_hypothesis)

        premise = self._composition(projected_enhanced_premise, premises_lengths)
        hypothesis = self._composition(projected_enhanced_hypothesis, hypothesis_lengths)
        # batch, seq_len, 2*hidden_size
        # premise = self.mulhead_attention(premise.transpose(1, 2), premise_mask).transpose(1, 2)
        # hypothesis = self.mulhead_attention(hypothesis.transpose(1, 2), hypothesis_mask).transpose(1, 2)
        # premise,_ = self.average_attention(premise, mask=premise_mask)
        # hypothesis,_ = self.average_attention(hypothesis, hypothesis_mask)

        if self.args.use_char_emb:
            cpremises_indices = inputs[2]
            chypothesis_indices = inputs[3]
            # batch, 1
            cpremises_lengths = torch.sum(cpremises_indices != 0, dim=-1)
            chypothesis_lengths = torch.sum(chypothesis_indices != 0, dim=-1)
            # batch, seq_len
            cpremise_mask = get_mask(cpremises_indices, cpremises_lengths).to(self.args.device)
            chypothesis_mask = get_mask(chypothesis_indices, chypothesis_lengths).to(self.args.device)

            cembed_premise = self.cembed(cpremises_indices)
            cembed_hypothesis = self.cembed(chypothesis_indices)
            # batch, seq_len, embed_dim
            """
            embed_premise = embed_premise.transpose(0, 1)
            embed_hypothesis = embed_hypothesis.transpose(0, 1)
            # seq_len, batch
            premise_mask = premise_mask.transpose(0, 1)
            hypothesis_mask = hypothesis_mask.transpose(0, 1)
            """

            cembed_premise = self._rnn_dropout(cembed_premise)
            cembed_hypothesis = self._rnn_dropout(cembed_hypothesis)
            # ----Encoder Layer----
            # (batch, seq_len, 2*hidden_size)
            cencode_premise = self.char_encoder(cembed_premise, cpremises_lengths)
            cencode_hypothesis = self.char_encoder(cembed_hypothesis, chypothesis_lengths)
            # (batch, seq_len, 2*4*hidden_size)
            # Co-Attention Layer
            # cencode_premise,_ = self.caverage_attention(cencode_premise, cpremise_mask)
            # cencode_hypothesis,_ = self.caverage_attention(cencode_hypothesis, chypothesis_mask)

            # cattended_premise, cattended_hypothesis = self._attention(cencode_premise, cpremise_mask,
            #                                                        cencode_hypothesis, chypothesis_mask)
            cseq_len_p = cencode_premise.size(1)
            cseq_len_h = cencode_hypothesis.size(1)

            _chypothesis_mask = chypothesis_mask.unsqueeze(1).expand(-1, cseq_len_p, -1)  # batch, p_seq_len, h_seq_len
            _cpremise_mask = cpremise_mask.unsqueeze(2).expand(-1, -1, cseq_len_h)  # batch, p_seq_len, h_seq_len
            # print(premise_mask.size())

            _cencode_premise = cencode_premise.unsqueeze(2).expand(-1, -1, cseq_len_h, -1)
            _cencode_hypothesis = cencode_hypothesis.unsqueeze(1).expand(-1, cseq_len_p, -1, -1)

            cp_h = torch.cat([_cencode_premise, _cencode_hypothesis,
                             _cencode_premise - _cencode_hypothesis,
                             _cencode_premise * _cencode_hypothesis], dim=-1)  # batch, seq_len1, seq_len2, 4*2*hidden_size

            cp_h = self.c_trans(cp_h).squeeze(-1)  # batch, seq_len1, seq_len2
            # print(cp_h.size())

            csimilarity_matrix_hyp = cp_h + (-999999 * (_chypothesis_mask == 0).float())
            csimilarity_matrix_pre = cp_h + (-999999 * (_cpremise_mask == 0).float())
            # softmax attention weight

            cattention_a = F.softmax(csimilarity_matrix_pre, dim=2)  # batch, p_seq_len, h_seq_len
            cattention_b = F.softmax(csimilarity_matrix_hyp, dim=1)  # batch,

            cattended_premise = torch.bmm(cattention_a, cencode_hypothesis)  # batch, p_seq_len, hidden_size
            cattended_hypothesis = torch.bmm(cattention_b.transpose(1, 2),
                                            cencode_premise)  # batch, h_seq_len, hidden_size

            # the enhancement layer
            # (batch, seq_len, 2*4*hidden_size)
            cpremise_enhanced = torch.cat([cencode_premise, cattended_premise,
                                          cencode_premise - cattended_premise,
                                          cencode_premise * cattended_premise], dim=-1)
            chypothesis_enhanced = torch.cat([cencode_hypothesis, cattended_hypothesis,
                                             cencode_hypothesis - cattended_hypothesis,
                                             cencode_hypothesis * cattended_hypothesis], dim=-1)
            # (batch, seq_len, hidden_size)
            cprojected_enhanced_premise = self.char_projection(cpremise_enhanced)
            cprojected_enhanced_hypothesis = self.char_projection(chypothesis_enhanced)

            # (batch, seq_len, 2*hidden_size)
            # cpremise = self.char_pair_encoder(cprojected_enhanced_premise, cprojected_enhanced_hypothesis, chypothesis_mask)
            # chypothesis = self.char_pair_encoder(cprojected_enhanced_hypothesis, cprojected_enhanced_premise, cpremise_mask)
            cprojected_enhanced_premise = self._rnn_dropout(cprojected_enhanced_premise)
            cprojected_enhanced_hypothesis = self._rnn_dropout(cprojected_enhanced_hypothesis)

            cpremise = self._char_composition(cprojected_enhanced_premise, cpremises_lengths)
            chypothesis = self._char_composition(cprojected_enhanced_hypothesis, chypothesis_lengths)

            # cpremise = self.cmulhead_attention(cpremise.transpose(1, 2), cpremise_mask).transpose(1, 2)
            # chypothesis = self.cmulhead_attention(chypothesis.transpose(1, 2), chypothesis_mask).transpose(1, 2)

            # cpremise,_ = self.average_attention(cpremise, cpremise_mask)
            # chypothesis,_ = self.average_attention(chypothesis, chypothesis_mask)

            cpremise_avg = torch.sum(cpremise * cpremise_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(cpremise_mask,
                                                                                                        dim=1,
                                                                                                        keepdim=True)
            chypothesis_avg = torch.sum(chypothesis * chypothesis_mask.unsqueeze(1).
                                   transpose(2, 1), dim=1) / torch.sum(chypothesis_mask, dim=1, keepdim=True)

            cpremise_max, _ = max_pooling(cpremise, cpremise_mask, dim=1)
            chypothesis_max, _ = max_pooling(chypothesis, chypothesis_mask, dim=1)

            # batch, 2*2*hidden
            c_premise_max_avg = torch.cat([cpremise_avg-cpremise_max, cpremise_avg*cpremise_max], dim=1)
            c_hypothesis_max_avg = torch.cat([chypothesis_avg-chypothesis_max, chypothesis_avg*chypothesis_max], dim=1)


        # premise = self.self_match_encoder(premise, premise, premise_mask)
        # hypothesis = self.self_match_encoder(hypothesis, hypothesis, hypothesis_mask)

        premise_avg = torch.sum(premise*premise_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(premise_mask, dim=1, keepdim=True)
        hypothesis_avg = torch.sum(hypothesis*hypothesis_mask.unsqueeze(1).
                                   transpose(2, 1),dim=1) / torch.sum(hypothesis_mask, dim=1, keepdim=True)

        premise_max, _ = max_pooling(premise, premise_mask, dim=1)
        hypothesis_max, _ = max_pooling(hypothesis, hypothesis_mask, dim=1)


        premise_avg_max = torch.cat([premise_avg-premise_max, premise_avg*premise_max], dim=1)
        hypothesis_avg_max = torch.cat([hypothesis_avg-hypothesis_max, hypothesis_avg*hypothesis_max], dim=1)

        v = torch.cat([premise_avg, premise_max, hypothesis_avg, hypothesis_max,
                       cpremise_avg, cpremise_max, chypothesis_avg, chypothesis_max], dim=1)
        logits = self._classification(v)

        return logits


def _init_model_weights(module):
    """
        Initialise the weights of the ESIM model.
        """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        # nn.init.constant_(module.bias.data, 0.0)

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











