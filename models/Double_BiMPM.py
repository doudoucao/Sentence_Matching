from allennlp.models import BiMpm
from allennlp.modules import BiMpmMatching

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.HighWay import HighWay


class BIMPM(nn.Module):
    def __init__(self, args, word_embedding, char_embedding):
        super(BIMPM, self).__init__()
        self.args = args
        self.l = self.args.num_perspective
        # self.d = self.args.embed_dim + int(self.args.use_char_emb)*self.args.char_hidden_size
        self.d = self.args.embed_dim + int(self.args.use_char_emb) * self.args.char_dim

        # ---word representation layer ---
        if args.use_char_emb:
            self.char_emb = nn.Embedding(len(char_embedding), args.char_dim, padding_idx=0)
            self.char_emb.weight.data.copy_(torch.tensor(char_embedding, dtype=torch.float))
            self.char_emb.weight.requires_grad = False

        self.word_emb = nn.Embedding(len(word_embedding), args.embed_dim)
        # initialize word embedding with pretraining word vectors
        self.word_emb.weight.data.copy_(torch.tensor(word_embedding, dtype=torch.float))
        # no fine_tuning for word vectors
        self.word_emb.weight.requires_grad = False

        self.char_LSTM = nn.LSTM(input_size = self.args.char_dim,
                                 hidden_size=self.args.char_hidden_size,
                                 num_layers=1,
                                 bidirectional=True,
                                 batch_first=True)

        self.context_LSTM = nn.LSTM(input_size=self.args.embed_dim,
                                    hidden_size=self.args.hidden_size,
                                    num_layers=1,
                                    bidirectional=True,
                                    batch_first=True)
        self.highway = HighWay(4*self.args.hidden_size)

        # ---Matching layer---
        for i in range(1, 7):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(self.l, self.args.hidden_size)))

        for i in range(1, 7):
            setattr(self, f'char_w{i}', nn.Parameter(torch.rand(self.l, self.args.char_hidden_size)))

        # --- Aggregation Layer ---
        self.aggregation_LSTM = nn.LSTM(input_size=self.l*6,
                                        hidden_size=self.args.hidden_size,
                                        num_layers=1,
                                        bidirectional=True,
                                        batch_first=True
                                        )
        # ---Prediction Layer ---
        self.pred_fc1 = nn.Linear(self.args.hidden_size*8, self.args.hidden_size*2)
        self.pred_fc2 = nn.Linear(self.args.hidden_size*2, self.args.class_size)

        self.reset_parameters()

    def reset_parameters(self):
        # --- word representation layer ---

        # nn.init.uniform(self.char_emb.weight, -0.005, 0.005)
        # zero vectors for padding
        # self.char_emb.weight.data[0].fill_(0)

        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.word_emb.weight.data[-1], -0.1, 0.1)

        nn.init.kaiming_normal_(self.char_LSTM.weight_ih_l0)
        nn.init.constant_(self.char_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.char_LSTM.weight_hh_l0)
        nn.init.constant_(self.char_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.char_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.char_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.char_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.char_LSTM.bias_hh_l0_reverse, val=0)


        # --- context representation layer ---
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # ---match layer ---
        for i in range(1, 7):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)

        for i in range(1, 7):
            w = getattr(self, f'char_w{i}')
            nn.init.kaiming_normal_(w)

        # --- Aggregation Layer ---
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)

    def multi_perspective_match(self, v1, v2, weight):
        """
        :param v1: (batch, seq_len, hidden_size)
        :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
        :param weight: (l, hidden_size)
        :return: (batch, l)
        """
        assert v1.size(0) == v2.size(0)
        assert weight.size(1) == v1.size(2)

        seq_len = v1.size(1)
        # (1, 1, hidden_size, l)
        weight = weight.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        # (batch, seq_len, hidden_size, l)
        v1 = weight * torch.stack([v1] * self.l, dim=3)
        if len(v2.size()) == 3:
            v2 = weight*torch.stack([v2]*self.l, dim=3)
        else:
            v2 = weight*torch.stack([torch.stack([v2]*seq_len, dim=1)]*self.l, dim=3)
        # batch_size, seq_len, num_perspectives
        m = F.cosine_similarity(v1, v2, dim=2)

        return m

    def multi_perspective_match_pairwise(self, v1, v2, weight, eps=1e-8):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :param weight: (num_perspective, hidden_size)
        :return:
        """
        num_perspective = weight.size(0)
        # (1, num_perspective, 1, hidden_size)
        weight = weight.unsqueeze(0).unsqueeze(2)
        # (batch, num_perspective, seq_len*, hidden_size)
        v1 = weight * v1.unsqueeze(1).expand(-1, num_perspective, -1, -1)
        v2 = weight * v2.unsqueeze(1).expand(-1, num_perspective, -1, -1)
        # (batch, num_perspective, seq_len*, 1)
        v1_norm = v1.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2.norm(p=2, dim=3, keepdim=True)
        # (batch, num_perspective, seq_len1, seq_Len2)
        mul_result = torch.matmul(v1, v2.transpose(2, 3))
        norm_value = v1_norm * v2_norm.transpose(2, 3)
        # (batch, seq_len1, seq_len2, num_perspective)
        return (mul_result / norm_value.clamp(min=eps)).permute(0, 2, 3, 1)

    def attention(self, v1, v2, eps=1e-8):
        """
        :param v1: (batch, seq_len1, hidden_size)
        :param v2: (batch, seq_len2, hidden_size)
        :return: (batch, seq_len1, seq_len2)
        """
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

        a = torch.bmm(v1, v2.transpose(1, 2))
        d = v1_norm * v2_norm
        # batch, seq_len1, seq_len2
        return a / d.clamp(min=eps)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout)

    def forward(self, inputs):
        # ---- word representation layer ---
        p = self.word_emb(inputs[0])
        h = self.word_emb(inputs[1])

        if self.args.use_char_emb:
            # (batch, seq_len, max_word_len) --> (batch*seq_len, max_word_len)
            # seq_len_p = inputs[2].size(1)
            # seq_len_h = inputs[3].size(1)
            char_p = inputs[2]
            char_h = inputs[3]

            # char_p = self.char_emb(char_p)
            # char_h = self.char_emb(char_h)

            # (batch* seq_len, max_word_len, char_dim) -> (1, batch*seq_len, char_hidden_size)
            # batch, char_len, char_hidden_size*2
            char_p, (_, _) = self.char_LSTM(self.char_emb(char_p))
            char_h, (_, _) = self.char_LSTM(self.char_emb(char_h))

            # batch, seq_len, char_hidden_size
            # char_p = char_p.view(-1, seq_len_p, self.args.char_hidden_size)
            # char_h = char_h.view(-1, seq_len_h, self.args.char_hidden_size)

            # (char, seq_len, word_dim + char_hidden_size)
            # p = torch.cat([p, char_p], dim=-1)
            # h = torch.cat([h, char_h], dim=-1)

            char_p = self.dropout(char_p)
            char_h = self.dropout(char_h)

            # (batch, seq_len, hidden_size)
            char_p_fw, char_p_bw = torch.split(char_p, self.args.char_hidden_size, dim=-1)
            char_h_fw, char_h_bw = torch.split(char_h, self.args.char_hidden_size, dim=-1)

            # 1. Full-Matching

            # (batch, seq_len, hidden_size), (batch, hidden_size) -> (batch, seq_len, l)
            cmv_p_full_fw = self.multi_perspective_match(char_p_fw, char_h_fw[:, -1, :], self.char_w1)
            cmv_p_full_bw = self.multi_perspective_match(char_p_bw, char_h_bw[:, 0, :], self.char_w2)
            cmv_h_full_fw = self.multi_perspective_match(char_h_fw, char_p_fw[:, -1, :], self.char_w1)
            cmv_h_full_bw = self.multi_perspective_match(char_h_bw, char_p_bw[:, 0, :], self.char_w2)

            # 2. Maxpooling-Matching
            """
            # (batch, seq_len1, seq_len2, l)
            mv_max_fw = self.multi_perspective_match_pairwise(con_p_fw, con_h_fw, self.mp_w3)
            mv_max_bw = self.multi_perspective_match_pairwise(con_p_bw, con_h_bw, self.mp_w4)

            # (batch, seq_len, l)
            mv_p_max_fw, _ = mv_max_fw.max(dim=2)
            mv_p_max_bw, _ = mv_max_bw.max(dim=2)
            mv_h_max_fw, _ = mv_max_fw.max(dim=1)
            mv_h_max_bw, _ = mv_max_bw.max(dim=1)
            """
            # 3. Attentive Matching
            # (batch, seq_len1, seq_len2)
            att_fw = self.attention(char_p_fw, char_h_fw)
            att_bw = self.attention(char_p_bw, char_h_bw)

            # (batch, seq_len2, hidden_size)  --> (batch, 1, seq_len2, hidden_size)
            # (batch, seq_len1, seq_len2) --> (batch, seq_len1, seq_Len2, 1)
            # (batch, seq_len1, seq_len2, hidden_size)
            att_h_fw = char_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
            att_h_bw = char_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)

            att_p_fw = char_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
            att_p_bw = char_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
            # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) --> (batch, seq_len1, hidden_size)
            catt_mean_h_fw = att_h_fw.sum(dim=2) / att_fw.sum(dim=2, keepdim=True).clamp(min=1e-8)
            catt_mean_h_bw = att_h_bw.sum(dim=2) / att_bw.sum(dim=2, keepdim=True).clamp(min=1e-8)
            # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1)
            catt_mean_p_fw = att_p_fw.sum(dim=1) / att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1).clamp(min=1e-8)
            catt_mean_p_bw = att_p_bw.sum(dim=1) / att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1).clamp(min=1e-8)

            # (batch, seq_len, l)
            cmv_p_att_mean_fw = self.multi_perspective_match(char_p_fw, catt_mean_h_fw, self.mp_w3)
            cmv_p_att_mean_bw = self.multi_perspective_match(char_p_bw, catt_mean_h_bw, self.mp_w4)
            cmv_h_att_mean_fw = self.multi_perspective_match(char_h_fw, catt_mean_p_fw, self.mp_w3)
            cmv_h_att_mean_bw = self.multi_perspective_match(char_h_bw, catt_mean_p_bw, self.mp_w4)

            # 4. Max-Attentive Matching
            # (batch, seq_len1, hidden_size)
            att_max_h_fw, _ = att_h_fw.max(dim=2)
            att_max_h_bw, _ = att_h_bw.max(dim=2)
            # (batch, seq_len2, hidden_size)
            att_max_p_fw, _ = att_p_fw.max(dim=1)
            att_max_p_bw, _ = att_p_bw.max(dim=1)

            # (batch, seq_len, l)
            cmv_p_att_max_fw = self.multi_perspective_match(char_p_fw, att_max_h_fw, self.mp_w5)
            cmv_p_att_max_bw = self.multi_perspective_match(char_p_bw, att_max_h_bw, self.mp_w6)
            cmv_h_att_max_fw = self.multi_perspective_match(char_h_fw, att_max_p_fw, self.mp_w5)
            cmv_h_att_max_bw = self.multi_perspective_match(char_h_bw, att_max_p_bw, self.mp_w6)

            # (batch, seq_len, l*8)
            cmv_p = torch.cat(
                [cmv_p_full_fw, cmv_p_att_mean_fw, cmv_p_att_max_fw,
                 cmv_p_full_bw, cmv_p_att_mean_bw, cmv_p_att_max_bw], dim=2)
            cmv_h = torch.cat(
                [cmv_h_full_fw, cmv_h_att_mean_fw, cmv_h_att_max_fw,
                 cmv_h_full_bw, cmv_h_att_mean_bw, cmv_h_att_max_bw], dim=2)

            cmv_p = self.dropout(cmv_p)
            cmv_h = self.dropout(cmv_h)

            _, (cagg_p_last, _) = self.aggregation_LSTM(cmv_p)
            _, (cagg_h_last, _) = self.aggregation_LSTM(cmv_h)

            # 2*(2, batch, hidden_size) -> 2*(batch, hidden_size*2) -> (batch, hidden_size * 4)
            cx = torch.cat([cagg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2),
                        cagg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * 2)], dim=1)
            cx = self.dropout(cx)



        # p = self.dropout(p)
        # h = self.dropout(h)

        # --- context representation layer ---
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p, self.args.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.args.hidden_size, dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size) -> (batch, seq_len, l)
        mv_p_full_fw = self.multi_perspective_match(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = self.multi_perspective_match(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = self.multi_perspective_match(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = self.multi_perspective_match(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching
        """
        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = self.multi_perspective_match_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = self.multi_perspective_match_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)
        """
        # 3. Attentive Matching
        # (batch, seq_len1, seq_len2)
        att_fw = self.attention(con_p_fw, con_h_fw)
        att_bw = self.attention(con_p_bw, con_h_bw)

        # (batch, seq_len2, hidden_size)  --> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) --> (batch, seq_len1, seq_Len2, 1)
        # (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)

        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) --> (batch, seq_len1, hidden_size)
        att_mean_h_fw = att_h_fw.sum(dim=2) / att_fw.sum(dim=2, keepdim=True).clamp(min=1e-8)
        att_mean_h_bw = att_h_bw.sum(dim=2) / att_bw.sum(dim=2, keepdim=True).clamp(min=1e-8)
        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1)
        att_mean_p_fw = att_p_fw.sum(dim=1) / att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1).clamp(min=1e-8)
        att_mean_p_bw = att_p_bw.sum(dim=1) / att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1).clamp(min=1e-8)

        # (batch, seq_len, l)
        mv_p_att_mean_fw = self.multi_perspective_match(con_p_fw, att_mean_h_fw, self.mp_w3)
        mv_p_att_mean_bw = self.multi_perspective_match(con_p_bw, att_mean_h_bw, self.mp_w4)
        mv_h_att_mean_fw = self.multi_perspective_match(con_h_fw, att_mean_p_fw, self.mp_w3)
        mv_h_att_mean_bw = self.multi_perspective_match(con_h_bw, att_mean_p_bw, self.mp_w4)

        # 4. Max-Attentive Matching
        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)

        # (batch, seq_len, l)
        mv_p_att_max_fw = self.multi_perspective_match(con_p_fw, att_max_h_fw, self.mp_w5)
        mv_p_att_max_bw = self.multi_perspective_match(con_p_bw, att_max_h_bw, self.mp_w6)
        mv_h_att_max_fw = self.multi_perspective_match(con_h_fw, att_max_p_fw, self.mp_w5)
        mv_h_att_max_bw = self.multi_perspective_match(con_h_bw, att_max_p_bw, self.mp_w6)

        # (batch, seq_len, l*8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ---Aggregation Layer ---
        # (batch, seq_len 8*l) --> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # 2*(2, batch, hidden_size) -> 2*(batch, hidden_size*2) -> (batch, hidden_size * 4)
        x = torch.cat([agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size*2),
                      agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size *2)], dim=1)

        x = self.dropout(x)

        # batch, hidden_size*8
        if self.args.use_char_emb:
            x = torch.cat([cx, x], dim=1)

        # ---Prediction Layer ---
        x = F.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x
