import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.HighWay import HighWay


class CoAtt(nn.Module):
    def __init__(self, args, word_embedding, char_embedding):
        super(CoAtt, self).__init__()
        self.args = args

        if self.args.use_char_emb:
            self.char_emb = nn.Embedding(len(char_embedding), args.char_dim, padding_idx=0)
            self.char_emb.weight.data.copy_(torch.tensor(char_embedding, dtype=torch.float))
            self.char_emb.weight.requires_grad = False

        self.word_emb = nn.Embedding(len(word_embedding), args.embed_dim)
        self.word_emb.weight.data.copy_(torch.tensor(word_embedding, dtype=torch.float))
        self.word_emb.weight.requires_grad = False

        self.d = self.args.char_dim + self.args.embed_dim

        self.highway_net = HighWay(self.d)

        self.ctx_embd_layer = nn.GRU(self.d, self.args.hidden_size, bidirectional=True, dropout=0.2, batch_first=True)

        self.W = nn.Linear(6*self.args.hidden_size, 1, bias=False)
        self.modeling_layer = nn.GRU(8*self.args.hidden_size, self.args.hidden_size, num_layers=1, bidirectional=True, dropout=0.2, batch_first=True)

        self.p1_layer = nn.Linear(self.args.hidden_size*2, self.args.class_size, bias=False)


    def forward(self, inputs):
        # batch, seq_len, char_dim
        char_p_embed = self.char_emb(inputs[2])
        char_h_embed = self.char_emb(inputs[3])
        # batch, seq_len, word_dim
        word_p_embed = self.word_emb(inputs[0])
        word_h_embed = self.word_emb(inputs[1])
        # batch, seq_len, char_dim+word_dim
        embed_p = torch.cat((char_p_embed, word_p_embed), 2)
        embed_h = torch.cat((char_h_embed, word_h_embed), 2)

        # embed_p = self.highway_net(embed_p)
        # embed_h = self.highway_net(embed_h)

        batch_size = embed_p.size(0)
        seq_len = self.args.max_seq_len
        # batch, seq_len, hidden_size*2
        context_p, _ = self.ctx_embd_layer(embed_p)
        context_h, _ = self.ctx_embd_layer(embed_h)

        # Attention Flow Layer
        shape = (batch_size, seq_len, seq_len, 2*self.args.hidden_size)
        p_ex = context_p.unsqueeze(2) # batch, seq_len, 1, 2*hidden_size
        p_ex = p_ex.expand(shape)
        h_ex = context_h.unsqueeze(1) # batch, 1, seq_len, 2hidden
        h_ex = h_ex.expand(shape)
        p_elmwise_mul_h = torch.mul(p_ex, h_ex) # batch, seq_len, seq_len, 2hidden
        cat_data = torch.cat((p_ex, h_ex, p_elmwise_mul_h), 3) # batch, seq_len, seq_len, 6hidden

        S = self.W(cat_data).view(batch_size, seq_len, seq_len)  # batch, seq_len, seq_len

        # p2h
        p2h = torch.bmm(F.softmax(S, dim=-1), context_h) # batch, seq_len, 2hidden
        h2p = torch.bmm(F.softmax(S.transpose(1, 2)), context_p) # batch, seq_len, 2hidden

        G = torch.cat((context_p, context_h, p2h, h2p), 2) # (batch, seq_len, 8hidden)

        _, M = self.modeling_layer(G) # 2, batch, hidden

        P = M.permute(1, 0, 2).contiguous().view(-1, 2*self.args.hidden_size) # (batch, hidden)

        x = self.p1_layer(P)

        return x











