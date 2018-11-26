import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese_Net(nn.Module):
    def __init__(self, args, word_embeddings):
        super(Siamese_Net, self).__init__()
        self.args = args
        self.embed = nn.Embedding(len(word_embeddings), self.args.embed_dim)
        self.embed.weight.data.copy_(torch.tensor(word_embeddings, dtype=torch.float))
        self.embed.weight.requires_grad=True
        self.encoding= nn.LSTM(input_size=self.args.embed_dim, hidden_size=self.args.hidden_size, bidirectional=False, num_layers=1, batch_first=True)
        self.linear_1 = nn.Linear(3*self.args.hidden_size, self.args.hidden_size)
        self.out = nn.Linear(self.args.hidden_size, self.args.class_size)

    def dropout(self, x):
        return F.dropout(x, p=self.args.dropout)

    def forward(self, inputs):
        p = self.embed(inputs[0])
        h = self.embed(inputs[1])
        #(1, batch, hidden_size)
        _,(p_context,_) = self.encoding(p)
        _,(h_context,_ )= self.encoding(h)
        p_context = p_context.squeeze(0)
        h_context = h_context.squeeze(0)
        # (batch, hidden_size*3)
        p_h = torch.cat((p_context, h_context, p_context-h_context), dim=1)

        fusion_ph = F.relu(self.linear_1(p_h))
        fusion_ph = self.dropout(fusion_ph)
        logit = F.softmax(self.out(fusion_ph))

        return logit




