import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
from operations import *


class Controller(nn.Module):
    def __init__(self, embedding_size = 128, dim = 16, hidden_size = 100, steps = 4, device = 'cpu'):
        super(Controller, self).__init__()
        self.embedding_size = embedding_size
        self.dim = dim
        self.blocks = steps+1
        self.hidden_size = hidden_size
        self.scope_size = steps
        self.device = device
        self.len_OPS = len(OP_NAME)
        self.embedding = nn.Embedding(self.scope_size+self.len_OPS, embedding_size)
        self.block_decoders = nn.ModuleList()
        self.op_decoder = nn.Linear(hidden_size, self.len_OPS)
        self.rnn = nn.LSTMCell(self.embedding_size, hidden_size)
        self.init_parameters()
        for block in range(self.blocks):
            self.block_decoders.append(nn.Linear(hidden_size, self.dim+block+1))
        self.init_parameters()

    def forward(self, input, h_t, c_t, decoder):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        return h_t, c_t, logits

    def sample(self):
        input = torch.LongTensor([self.scope_size + self.len_OPS-1]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []

        for block in range(self.blocks):
            for step in range(self.scope_size):
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.block_decoders[block])
                action_index = Categorical(logits = logits).sample()
                p = F.softmax(logits, dim = -1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0,action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)
                # input = action_index

    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.block_decoders:
            decoder.bias.data.fill_(0)
        self.op_decoder.bias.data.fill_(0)
        # self.comb_decoder.bias.data.fill_(0)


    def init_hidden(self):
        h_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        c_t = torch.zeros(1, self.hidden_size, dtype=torch.float, device=self.device)
        return (h_t, c_t)


controller = Controller()

controller.sample()
