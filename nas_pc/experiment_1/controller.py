import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
from operations import *
from networkspn import *

class Controller(nn.Module):
    def __init__(self, embedding_size = 128, dim = 16, hidden_size = 100, steps = 4, device = 'cpu'):
        super(Controller, self).__init__()
        self.steps = steps
        self.embedding_size = embedding_size
        self.dim = dim
        self.blocks = steps+1
        self.hidden_size = hidden_size
        self.scope_size = steps
        self.device = device
        self.len_OPS = len(OP_NAME)
        self.len_actions = self.dim + self.steps + self.len_OPS
        self.embedding = nn.Embedding(self.len_actions, embedding_size)
        self.block_decoders = nn.ModuleList()
        self.op_decoder = nn.Linear(hidden_size, self.len_OPS)
        self.rnn = nn.LSTMCell(self.embedding_size, hidden_size)
        self.init_parameters()
        for block in range(self.blocks):
            self.block_decoders.append(nn.Linear(hidden_size, self.dim + block))
        self.init_parameters()

    def forward(self, input, h_t, c_t, decoder):
        input = self.embedding(input)
        h_t, c_t = self.rnn(input, (h_t, c_t))
        logits = decoder(h_t)
        return h_t, c_t, logits

    def check_duplicate(self, action_index_step, action_index):
        for action in action_index_step:
            if torch.eq(action, action_index):
                return True
        return False

    def sample(self):
        input = torch.LongTensor([self.len_actions-1]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []
        scope_dict = {}
        for dim in range(self.dim):
            scope_dict[dim] = [dim]
        for block in range(self.blocks):
            scope_dict[dim+block+1] = []
            action_index_step = []
            for step in range(self.scope_size):
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.block_decoders[block])
                action_index = Categorical(logits = logits).sample()
                if step > 0:
                    while(self.check_duplicate(action_index_step, action_index)):
                        action_index = Categorical(logits = logits).sample()
                p = F.softmax(logits, dim = -1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0,action_index]
                scope_dict[dim+block+1] += scope_dict[action_index.detach().numpy()[0]]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)
                action_index_step.append(action_index)
                input = action_index
            h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
            action_index = Categorical(logits = logits).sample()
            p = F.softmax(logits, dim = -1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0,action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)
            action_index_step.append(action_index)
            input = action_index
        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)
        actions_index = torch.cat(actions_index)

        return actions_p, actions_log_p, actions_index, scope_dict

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
    
    def get_p(self, actions_index, scope_dict):
        input = torch.LongTensor([self.len_actions-1]).to(self.device)
        h_t, c_t = self.init_hidden()
        actions_p = []
        actions_log_p = []
        actions_index = []
        for block in range(self.blocks):
            action_index_step = []
            for step in range(self.scope_size):
                h_t, c_t, logits = self.forward(input, h_t, c_t, self.block_decoders[block])
                action_index = actions_index[t].unsqueeze(0)
                # if step > 0:
                #     while(self.check_duplicate(action_index_step, action_index)):
                #         action_index = Categorical(logits = logits).sample()
                p = F.softmax(logits, dim = -1)[0, action_index]
                log_p = F.log_softmax(logits, dim=-1)[0,action_index]
                actions_p.append(p.detach())
                actions_log_p.append(log_p.detach())
                actions_index.append(action_index)
                action_index_step.append(action_index)
                input = action_index

            h_t, c_t, logits = self.forward(input, h_t, c_t, self.op_decoder)
            action_index = actions_index[t].unsqueeze(0)
            p = F.softmax(logits, dim = -1)[0, action_index]
            log_p = F.log_softmax(logits, dim=-1)[0,action_index]
            actions_p.append(p.detach())
            actions_log_p.append(log_p.detach())
            actions_index.append(action_index)
            action_index_step.append(action_index)
            input = action_index
        actions_p = torch.cat(actions_p)
        actions_log_p = torch.cat(actions_log_p)
        actions_index = torch.cat(actions_index)

        return actions_p, actions_log_p, actions_index, scope_dict
    
controller = Controller()
actions_p, actions_log_p, actions_index, scope_dict = controller.sample()

print(actions_index)
print(scope_dict)

def parse_actions_index(actions_index, steps = 4):
    actions_index_ = actions_index.detach().numpy()
    blocks = []
    operations = []
    i = 0
    while(i < len(actions_index_)):
        curr_block = []
        j = 0
        while(j < steps):
            curr_block.append(actions_index_[i+j])
            j += 1
        operations.append(actions_index_[i+j])
        blocks.append(curr_block)
        i = i+j+1
    return blocks, operations

blocks, operations = parse_actions_index(actions_index)
print(blocks)
print(operations)

network_spn = NetworkSPN(blocks, operations, scope_dict, dim = 16, steps = 4)
network_spn.generate_structure()