import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.distributions import Categorical
from spn_utils import *

class Controller(torch.nn.Module):
    def __init__(self, xdim, num_node_features, embedding_size, device):
        super().__init__()
        self.xdim = xdim
        self.conv1 = GCNConv(num_node_features, embedding_size)
        self.bn1 = torch.nn.BatchNorm1d(embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.bn2 = torch.nn.BatchNorm1d(embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.bn3 = torch.nn.BatchNorm1d(embedding_size)
        self.linear1 = torch.nn.Linear(embedding_size, 1)
        self.linear2 = torch.nn.Linear(2*embedding_size, 1)
        self.device = device

    def sample(self):
        actions_p = []
        actions_log_p = []
        actions_index = []
        spn_utils = SPN_utils(self.xdim)
        spn = spn_utils.initialise_spn(self.device)
        mask = [1 for i in range(self.xdim)]
        n = self.xdim
        while(n > 0):
            z = self.forward(spn)
            s = torch.unsqueeze(z.sum(axis = 0), axis = 0)
            x = torch.cat([z, s])
            logits = self.linear1(x)
            logits = logits.transpose(0, 1)
            node1 = Categorical(logits = logits).sample()
            while(node1 < self.xdim and mask[node1] == 0):
                node1 = Categorical(logits = logits).sample()
            if node1 < self.xdim:
                mask[node1] = 0
            p1 = F.softmax(logits, dim = -1)[0, node1]
            log_p1 = F.log_softmax(logits, dim = -1)[0, node1]
            actions_p.append(p1)
            actions_log_p.append(log_p1)
            prior = x[node1, :]
            prior = prior.repeat(self.xdim+1, 1)
            x = torch.cat([x, prior], axis = 1)
            logits = self.linear2(x)
            logits = logits.transpose(0, 1)
            node2 = Categorical(logits = logits).sample()
            while((node2 == self.xdim and node2 == node1) or (node2 < self.xdim and mask[node2] == 0)):
                node2 = Categorical(logits = logits).sample()
            if node2 < self.xdim:
                mask[node2] = 0
            p2 = F.softmax(logits, dim = -1)[0, node1]
            log_p2 = F.log_softmax(logits, dim = -1)[0, node1]
            actions_p.append(p2)
            actions_log_p.append(log_p2)
            actions_index.append((node1.item(), node2.item()))
            if node1 == self.xdim or node2 == self.xdim:
                n = n-1
            else:
                n = n-2
            spn_utils.add_op(spn, node1.item(), node2.item(), self.device)
        return actions_index, torch.cat(actions_log_p), torch.cat(actions_p)

    def forward(self, spn):
        x, edge_index = spn.x, spn.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x[:self.xdim, :]
        return x