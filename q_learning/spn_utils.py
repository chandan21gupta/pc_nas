import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class SPN_utils:

    def __init__(self, xdim):
        self.xdim = xdim
        # self.controller = GCN(self.xdim+2, latent_size)

    def one_hot_encoding(self, classes, a):
        return np.eye(classes)[a]

    def add_op(self, spn, x1, x2, device):
        if x1 == self.xdim:
            x1 = spn.x.shape[0]-1
        elif x2 == self.xdim:
            x2 = spn.x.shape[0]-1
        p_index = spn.x.shape[0]
        s_index = p_index + 1
        x = []
        x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim)))
        x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim+1)))
        if spn.num_edges > 0 and (x1 != p_index-1 and x2 != p_index-1):
            # print("here1")
            x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim)))
            x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim+1)))
        ids = [self.xdim, self.xdim+1]
        if spn.num_edges > 0 and (x1 != p_index-1 and x2 != p_index-1):
            # print("here2")
            ids.append(self.xdim)
            ids.append(self.xdim+1)
        x = torch.tensor(x, dtype = torch.float).to(device)
        ids = torch.tensor(ids, dtype = torch.int).to(device)
        spn.x = torch.cat([spn.x, x])
        spn.ids = torch.cat([spn.ids, ids])
        src = [p_index, p_index, s_index]
        dest = [x1, x2, p_index]
        if spn.num_edges > 0 and (x1 != p_index-1 and x2 != p_index-1):
            # print("here3")
            src.append(s_index + 1)
            src.append(s_index + 1)
            src.append(s_index + 2)
            dest.append(p_index - 1)
            dest.append(s_index)
            dest.append(s_index + 1)
        edge_index = torch.tensor([src, dest], dtype = torch.long).to(device)
        if spn.edge_index == None:
            spn.edge_index = edge_index
        else:
            spn.edge_index = torch.cat([spn.edge_index, edge_index], axis = 1)
        return spn

    def initialise_spn(self, device):
        x = []
        ids = []
        for i in range(self.xdim):
            x.append(list(self.one_hot_encoding(self.xdim+2, i)))
            ids.append(i)
        x = torch.tensor(x, dtype = torch.float).to(device)
        edge_index = torch.tensor([[], []], dtype = torch.long).to(device)
        spn = Data(x = x, edge_index = edge_index)
        spn.ids = torch.tensor(ids, dtype = torch.int)
        return spn.to(device)