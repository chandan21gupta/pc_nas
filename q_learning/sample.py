import torch
from torch_geometric.data import Data
# import numpy as np
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch.distributions import Categorical
# from SPN_layers import *

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(DEVICE)
# def get_score(action_indices):
#     spn_struct = SPN_structure(action_indices, 16, 2, 2)

# class PolicyGradient(object):
#     def __init__(self, args, device):
#         self.args = args
#         self.device = device
#         self.arch_epochs = args.arch_epochs
#         self.arch_lr = args.arch_lr
#         self.episodes = args.episodes
#         self.entropy_weight = args.entropy_weight
#         self.epochs = self.epochs
#         self.controller = Controller(args, device = device).to(device)
#         self.adam = optim.Adam(params = self.controller.parameters(), lr = self.arch_lr)
#         self.baseline = None
#         self.baseline_weight = self.args.baseline_weight

#     def multi_solve_environment(self):
#         for arch_epoch in range(self.arch_epochs):
#             loss = 0
#             for episode in range(self.episodes):
#                 action_indices, actions_log_p, actions_p = self.controller.sample()
#                 actions_p = actions_p.cpu().numpy().tolist()
#                 actions_log_p = actions_log_p.cpu().numpy().tolist()
#                 actions_indices = actions_indices.cpu().numpy().tolist()
#                 likelihood = get_score(action_indices)
#                 loss += self.cal_loss(actions_p, actions_log_p, likelihood, self.baseline)

#             loss /= self.episodes
#             self.adam.zero_grad()
#             loss.backward()
#             self.adam.step()

#     def cal_loss(self, actions_p, actions_log_p, likelhood, baseline):
#         reward = likelihood - baseline
#         policy_loss = -1 * torch.sum(actions_log_p * reward)
#         entropy = -1 * torch.sum(actions_p * actions_log_p)
#         entropy_bonus = -1 * entropy * self.entropy_weight
#         return policy_loss + entropy_bonus

# class Controller(torch.nn.Module):
#     def __init__(self, xdim, num_node_features, embedding_size, action_size):
#         super().__init__()
#         self.xdim = xdim
#         self.conv1 = GCNConv(num_node_features, embedding_size)
#         self.bn1 = torch.nn.BatchNorm1d(embedding_size)
#         self.conv2 = GCNConv(embedding_size, embedding_size)
#         self.bn2 = torch.nn.BatchNorm1d(embedding_size)
#         self.conv3 = GCNConv(embedding_size, embedding_size)
#         self.bn3 = torch.nn.BatchNorm1d(embedding_size)
#         self.linear1 = torch.nn.Linear(embedding_size, 1)
#         self.linear2 = torch.nn.Linear(2*embedding_size, 1)

#     def sample(self):
#         actions_p = []
#         actions_log_p = []
#         actions_index = []
#         spn_utils = SPN_utils(self.xdim)
#         spn = spn_utils.initialise_spn()
#         mask = [1 for i in range(self.xdim)]
#         n = self.xdim
#         while(n > 0):
#             z = self.forward(spn)
#             s = torch.unsqueeze(z.sum(axis = 0), axis = 0)
#             x = torch.cat([z, s])
#             logits = self.linear1(x)
#             logits = logits.transpose(0, 1)
#             node1 = Categorical(logits = logits).sample()
#             while(node1 < self.xdim and mask[node1] == 0):
#                 # print("While Loop 1")
#                 node1 = Categorical(logits = logits).sample()
#             if node1 < self.xdim:
#                 mask[node1] = 0
#             p1 = F.softmax(logits, dim = -1)[0, node1]
#             log_p1 = F.log_softmax(logits, dim = -1)[0, node1]
#             actions_p.append(p1.detach())
#             actions_log_p.append(log_p1.detach())
#             prior = x[node1, :]
#             prior = prior.repeat(self.xdim+1, 1)
#             x = torch.cat([x, prior], axis = 1)
#             logits = self.linear2(x)
#             logits = logits.transpose(0, 1)
#             node2 = Categorical(logits = logits).sample()
#             while((node2 == self.xdim and node2 == node1) or (node2 < self.xdim and mask[node2] == 0)):
#                 # print(node2)
#                 # print("While Loop 2")
#                 node2 = Categorical(logits = logits).sample()
#             if node2 < self.xdim:
#                 mask[node2] = 0
#             p2 = F.softmax(logits, dim = -1)[0, node1]
#             log_p2 = F.log_softmax(logits, dim = -1)[0, node1]
#             actions_p.append(p2.detach())
#             actions_log_p.append(log_p2.detach())
#             actions_index.append((node1.item(), node2.item()))
#             if node1 == self.xdim or node2 == self.xdim:
#                 n = n-1
#             else:
#                 n = n-2
#             spn_utils.add_op(spn, node1.item(), node2.item())
#         return actions_index, torch.cat(actions_log_p), torch.cat(actions_p)

#     def forward(self, spn):
#         x, edge_index = spn.x, spn.edge_index
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = x[:self.xdim, :]
#         return x

# class SPN_utils:

#     def __init__(self, xdim):
#         self.xdim = xdim
#         # self.controller = GCN(self.xdim+2, latent_size)

#     def one_hot_encoding(self, classes, a):
#         return np.eye(classes)[a]

#     def add_op(self, spn, x1, x2):
#         if x1 == self.xdim:
#             x1 = spn.x.shape[0]-1
#         elif x2 == self.xdim:
#             x2 = spn.x.shape[0]-1
#         p_index = spn.x.shape[0]
#         s_index = p_index + 1
#         x = []
#         x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim)))
#         x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim+1)))
#         if spn.num_edges > 0 and (x1 != p_index-1 and x2 != p_index-1):
#             # print("here1")
#             x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim)))
#             x.append(list(self.one_hot_encoding(self.xdim+2, self.xdim+1)))
#         ids = [self.xdim, self.xdim+1]
#         if spn.num_edges > 0 and (x1 != p_index-1 and x2 != p_index-1):
#             # print("here2")
#             ids.append(self.xdim)
#             ids.append(self.xdim+1)
#         x = torch.tensor(x, dtype = torch.float)
#         ids = torch.tensor(ids, dtype = torch.int)
#         spn.x = torch.cat([spn.x, x])
#         spn.ids = torch.cat([spn.ids, ids])
#         src = [p_index, p_index, s_index]
#         dest = [x1, x2, p_index]
#         if spn.num_edges > 0 and (x1 != p_index-1 and x2 != p_index-1):
#             # print("here3")
#             src.append(s_index + 1)
#             src.append(s_index + 1)
#             src.append(s_index + 2)
#             dest.append(p_index - 1)
#             dest.append(s_index)
#             dest.append(s_index + 1)
#         edge_index = torch.tensor([src, dest], dtype = torch.long)
#         if spn.edge_index == None:
#             spn.edge_index = edge_index
#         else:
#             spn.edge_index = torch.cat([spn.edge_index, edge_index], axis = 1)
#         return spn

#     def initialise_spn(self):
#         x = []
#         ids = []
#         for i in range(self.xdim):
#             x.append(list(self.one_hot_encoding(self.xdim+2, i)))
#             ids.append(i)
#         x = torch.tensor(x, dtype = torch.float)
#         edge_index = torch.tensor([[], []], dtype = torch.long)
#         spn = Data(x = x, edge_index = edge_index)
#         spn.ids = torch.tensor(ids, dtype = torch.int)
#         return spn

# class single_scope(torch.nn.Module):

#     def __init__(self, x1):
#         super().__init__()
#         self.x1 = x1
    
#     def forward(self, x):
#         return x[:, self.x1, :]

# class combine_scopes_leaf(torch.nn.Module):

#     def __init__(self, S):
#         super().__init__()
#         self.crossproduct = CrossProduct(in_features = 2, in_channels = 2)
#         self.sumlayer = Sum(in_channels = 4, in_features = 1, out_channels = S)
    
#     def forward(self, x1, x2):
#         out = torch.cat([x1, x2], axis = 1)
#         out = self.crossproduct(out)
#         out = self.sumlayer(out)
#         return out

# class SPN_structure(torch.nn.Module):
#     def __init__(self, action_indices, xdim, I, S):
#         super().__init__()
#         self.action_indices = action_indices
#         self.xdim = xdim
#         self.I = I
#         self.S = S
#         self.model = self.decode_spn()

#     def decode_spn(self):
#         model = torch.nn.ModuleList([Bernoulli(self.xdim, self.I)])
#         for i in range(len(self.action_indices)):
#             block = self.action_indices[i]
#             left = block[0]
#             right = block[1]
#             indices = []
#             if left < self.xdim:
#                 indices.append(left)
#             if right < self.xdim:
#                 indices.append(right)
#             if len(indices) == 2:
#                 model.extend([single_scope(left)])
#                 model.extend([single_scope(right)])
#                 model.extend([combine_scopes_leaf(self.S)])
#                 if i > 0:
#                     if i == len(self.action_indices)-1:
#                         model.extend([combine_scopes_leaf(1)])
#                     else:
#                         model.extend([combine_scopes_leaf(self.S)])
#             else:
#                 model.extend([single_scope(indices[0])])
#                 if i > 0:
#                     if i == len(self.action_indices)-1:
#                         model.extend([combine_scopes_leaf(1)])
#                     else:
#                         model.extend([combine_scopes_leaf(self.S)])
#         return model

#     def forward(self, x):
#         x = self.model[0](x)
#         i = 1
#         while(i < len(self.model)):
#         # for i,l in enumerate(self.model):
#             block = self.action_indices[i]
#             left = block[0]
#             right = block[1]
#             indices = []
#             if left < self.xdim:
#                 indices.append(left)
#             if right < self.xdim:
#                 indices.append(right)
#             if len(indices) == 2:
#                 left_scope = self.model[i](x)
#                 right_scope = self.model[i+1](x)
#                 new_graph = self.model[i+2](left_scope, right_scope)
#                 if i == 0:
#                     curr_graph = new_graph
#                     i = i+3
#                 else:
#                     curr_graph = self.model[i+3](curr_graph, new_graph)
#                     i = i+4
#             else:
#                 new_graph = self.model[i](x)
#                 if i == 0:
#                     curr_graph = new_graph
#                     i = i+1
#                 else:
#                     curr_graph = self.model[i+1](curr_graph, new_graph)
#                     i = i+2
#         return curr_graph

# # spn_utils = SPN_utils(xdim = 16, latent_size = 128, time_steps = 20)
# # spn = spn_utils.sample()
# # controller = Controller(action_size = 17)
# action_indices, actions_log_p, actions_p = controller.sample()
# spn_struct = SPN_structure(action_indices, 16, 2, 2)
# # print(actions_index)
# # spn = initialise_spn(16, 2)
# # spn = add_op(spn, 0, 1, 16)
# # spn = add_op(spn, spn.x.shape[0]-1, 2, 16)
# # print(spn.x.shape)
# # print(spn.edge_index.shape)
# # print(spn.ids.shape)
# # spn = add_op(spn, 3, 4, 16)
# # print(spn.x.shape)
# # print(spn.edge_index.shape)
# # print(spn.ids.shape)
