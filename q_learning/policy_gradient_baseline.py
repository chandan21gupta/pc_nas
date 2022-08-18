import torch
import numpy as np
import torch.nn.functional as F
from controller import *
from train_model import *
import torch.optim as optim
import time
import torch.nn as nn

print(torch.cuda.is_available())

def consume(action_indices, data, device):
    likelihood = get_score(action_indices, data, device)
    return likelihood

class PolicyGradient(object):
    def __init__(self, arch_epochs = 100, arch_lr = 1e-3, episodes = 20, entropy_weight = 1e-5, 
                data = 'nltcs'):
        self.device_arch = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        print(self.device_arch)
        self.arch_epochs = arch_epochs
        self.arch_lr = arch_lr
        self.episodes = episodes
        self.entropy_weight = entropy_weight
        self.controller = Controller(xdim = 16, num_node_features = 18, embedding_size = 64, device = self.device_arch).to(self.device_arch)
        self.adam = optim.Adam(params = self.controller.parameters(), lr = self.arch_lr)
        self.baseline = 0
        self.data = data
        self.baseline_weight = 0.95

    def multi_solve_environment(self):
        for arch_epoch in range(self.arch_epochs):
            # print("arch epoch", arch_epoch)
            loss = 0
            start_time = time.time()
            for episode in range(self.episodes):
                print("arch epoch", arch_epoch, "episode", episode)
                actions_indices, actions_log_p, actions_p = self.controller.sample()
                # actions_p = actions_p.cpu().numpy().tolist()
                # actions_log_p = actions_log_p.cpu().numpy().tolist()
                # actions_indices = actions_indices.cpu().numpy().tolist()
                likelihood = consume(actions_indices, self.data, self.device)
                likelihood = torch.tensor(likelihood).to(self.device_arch)
                self.baseline = self.baseline*self.baseline_weight + likelihood*(1 - self.baseline_weight)
                loss += self.cal_loss(actions_p, actions_log_p, likelihood, self.baseline)

            # print(loss.grad_fn)
            print("--- %s seconds ---" % (time.time() - start_time))
            loss /= self.episodes
            self.adam.zero_grad()
            # print(loss.grad_fn)
            loss.backward()
            self.adam.step()

    def cal_loss(self, actions_p, actions_log_p, likelihood, baseline):
        # reward = likelihood - baseline
        reward = likelihood
        # print(actions_log_p)
        # print(actions_log_p.shape)
        policy_loss = -1 * torch.sum(actions_log_p * reward)
        # entropy = -1 * torch.sum(actions_p * actions_log_p)
        # entropy_bonus = -1 * entropy * self.entropy_weight
        # loss = policy_loss + entropy_bonus
        # print(loss.grad_fn)
        return policy_loss


pg = PolicyGradient()
pg.multi_solve_environment()