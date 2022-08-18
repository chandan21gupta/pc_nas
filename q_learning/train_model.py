from spn_layers import *
from spn_utils import *
from dataset_utils import *
import torch.optim as optim

lr = 3e-4
epochs = 4000
batch_size = 100

def log_density_fn(data, model):
    log_y = model(data)
    ld = torch.sum(log_y)
    return ld
    
def get_score(action_indices, data, device):
    model = SPN_structure(action_indices, 16, 3, 3)
    model = model.to(device)
    train_data, val_data, test_data = load_dataset(data)
    train_data = torch.tensor(train_data).float().to(device)
    val_data = torch.tensor(val_data).float().to(device)
    test_data = torch.tensor(test_data).float().to(device)

    def eval_model(it):
        print("in eval")
        with torch.no_grad():
            avg_train, avg_val, avg_test = 0.0, 0.0, 0.0
            def get_avg(data):
                avg = 0.0
                split_data = torch.split(data, batch_size)
                for batch_data in split_data:
                    ld = log_density_fn(batch_data, model).item()
                    avg += ld
                avg = avg / data.shape[0]
                return avg
            avg_train = get_avg(train_data)
            avg_val = get_avg(val_data)
            avg_test = get_avg(test_data)
            print('step: %u, train-all: %f, valid-all: %f, test-all: %f' % (it, avg_train, avg_val, avg_test) , flush=True)
            return avg_train, avg_val, avg_test

    def sample_batch(data):
        batch_indices = np.random.choice(data.shape[0], size=min(data.shape[0], batch_size), replace=False)
        return data[batch_indices]

    optimizer = optim.Adam(list(model.parameters()), lr = lr)

    for i in range(epochs):
        batch_train_data = sample_batch(train_data)
        optimizer.zero_grad()
        ld = log_density_fn(batch_train_data, model)
        (-ld).backward()
        optimizer.step()

        if i % 10 == 0:
            batch_valid_data = sample_batch(val_data)
            ld_valid = log_density_fn(batch_valid_data, model).item()

            batch_test_data = sample_batch(test_data)
            ld_test = log_density_fn(batch_test_data, model).item()

            avg_train = ld.item() / batch_train_data.shape[0]
            avg_valid = ld_valid / batch_valid_data.shape[0]
            avg_test = ld_test / batch_test_data.shape[0]

            # print('step: %u, train: %f, valid: %f, test: %f' % (i, avg_train, avg_valid, avg_test) , flush=True)
    avg_train, avg_val, avg_test = eval_model(100)
    # print("reward", avg_val)
    return avg_val

class single_scope(torch.nn.Module):

    def __init__(self, x1, I):
        super().__init__()
        self.b = Bernoulli(1, I)
        self.x1 = x1
    
    def forward(self, x):
        out =  x[:, self.x1, :]
        out = self.b(out)
        return torch.unsqueeze(out, axis = 1)


class combine_scopes_leaf(torch.nn.Module):

    def __init__(self, S):
        super().__init__()
        self.crossproduct = CrossProduct(in_features = 2, in_channels = S)
        self.sumlayer = Sum(in_channels = S*S, in_features = 1, out_channels = S)
    
    def forward(self, x1, x2):
        # x1 = torch.unsqueeze(x1, axis = 1)
        # x2 = torch.unsqueeze(x2, axis = 1)
        out = torch.cat([x1, x2], axis = 1)
        out = self.crossproduct(out)
        out = self.sumlayer(out)
        # print(out.shape)
        return out

class SPN_structure(torch.nn.Module):
    def __init__(self, action_indices, xdim, I, S):
        super().__init__()
        self.action_indices = action_indices
        # self.action_indices = action_indices
        self.xdim = xdim
        self.I = I
        self.S = S
        self.model = self.decode_spn()
        # print(self.model)
        # print(self.action_indices)

    def decode_spn(self):
        print(self.training)
        model = torch.nn.ModuleList()
        for i in range(len(self.action_indices)):
            block = self.action_indices[i]
            left = block[0]
            right = block[1]
            indices = []
            if left < self.xdim:
                indices.append(left)
            if right < self.xdim:
                indices.append(right)
            if len(indices) == 2:
                model.extend([single_scope(left, self.I)])
                model.extend([single_scope(right, self.I)])
                model.extend([combine_scopes_leaf(self.S)])
                if i > 0:
                    if i == len(self.action_indices)-1:
                        model.extend([combine_scopes_leaf(1)])
                    else:
                        model.extend([combine_scopes_leaf(self.S)])
            else:
                model.extend([single_scope(indices[0], self.I)])
                if i > 0:
                    if i == len(self.action_indices)-1:
                        model.extend([combine_scopes_leaf(1)])
                    else:
                        model.extend([combine_scopes_leaf(self.S)])
        return model

    def forward(self, x):
        # print(x.shape)
        # x = self.model[0](x[:, 0])
        # print(x.shape)
        x = x.float()
        x = torch.unsqueeze(x, dim = 2)
        # print(x.shape)
        i = 0
        j = 0
        while(i < len(self.model)):
        # for i,l in enumerate(self.model):
            block = self.action_indices[j]
            left = block[0]
            right = block[1]
            indices = []
            if left < self.xdim:
                indices.append(left)
            if right < self.xdim:
                indices.append(right)
            if len(indices) == 2:
                # print(i, self.model[i])
                left_scope = self.model[i](x)
                # print(i+1, self.model[i+1])
                right_scope = self.model[i+1](x)
                # print(i+2, self.model[i+2])
                new_graph = self.model[i+2](left_scope, right_scope)
                if i == 0:
                    curr_graph = new_graph
                    i = i+3
                else:
                    # print(i+3, self.model[i+3])
                    curr_graph = self.model[i+3](curr_graph, new_graph)
                    i = i+4
            else:
                # print(i, self.model[i])
                new_graph = self.model[i](x)
                if i == 0:
                    curr_graph = new_graph
                    i = i+1
                else:
                    # print(i+1, self.model[i+1])
                    curr_graph = self.model[i+1](curr_graph, new_graph)
                    i = i+2
            j = j + 1
        # print("output shape", curr_graph.shape)
        return curr_graph