from dataset_utils import load_dataset
import torch

device = 'cpu'

train_data, val_data, test_data = load_dataset('nltcs')

train_data = torch.tensor(train_data).to(device)
val_data = torch.tensor(val_data).to(device)
test_data = torch.tensor(test_data).to(device)




