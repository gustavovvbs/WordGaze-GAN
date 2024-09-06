import torch 
from torch import nn 
import json
import torchvision 

import os 
from torch.utils.data import Dataset 

class GestureDataset(Dataset):
    def __init__(self, root, mode = 'train', test_size = 20):
        self.root = root 
        self.mode = mode 
        self.test_size = test_size 
        self.data = self.load_data()
        self.data_len = len(self.data)
        
    def load_data(self):
        with open(self.root, 'r') as f:
            data = json.load(f)
            
        if self.mode == 'train':
            return data[self.test_size:]
        else:
            return data[:self.test_size]
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['path']).float(), self.data[idx]['word']

def dataloader(root, batch_size = 512, test_size = 20, mode = 'train'):
    dataset = GestureDataset(root, mode = mode, test_size = test_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    return dataloader, len(dataset)