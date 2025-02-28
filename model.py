import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class RLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EMBModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(EMBModel, self).__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = self.softmax(x)
        x = torch.log(x)
        return x
class RLModel(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(RLModel, self).__init__()
        self.emb_size = emb_size 
        self.vocab_size = vocab_size
        self.embedding_items = nn.Embedding(vocab_size, emb_size)
        self.embedding_types = nn.Embedding(3, 1)
        self.linear = nn.Linear(emb_size + 1, vocab_size)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x1 = self.embedding_items(x[0])
        x2 = self.embedding_types(x[1])
        x = torch.cat((x1, x2), dim=0)
        x = self.linear(x)
        x = self.softmax(x)
        x = torch.log(x)
        return x
