import json, pickle
import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

import multiprocessing
import multiprocessing.managers

from tqdm import tqdm

import itertools
import warnings
warnings.filterwarnings("ignore")

# Import local file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
from data_split import DataSplit, itemsMAXID
from model import EMBModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Files:
    # Data path for model training
    train = "../train.jsonl"
    data_chunks_dir = "../data_chunk/"
    data_info = "../datainfo.json"
    # local files (for each algorithm)
    tmp_dir = "tmp/"
    model_dir = "checkpoints/"
    config = "tmp/config.json"

checkDir([Files.data_chunks_dir, Files.tmp_dir, Files.model_dir])

# Need to store config when training
class Config:
    # Store while training
    chunk_id = -1
    
    epoch_tmp = 0
    not_improve_cnt = 0
    eval_best_loss = float("inf")
    lr = 0.01
    
    # Training info
    load_batch_size = DataSplit.chunk_size
    epochs = 2
    train_batch_size = 64
    embed_size = 10
    vocab_size = itemsMAXID()
    scheduler_step = 2
    scheduler_gamma = 0.9
    
    # Train test 
    train_rate = .8

class EMBDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), padToTensor(self.target[idx], Config.vocab_size)

class EMBTrain(Config):
    def __init__(self, chunk_id):
        """
        - Config
        - Train dataset
        """
        if os.path.exists(Files.config):
            tmp_config = loadConfig(Files.config)
            # Restore train status
            print(f"Load previous training status: {tmp_config}")
            for attr in dir(self):
                if not attr.startswith("__") and attr in tmp_config.keys():
                    setattr(self, attr, tmp_config[attr])    
                    
        # Initialze training data
        print(f"Start chunk: {self.chunk_id+1}")
        self.trainset = DataSplit.loadChunkData(Files.data_chunks_dir + str(chunk_id) + ".json")
        self.trainloader, self.testloader = self.loader()
        
        # Initialize model
        print(f"Model embed size: {self.embed_size}, vocab_size: {self.vocab_size}")
        self.model = EMBModel(self.vocab_size, self.embed_size).to(device)
        self.model = loadModel(self.model, Files.model_dir)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)  
        
    # Load data to train
    def loader(self):
        print("loading...")
        results = self.nearData()
        split_idx = int(self.train_rate * DataSplit.chunk_size)
        X, y = zip(*results[:split_idx])
        X = list(itertools.chain(*X))
        y = list(itertools.chain(*y))
        traindata = EMBDataset(X, y)
        trainloader = DataLoader(traindata, batch_size=self.train_batch_size, shuffle=True, num_workers=2)
        
        X, y = zip(*results[split_idx:])
        X = list(itertools.chain(*X))
        y = list(itertools.chain(*y))
        testdata = EMBDataset(X, y)
        testloader = DataLoader(testdata, batch_size=self.train_batch_size, shuffle=True, num_workers=1)
        return trainloader, testloader

    def nearData(self) -> tuple:        
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(EMBTrain.taskGetNeighbor, tqdm(self.trainset))
        pool.close()

        return results
    
    # Train
    def train(self):      
        for epoch in range(self.epochs):  
            if epoch < self.epoch_tmp:
                continue
            
            self.model.train()
            for x, y in tqdm(self.trainloader, leave=False):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                embedded = self.model(x)
                loss = - torch.mean(torch.diagonal(embedded@y.T))
                loss.backward()
                self.optimizer.step()
                break
            
            self.scheduler.step()
            self.lr = getLr(self.optimizer)
            if self.lr < 0.0000000001:
                print("Lr too small")
                break        
            
            self.epoch_tmp = epoch + 1
            
            # Evaluate and save model
            loss_eval = self.evaluate()       
            if loss_eval < self.eval_best_loss:
                torch.save(self.model.state_dict(), Files.model_dir + f"ckpt-best-session-{c}.pt")
                self.eval_best_loss = loss_eval
                self.not_improve_cnt = 0
            else:
                self.not_improve_cnt += 1
                if self.not_improve_cnt >= 100:
                    print("Not improving")
                    os.remove(Files.config)
                    break
            
            # Update epoch for the training chunk
            self.epoch_tmp = epoch
            
            # Store train status
            tmp_config = {
                "chunk_id": self.chunk_id, 
                "epoch_tmp": self.epoch_tmp, 
                "not_improve_cnt": self.not_improve_cnt, 
                "eval_best_loss": self.eval_best_loss, 
                "lr": self.lr
                }
            storeConfig(Files.config, tmp_config)
                        
            print(f"Epoch: {epoch} | Loss: {loss_eval}")
        
        # Reset training status when finish training
        self.chunk_id += 1
        self.epoch_tmp = 0
        self.lr = Config.lr
        self.not_improve_cnt = 0
        self.eval_best_loss = float("inf")
        tmp_config["chunk_id"] = self.chunk_id
        tmp_config["epoch_tmp"] = self.epoch_tmp
        tmp_config["lr"] = self.lr
        tmp_config["not_improve_cnt"] = self.not_improve_cnt
        tmp_config["eval_best_loss"] = self.eval_best_loss
        storeConfig(Files.config, tmp_config)
        
        print("Training chunk finished")
        
        torch.cuda.empty_cache()
    
    def evaluate(self):
        print("Evaluating...")
        loss = 0.
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for x, y in tqdm(self.testloader, leave=False):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                embedded = self.model(x)
                loss = - torch.mean(torch.diagonal(embedded@y.T))
                total_loss += loss.item()
                break
            
        return total_loss / len(self.testloader)
    
    @staticmethod
    def taskGetNeighbor(session: list, context_window: int = 60*60*24*1000) -> None:
        X = []
        y = []
        session_ = pd.DataFrame(session["events"])
        for ev in session["events"]:
            X.append(ev["aid"])
            y.append(list(session_[(session_["ts"] < ev["ts"] + context_window) & (session_["ts"] > ev["ts"] - context_window)]["aid"]))        
        return X, y
    
    @staticmethod
    def createInstance(chunk_id):
        # Update config
        if os.path.exists(Files.config):
            tmp_config = loadConfig(Files.config)
            # Check condition
            if chunk_id <= tmp_config["chunk_id"]:
                return None
            
        print("Create trainer")
        return EMBTrain(chunk_id)

for c in range(DataSplit.chunk_num):
    trainer = EMBTrain.createInstance(c)
    if trainer is not None:
       trainer.train()
    else:
        print(f"Already trained this chunk {c}!")
    # break


