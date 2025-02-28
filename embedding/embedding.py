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
from files_name import Files
INCLUDE_ATTRS = {"train", "data_info", "data_chunks_dir"}
for attr in dir(Files):
    if not attr.startswith("__") and attr in INCLUDE_ATTRS:
        value = getattr(Files, attr) 
        if isinstance(value, str):  
            setattr(Files, attr, os.path.join("..", value)) 
from utils import *
from data_split import *
from model import EMBModel
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

checkDir()

# Need to store config when training
class Config:
    # Loading chunk data of size 10000
    start_ses = 0
    load_batch_size = DataSplit.chunk_size
    not_improve_cnt = 0
    eval_best_loss = float("inf")
    
    # Training info
    epochs = 20
    epoch_tmp = 0
    lr = 0.001
    num_worker = multiprocessing.cpu_count()
    train_batch_size = 32
    embed_size = 10
    vocab_size = itemsMAXID()
    scheduler_step = 2
    scheduler_gamma = 0.5
    
    # Train test
    train_rate = .8

class EMBDataset(Dataset, Config):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.vocab_size = Config.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = padToTensor(self.target[idx], self.vocab_size)
        return torch.tensor(self.data[idx]), y

class EMBTrain(Config):
    def __init__(self, chunk_id):
        """
        - Config
        - Train dataset
        """
        # Update config
        if os.path.exists(Files.config):
            with open(Files.config, "rb") as f:
                config = pickle.load(f)
                self.__dict__.update(config.__dict__)     
        
        
        # Initialze training data
        print(f"Start session: {Config.start_ses}")
        self.trainset = DataSplit.loadChunkData(Files.data_chunks_dir + str(chunk_id) + ".json")
        self.trainloader, self.testloader = self.loader()
        
        print(f"Model embed size: {Config.embed_size}, vocab_size: {Config.vocab_size}")
        self.model = EMBModel(Config.vocab_size, Config.embed_size).to(device)
        self.model = loadModel(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.lr)
        self.scheduler = StepLR(self.optimizer, step_size=Config.scheduler_step, gamma=Config.scheduler_gamma)  
        
    # Load data to train
    def loader(self):
        print("loading...")
        results = self.nearData()
        split_idx = int(Config.train_rate * DataSplit.chunk_size)
        X, y = zip(*results[:split_idx])
        X = list(itertools.chain(*X))
        y = list(itertools.chain(*y))
        traindata = EMBDataset(X, y)
        trainloader = DataLoader(traindata, batch_size=Config.train_batch_size, shuffle=True, num_workers=Config.num_worker)
        
        X, y = zip(*results[split_idx:])
        X = list(itertools.chain(*X))
        y = list(itertools.chain(*y))
        testdata = EMBDataset(X, y)
        testloader = DataLoader(testdata, batch_size=Config.train_batch_size, shuffle=True, num_workers=Config.num_worker)
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
            
            total_loss = 0.
            self.model.train()
            for ses_off, (x, y) in enumerate(tqdm(self.trainloader, leave=False)):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                embedded = self.model(x)
                loss = - torch.mean(torch.diagonal(embedded@y.T))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.scheduler.step()
            Config.lr = getLr(self.optimizer)
            if getLr(self.optimizer) < 0.000001:
                print("Training finished")
                break        
            
            total_loss_mean = total_loss / ses_off
            Config.epoch_tmp = epoch + 1
            
            # Evaluate and save modle
            loss_eval = self.evaluate()       
            if loss_eval < Config.eval_best_loss:
                torch.save(self.model.state_dict(), Files.model_dir + f"ckpt-best-session-{c}.pt")
                Config.eval_best_loss = loss_eval
                Config.not_improve_cnt = 0
                storeConfig(Config)     
                
                # Reset Config for each chunk train end           
                if epoch == Config.epochs - 1: 
                    os.remove(Files.config)
            else:
                Config.not_improve_cnt += 1
                storeConfig(Config)
                if Config.not_improve_cnt >= 5:
                    os.remove(Files.config)
                    break
            
            print(f"Epoch: {epoch} Session {Config.start_ses + ses_off*Config.train_batch_size} | Loss: {loss_eval}")
        
        print("Training chunk finished")
    
    def evaluate(self):
        loss = 0.
        self.model.eval()
        with torch.no_grad():
            for ses_off, (x, y) in enumerate(tqdm(self.trainloader, leave=False)):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                embedded = self.model(x)
                loss = - torch.mean(torch.diagonal(embedded@y.T))
                loss += loss.item()
            
        return loss
    
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
            with open(Files.config, "rb") as f:
                config = pickle.load(f)
                Config.__dict__.update(config.__dict__)
        # Check condition
        if (chunk_id+1) * Config.load_batch_size < Config.start_ses:
            return None
        print("Create trainer")
        return EMBTrain(chunk_id)

for c in range(DataSplit.chunk_num):
    trainer = EMBTrain.createInstance(c)
    trainer.train()
    break
    
"""    print("trianing...")
    not_improve_cnt = 0
    for epoch in range(epochs):
        total_loss = 0.
        best = float("inf")
        for ses_off, (x, y) in tqdm(enumerate(trainloader), leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            embedded = model(x)
            loss = torch.mean(torch.diagonal(embedded@y.T))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        train_config["lr"] = getLr(optimizer)
        storeTrainingConfig(train_config)
        if getLr(optimizer) < 0.000001:
            print("Training finished")
            break        
        
        total_loss_mean = total_loss / ses_off
        if total_loss_mean < best:
            best = total_loss_mean
            not_improve_cnt = 0
            torch.save(model.state_dict(), Files.embedding_model)
            
        else:
            not_improve_cnt += 1
            if not_improve_cnt >= 5:
                print("Training finished")
                break
        
        print(f"Epoch: {epoch} Session {start_ses + ses_off*train_batch_size} | Loss: {total_loss_mean}")
    
    
    break
"""