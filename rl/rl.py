import json 
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

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
INCLUDE_ATTRS = {"train", "data_chunks", "max_itemid"}
for attr in dir(Files):
    if not attr.startswith("__") and attr in INCLUDE_ATTRS:
        value = getattr(Files, attr) 
        if isinstance(value, str):  
            setattr(Files, attr, os.path.join("..", value)) 
from data_split import *
from model import RLModel, RLDataset
import itertools


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(
f"""CHECK SOME FILE
Files.data_chunks: {os.path.exists(Files.data_chunks)}
File.max_itemid: {os.path.exists(Files.max_itemid)}
Files.train: {os.path.exists(Files.train)}
"""
)

# Load set
info = loadConfig()
start_ses = info["start_ses"]
batch_size = info["batch_size"]
not_improve_cnt = info["not_improve_cnt"]
best = float("inf")
print(f"Start session: {start_ses}")

# Train set  
training_config = trainingConfig()
epochs = training_config["epochs"]
lr = training_config["lr"]
train_batch_size = training_config["batch_size"]
embed_size = training_config["embed_size"]
vocab_size = training_config["vocab_size"]
scheduler_step = training_config["scheduler_step"]
scheduler_gamma = training_config["scheduler_gamma"]

model = RLModel(vocab_size, embed_size).to(device)
model = loadModel(model, vocab_size, embed_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)  # Reduce LR every 10 epochs

def getXy(session):
    X, y = [], []
    aids, ts, types = [], [], []
    for ev in session["events"]:
        aids.append(ev["aid"])
        ts.append(ev["ts"])
        types.append(ev["type"])
    for i in range(len(session["events"])):
        x = session["events"][i]["aid"]
        y_click, y_cart, y_order = [], [], []
        for j in range(i+1, len(session["events"])):
            if session["events"][j]["type"] == "clicks":
                y_click.append(session["events"][j]["aid"])
                break
        for j in range(i+1, len(session["events"])):
            if session["events"][j]["type"] == "carts":
                y_cart.append(session["events"][j]["aid"])
            if session["events"][j]["type"] == "orders":
                y_order.append(session["events"][j]["aid"])
        X.append(x)
        y.append((y_click, y_cart, y_order))
    return X, y


for c in range(DataSplit.chunk_num):
    # Load data
    print("loading...")
    trainset = DataSplit.loadChunkData(Files.data_chunks + str(c) + ".json")
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(getXy, tqdm(trainset))
    pool.close()
    
    X, y = zip(*results)
    X = list(itertools.chain(*X))
    y = list(itertools.chain(*y))
    print(len(X), len(y))
    
    X = torch.tensor(X)
    y = [(torch.tensor(i[0]), torch.tensor(i[1]), torch.tensor(i[2])) for i in y]
    
    
    # Get data for train        
    print("trianing...")
    model.train()   
    for i in range(epochs):
        total_loss = []
        train_batch_cnt = 0 
        
        for x, target in tqdm((X, y), leave=False):
            x = x.to(device)
            click, cart, order = target[0].to(device), target[1].to(device), target[2].to(device)
            loss = torch.tensor(0.).to(device)
            
            optimizer.zero_grad()
            out = model(x)
            print(out)
            # loss.backward()
            # optimizer.step()
            
            # total_loss.append(loss.item())
            # train_batch_cnt += 1
            
        scheduler.step()
        training_config["lr"] = getLr(optimizer)
        storeTrainingConfig(training_config)
        
    
    total_loss_mean = np.mean(total_loss)
    if total_loss_mean < best:
        best = total_loss_mean
        not_improve_cnt = 0
        torch.save(model.state_dict(), Files.embedding_model)
        
    else:
        not_improve_cnt += 1
    
    print(f"Session start: {start_ses} Epoch: {i} Lr: {getLr(optimizer):.6f}| Loss: {total_loss_mean}")
    start_ses += batch_size
    info["start_ses"] = start_ses
    storeLoadConfig(info)
    
    if getLr(optimizer) < 0.000001:
        print("Training finished")
        break
    
    if not_improve_cnt >= 100:
        print("Training finished")
        break
    
    
    
    