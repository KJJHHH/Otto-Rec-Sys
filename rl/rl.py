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
# Update file path (file path for all models)
INCLUDE_ATTRS = {"train", "data_chunks_dir", "max_itemid"}
for attr in dir(Files):
    if not attr.startswith("__") and attr in INCLUDE_ATTRS:
        value = getattr(Files, attr) 
        if isinstance(value, str):  
            setattr(Files, attr, os.path.join("..", value)) 
from data_split import *
from model import RLModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(
f"""CHECK SOME FILE
Files.data_chunks: {os.path.exists(Files.data_chunks_dir)}
File.max_itemid: {os.path.exists(Files.max_itemid)}
Files.train: {os.path.exists(Files.train)}
"""
)
if not os.path.exists(Files.model_dir):
    os.makedirs(Files.model_dir)

# Load set
load_config = loadConfig()
start_ses = load_config["start_ses"]
batch_size = load_config["batch_size"]
not_improve_cnt = load_config["not_improve_cnt"]
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
model = loadModel(model)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)  # Reduce LR every 10 epochs

for c in range(DataSplit.chunk_num):
    # Load data
    print("loading...")
    trainset = DataSplit.loadChunkData(Files.data_chunks_dir + str(c) + ".json")
    results = trainData(trainset)
    
    # Get data for train        
    print("TRAINING...")
    model.train()    
    for i in range(epochs):
        total_ses_loss = 0
        num_ev = 0
        stat = [0, 0, 0]  # [total_ses_click_hit, total_ses_cart_hit, total_ses_order_hit]
        
        for ses_id, session in enumerate(results):
            
            print(f"TRAINING session: {ses_id + start_ses} epoch {i}", end="\r")
            events, y = session
            events = torch.tensor(events)
            y = [(torch.tensor(j[0], dtype = torch.int32), torch.tensor(j[1], dtype = torch.int32), torch.tensor(j[2], dtype = torch.int32)) for j in y ] # Click, Cart, Order
            
            for ev_id, (ev, target) in enumerate(zip(events, y)):          
                optimizer.zero_grad()
                ev = ev.to(device)      
                out = model(ev)
                
                loss = computeLoss(out, target, stat)
                loss.backward()
                optimizer.step()
                
                total_ses_loss += loss.item()
                num_ev += 1
            
            if (ses_id + start_ses) % 10 == 0:
                print(f"Epoch: {i} Session : {ses_id + start_ses} | Mean session loss: {total_ses_loss/num_ev:.3f}, \
                    Mean click hit (per event): {stat[0]/num_ev:.3f}, \
                    Mean cart hit (per event): {stat[1]/num_ev:.3f}, \
                    Mean order hit (per event): {stat[2]/num_ev:.3f}")
                # Store checkpoints
                torch.save(model.state_dict(), Files.model_dir + f"ckpt-{ses_id}.pt")
                load_config["start_ses"] = ses_id
                training_config["lr"] = getLr(optimizer)
                storeLoadConfig(load_config)
                storeTrainingConfig(training_config)
                total_ses_loss = 0
                num_ev = 0
                stat = [0, 0, 0]  # [total_ses_click_hit, total_ses_cart_hit, total_ses_order_hit]
        
        # scheduler.step()
        training_config["lr"] = getLr(optimizer)
        storeTrainingConfig(training_config)
        
    
    torch.save(model.state_dict(), Files.model_dir)
        
    
    print(f"Session start: {start_ses} Epoch: {i} Lr: {getLr(optimizer):.6f}| Loss:")
    start_ses += batch_size
    load_config["start_ses"] = start_ses
    storeLoadConfig(load_config)
    
    if getLr(optimizer) < 0.000001:
        print("Training finished")
        break
    
    if not_improve_cnt >= 100:
        print("Training finished")
        break
    
    
    
    