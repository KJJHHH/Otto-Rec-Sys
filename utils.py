import json, pickle
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm

import torch

import warnings
warnings.filterwarnings("ignore")

from files_name import Files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type_enum = {
    "clicks": 0,
    "carts": 1,
    "orders": 2
}
    
# Data info
def getDataInfo():
    num_sessions = numSessions()
    max_item_id = itemsMAXID()
    chunk_size = 10000
    return {
        "num_sessions": num_sessions,
        "max_item_id": max_item_id,
        "chunk_size": chunk_size
    }

def storeDataInfo(data_info):
    with open(Files.data_info, "w") as f:
        f.write(json.dumps(data_info))

def loadDataInfo():
    with open(Files.data_info, "r") as f:
        data_info = json.loads(f.read())
    return data_info

def itemsMAXID() -> int:
    filename = Files.data_info
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.loads(f.read())["max_item_id"]
    else:
        items = getItems()
        with open(filename, "w") as f:
            f.write(json.dumps({"max_item_id": max(items) + 1}))
        return max(items) + 1

def getItems() -> set:
    filename = Files.train
    items = set()
    with open(filename, "r", encoding="utf-8") as file:
        for line in tqdm(file): 
            session = json.loads(line)  
            for ev in session["events"]:
                items.add(ev["aid"])
    
    return items

def getSessions(trainfile) -> set:
    sessions = set()
    with open(trainfile, "r", encoding="utf-8") as file:
        for id, line in tqdm(enumerate(file)):
            try:
                data = json.loads(line)  # Convert JSON string to dictionary
                sessions.add(data["session"])
            except: 
                print("bad lines")
                continue
    return sessions

def numSessions() -> int:
    trainfile = Files.train
    sum = 0
    if os.path.exists(Files.data_info):
        with open(Files.data_info, "r") as f:
            return json.loads(f.read())["num_sessions"]
    else:
        with open(trainfile, "r", encoding="utf-8") as file:
            for line in file:
                sum += 1
        return sum

def checkDir(type = "train"):
    directories = [Files.data_chunks_dir, Files.tmp, Files.model_dir]
    if type == "data":
        directories = [Files.data_chunks_dir]
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Dir not exist: {directory}")
            os.makedirs(directory)

# Load config: load train info
def storeConfig(config):
    with open(Files.config, "wb") as f:
        pickle.dump(config, f)

def loadModel(model):
    checkpoint_files = [f for f in os.listdir(Files.model_dir)]

    checkpoint_numbers = [int(f.split('-')[-1].split('.pt')[0]) for f in checkpoint_files]
    if checkpoint_files != []:
        latest_ckpt = f"ckpt-best-session-{max(checkpoint_numbers)}.pt"
        checkpoint_path = os.path.join(Files.model_dir, latest_ckpt)
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=True))
        print(f"Loaded checkpoint: {latest_ckpt}")
        
    return model

def getLr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Clear data
def getFutureEvents(session: dict) -> (list, list): 
    """Input and target for predict model for each  session

    Args:
        session (dict): a session in trainset

    Returns:
        X, y: for model predict training
    """
    X, y = [], []
    aids, ts, types = [], [], []
    for ev in session["events"]:
        aids.append(ev["aid"])
        ts.append(ev["ts"])
        types.append(ev["type"])
        
    for i in range(len(session["events"])):
        if i == len(session["events"]) - 1:
            break
        
        x = [session["events"][i]["aid"], type_enum[session["events"][i]["type"]]]
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
        
        # If last event
        if len(y_click) == 0 and len(y_cart) == 0 and len(y_order) == 0:
            break
        
        X.append(x)
        y.append((y_click, y_cart, y_order))
        
    return X, y

def trainData(trainset: list) -> tuple:    
    """Get the input and target of predict model

    Args:
        trainset: list of sessions

    Returns:
        result: ([X, y], [X, y], ...)
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(getFutureEvents, tqdm(trainset))
    pool.close()
    
    return results

def padToTensor(indices: list, vocab_size: int) -> torch.tensor:
    binary_matrix = torch.zeros((vocab_size), dtype=torch.float32)
    for index in indices:
        binary_matrix[index] = 1  
    return binary_matrix

# Train
# RL loss
def computeLoss(out, target, stat: list):
    # total_ses_click_hit, total_ses_cart_hit, total_ses_order_hit = stat[0], stat[1], stat[2]
    
    reward = torch.tensor(0.).to(device)
    click, cart, order = target[0].to(device), target[1].to(device), target[2].to(device)
    
    # RL method
    """
    top_values, selected_indices = torch.topk(out, 20)
    success_click = torch.tensor([x for x in click if x in selected_indices], dtype=torch.int32)
    success_carts = torch.tensor([x for x in cart if x in selected_indices], dtype=torch.int32)
    success_orders = torch.tensor([x for x in order if x in selected_indices], dtype=torch.int32)
    fail_select = torch.tensor([x for x in selected_indices if x not in torch.cat((click, cart, order), dim = 0)], dtype=torch.int32)
    if success_click.numel() != 0:
        reward += torch.sum(out[click]) * 10
        stat[0] += 1
    if success_carts.numel() != 0:
        reward += torch.sum(out[success_carts])/cart.numel() * 30
        stat[1] += success_carts.numel()
    if success_orders.numel() != 0:
        reward += torch.sum(out[success_orders])/order.numel() * 60
        stat[2] += success_orders.numel()
    if fail_select.numel() != 0:
        reward -= torch.sum(out[fail_select]) * 100
        
    # Match the ground true
    """
    top_values, selected_indices = torch.topk(out, 20)
    success_click = torch.tensor([x for x in click if x in selected_indices], dtype=torch.int32)
    success_carts = torch.tensor([x for x in cart if x in selected_indices], dtype=torch.int32)
    success_orders = torch.tensor([x for x in order if x in selected_indices], dtype=torch.int32)
    if success_click.numel() != 0:
        stat[0] += 1
    if success_carts.numel() != 0:
        stat[1] += success_carts.numel()
    if success_orders.numel() != 0:
        stat[2] += success_orders.numel()
        
    all_indices = torch.arange(out.size(0)).to(device)
    not_indices = ~torch.isin(all_indices, torch.cat((click, cart, order), dim = 0))
    reward +=  torch.sum(out[click]) * 1
    reward += torch.sum(out[cart]) * 3
    reward += torch.sum(out[order]) * 6
    reward -= torch.sum(out[not_indices]) / not_indices.numel()
    
    return - reward



"""
def loadConfig() -> dict:
    file = Files
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    else:
        return {
            "start_ses": 0,
            "batch_size": 10000,
            "not_improve_cnt": 0}

def storeLoadConfig(info: dict):
    file = Files.load_info
    with open(file, 'w') as f:
        json.dump(info, f)
        
def trainingConfig() -> dict:    
    file = Files.training_config
    if os.path.exists(file):
        with open(file, 'r') as f:
            return json.load(f)
    else:
        return {
            "epochs": 20, 
            "lr": 0.001, 
            "batch_size": 128, 
            "embed_size": 20, 
            "vocab_size": itemsMAXID(), 
            "scheduler_step": 10, 
            "scheduler_gamma": 0.5}
    
def storeTrainingConfig(config: dict):
    file = Files.training_config
    with open(file, 'w') as f:
        json.dump(config, f)
    
"""