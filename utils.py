import json, pickle
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm

import torch

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type_enum = {
    "clicks": 0,
    "carts": 1,
    "orders": 2
}
    
def checkDir(directories:list):
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Dir not exist: {directory}")
            os.makedirs(directory)

# Load config: load train status
def storeConfig(filename, tmp_config):
    with open(filename, "w") as f:
        json.dump(tmp_config, f)
    
def loadConfig(filename):
    print("Load previous training status")
    with open(filename, "r") as f:
        tmp_config = json.load(f)
    return tmp_config

def loadModel(model, model_dir):
    checkpoint_files = [f for f in os.listdir(model_dir)]

    checkpoint_numbers = [int(f.split('-')[-1].split('.pt')[0]) for f in checkpoint_files]
    if checkpoint_files != []:
        latest_ckpt = f"ckpt-best-session-{max(checkpoint_numbers)}.pt"
        checkpoint_path = os.path.join(model_dir, latest_ckpt)
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
