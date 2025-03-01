import json 
import os
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
import itertools
warnings.filterwarnings("ignore")

from utils import *

checkDir("data")

class Files:
    # Data path for model training
    train = "train.jsonl"
    data_chunks_dir = "data_chunk/"
    data_info = "datainfo.json"

def getDataInfo():
    # Get data information
    num_sessions = numSessions()
    max_item_id = itemsMAXID()
    chunk_size = 10000
    return {
        "num_sessions": num_sessions,
        "max_item_id": max_item_id,
        "chunk_size": chunk_size
    }

def storeDataInfo(data_info):
    # Store data information
    with open(Files.data_info, "w") as f:
        f.write(json.dumps(data_info))

def loadDataInfo():
    # Load data informatioon
    with open(Files.data_info, "r") as f:
        data_info = json.loads(f.read())
    return data_info

def itemsMAXID() -> int:
    # Find items max id
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
    # Get item ids
    filename = Files.train
    items = set()
    with open(filename, "r", encoding="utf-8") as file:
        for line in tqdm(file): 
            session = json.loads(line)  
            for ev in session["events"]:
                items.add(ev["aid"])
    
    return items

def getSessions(trainfile) -> set:
    # Get session id
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
    # Number of session
    trainfile = "train.jsonl"
    file_data_info = "datainfo.json"
    sum = 0
    if os.path.exists(file_data_info):
        with open(file_data_info, "r") as f:
            return json.loads(f.read())["num_sessions"]
    else:
        with open(trainfile, "r", encoding="utf-8") as file:
            for line in file:
                sum += 1
        return sum


# Datas information
class DataSplit():
    # Data path for model train
    for attr in dir(Files):  
        if not attr.startswith("__") and isinstance(getattr(Files, attr), str):  
            setattr(Files, attr, "../" + getattr(Files, attr))
    if not os.path.exists(Files.data_info):
        print(f"Data info file not exist: create {Files.data_info}")
        data_info = getDataInfo()
        storeDataInfo(data_info)
    data_info = loadDataInfo()
    chunk_size = data_info["chunk_size"]
    max_item_id = data_info["max_item_id"]
    num_sessions = data_info["num_sessions"]
    chunk_num = num_sessions // chunk_size + 1
    
    def __init__(self):
        pass
        
    def checkPath(self):
        if not os.path.exists(Files.data_chunks_dir):
            os.makedirs(Files.data_chunks_dir)
    
    def storeDataChunks(self):

        chunk_ids = list(range(self.chunk_num + 1))
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        for _ in tqdm(pool.imap_unordered(self.store, chunk_ids), total=len(chunk_ids)):
            pass
        pool.close()
    
    def store(self, chunk_id, test = False):
        # load line chunks
        batch_filename = f"{Files.data_chunks_dir}{chunk_id}.json"
        
        with open(Files.train, "r", encoding="utf-8") as lf:
            with open(batch_filename, "w", encoding="utf-8") as sf:
                
                # Skip lines efficiently
                start = chunk_id * self.chunk_size
                end = (chunk_id + 1) * self.chunk_size

                lines = itertools.islice(lf, start, end)  

                for line in lines:
                    sf.write(json.dumps(json.loads(line)) + "\n")
                
        
        if test:
            with open(batch_filename, "r", encoding="utf-8") as f:
                d = json.load(f)
            id = []
            for ses in d:
                id.append(ses['session'])
            print(min(id) == start, max(id) == end - 1, len(id) == self.chunk_size)
            
    @staticmethod
    def loadChunkData(filename) -> list:
        trainset = []
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line) 
                trainset.append(data)
                
        return trainset

if __name__ == "__main__":
    split = DataSplit()
    split.checkPath()
    split.storeDataChunks()
                
                