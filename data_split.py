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

# Datas information
class DataSplit():
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
                
                