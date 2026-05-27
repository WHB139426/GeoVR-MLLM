import torch
import json
import random
import os
import sys
import csv
import numpy as np
from pathlib import Path
from torch.backends import cudnn

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def load_csv(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(dict(row))
    return data

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_jsonl(path):
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return data

def save_json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num} 