import torch
import numpy as np
from tqdm import tqdm

X_train, y_train, meta_train = torch.load('data/train_sequences.pt')
X_val, y_val, meta_val = torch.load('data/val_sequences.pt')

alpha = 4.0  # Weighting factor, will be hyperparameter tuned later
weight_cap = torch.quantile(y_train, 0.95) #cap weights at 95th percentile


mean = y_train.mean()
dev = (y_train - mean).abs()
weights = 1 + alpha * dev / dev.max()
weights = torch.clamp(weights, max=weight_cap)