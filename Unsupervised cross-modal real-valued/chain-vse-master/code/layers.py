import torch
import torch.nn as nn


def l2norm(X):
    """L2-normalize columns of X
    """    
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    a = norm.expand_as(X)    
    X = torch.div(X, a)    
    return X