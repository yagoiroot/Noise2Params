"""
noise2image.var — sparse-difference helpers from the upstream Noise2Image
codebase.  Imported by :mod:`noise2image.utils`; not directly used by
the Noise2Params pipeline.  Retained for import-time parity.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import torch
from tqdm import tqdm
import time

def sparse_mean(sparse_tensor, dim=2):
    return sparse_tensor.sum(dim=dim) / sparse_tensor.shape[dim]

# @torch.jit.script
def sparse_diff(x, y, t, shape: tuple[int, int]=(720, 1280)):
    coordinate = x*shape[1] + y #convert to linear index
    stable_argsorted = torch.argsort(coordinate, stable=True) #argsort by linear index
    sorted_coordinate = coordinate[stable_argsorted] #sort coordinate
    sorted_t = t[stable_argsorted] #sort t by coordinate
    coord_switch = np.where(np.diff(sorted_coordinate) > 0)[0]+1 #find where coordinate changes
    t_diff = torch.diff(
        sorted_t,
        prepend=torch.zeros(1, device=x.device)
    ).to(dtype=sorted_t.dtype) #compute time difference between event times
    t_diff[coord_switch] = 0#set time difference to 0 where coordinate changes
    
    nevents = (
        -1 + #Subtract 1 to account for first event
        torch.sparse_coo_tensor(
            coordinate.unsqueeze(0),
            torch.ones_like(t_diff),
            size=(shape[0]*shape[1],)
        ).to_dense()
    ).clamp(min=1) #compute number of events per pixel. 

    means = torch.sparse_coo_tensor(
        coordinate.unsqueeze(0),
        t_diff,
        size=(shape[0]*shape[1],)
    ).to_dense() / nevents #compute mean time difference per pixel
    means = means[sorted_coordinate] #extract mean time difference per pixel in original order
    t_diff = (t_diff - means) ** 2 #compute squared difference from mean
    t_diff[coord_switch] = 0 #set squared difference to 0 where coordinate changes
        
    sp = torch.sparse_coo_tensor(
        coordinate.unsqueeze(0),
        t_diff,
        size=(shape[0]*shape[1],)
    ) #convert to sparse tensor for sum reduction
    
    return (sp.to_dense()/nevents.clamp(1)).reshape(shape)
