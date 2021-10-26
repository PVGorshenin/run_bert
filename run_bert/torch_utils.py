import torch
import pandas as pd
from torch.nn import Sigmoid


def to_numpy(x_tensor):
    return x_tensor.cpu().detach().numpy()


def to_probits(arr):
    return Sigmoid()(torch.Tensor(arr))



