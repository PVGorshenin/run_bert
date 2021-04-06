import torch
import pandas as pd
from torch.nn import Sigmoid


def to_numpy(x_tensor):
    return x_tensor.numpy()


def to_probits(arr):
    return Sigmoid()(torch.Tensor(arr))



