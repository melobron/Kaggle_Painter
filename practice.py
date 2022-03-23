import torch
import torch.nn as nn
from torchvision import models
import numpy as np

batch_size = 2
diag = np.eye(2 * batch_size)
l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
mask = torch.from_numpy((diag + l1 + l2))
mask = (1 - mask).type(torch.bool)

print(mask)