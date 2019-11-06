import torch
import torch.utils.data
from model import *
from tqdm import *
from util import *



# Parameters
params = {'batch_size': 120,
'shuffle': False,
'num_workers': 6}

max_epochs = 200
