from typing import *
import torch
from torch import nn

class NetNiet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.batch0 = nn.BatchNorm1d(num_features=4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)        
        self.linear1 = nn.Linear(4,96,bias=True)
        self.batch1 = nn.BatchNorm1d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.silu= torch.nn.SiLU(inplace=False)
        self.linear6 = nn.Linear(96,1,bias=True)    
        self.relu = nn.ReLU(inplace=False)
        self.elu = nn.ELU(alpha=1.0, inplace=False)
        
    def forward(self, x):
        
        x=self.batch0(x)
         
        x=self.linear1(x)
        
        x=self.batch1(x)

        x=self.elu(x)
                
        x=self.linear6(x)
              
        return x
        