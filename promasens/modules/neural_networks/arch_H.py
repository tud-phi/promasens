from typing import *
import torch
from torch import nn

class NetNiet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(4,96,bias=True)
        self.linear2 = nn.Linear(24,96,bias=True)
        self.linear3 = nn.Linear(96,128,bias=True)
        self.linear4 = nn.Linear(128,96,bias=True)
        self.linear5 = nn.Linear(96,24,bias=True)
        self.linear6 = nn.Linear(96,1,bias=True)    
        
        
        self.batch0 = nn.BatchNorm1d(num_features=4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch1 = nn.BatchNorm1d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch2 = nn.BatchNorm1d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch3 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch4 = nn.BatchNorm1d(num_features=96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        
        self.drop = nn.Dropout(p=0.2)

        self.leaky = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.relu= torch.nn.functional.relu
        self.elu = torch.nn.ELU(alpha=1.0, inplace=False)
        self.silu= torch.nn.SiLU(inplace=False)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        
    def forward(self, x):
        
        x=self.batch0(x)
        
        x=self.drop(x)
         
        x=self.linear1(x)
        
        x=self.batch1(x)

        x=self.leaky(x)
        
        x=self.drop(x)
                
        x=self.linear6(x)
              
        return x
        