from typing import *
import torch
from torch import nn

class NetNiet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_batch = nn.BatchNorm1d(num_features=4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        
        self.block1 = Block(4, 12)
        self.block2 = Block(12, 48)
        self.block3 = Block(48, 128)
        self.block4 = Block(128, 48)
        self.block5 = Block(48, 12)

        self.final_linear = nn.Linear(12, 1, bias=True)    

        
    def forward(self, x):
        x = self.input_batch(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
                
        x = self.final_linear(x)
              
        return x

class Block(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.01)
        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.leaky_relu = torch.nn.LeakyReLU(inplace=False)

        self.batch = nn.BatchNorm1d(num_features=out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

    def forward(self, x):
        x = self.dropout(x)

        x = self.linear(x)

        x = self.leaky_relu(x)

        x = self.batch(x)

        return x
        