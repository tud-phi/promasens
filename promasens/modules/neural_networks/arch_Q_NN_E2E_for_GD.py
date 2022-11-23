import torch
from torch import nn
#This is the same architecture used for the E2E, but then with different intial nodes suited for the 3NN


class NetNiet(nn.Module):
    def __init__(self, state_dim: int = 5):
        super().__init__()
        
        self.nodes = 25
        
        self.linear1 = nn.Linear(5,self.nodes,bias=True)
        self.linear6 = nn.Linear(self.nodes,3,bias=True)    
        
        self.batch0 = nn.BatchNorm1d(num_features=5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
        self.batch1 = nn.BatchNorm1d(num_features=self.nodes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
  
        # self.relu= torch.nn.functional.relu
        # self.elu = torch.nn.ELU(alpha=1.0, inplace=False)
        self.silu= torch.nn.SiLU(inplace=False)
        # self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        
    def forward(self, x):
        
        x=self.batch0(x)
         
        x=self.linear1(x)
        
        x=self.batch1(x)

        x=self.silu(x)
        #x=self.relu(x)
                
        x=self.linear6(x)
              
        return x