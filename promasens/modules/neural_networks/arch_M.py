from torch import nn


from .model_parts import FeedforwardReluBlock

class NetNiet(nn.Module):
    def __init__(self, state_dim: int = 5, **kwargs):
        super().__init__()

        self.input_batch  = nn.BatchNorm1d(num_features=state_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.block1 = FeedforwardReluBlock(state_dim, 96)
        self.block2 = FeedforwardReluBlock(96, 256)
        
        self.final_linear = nn.Linear(256,1,bias=True)
        
    def forward(self, x):
        x = self.input_batch(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_linear(x)
              
        return x
        

