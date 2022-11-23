from torch import nn

from .model_parts import FeedforwardReluBlock
from .model_parts import FeedforwardLeakyReluBlock




class NetNiet(nn.Module):
    def __init__(self, state_dim: int = 5):
        super().__init__()

        self.input_batch = nn.BatchNorm1d(num_features=state_dim, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)

        self.block1 = FeedforwardReluBlock(state_dim, 50)
        self.block2 = FeedforwardReluBlock(50, 150)
        self.block3 = FeedforwardReluBlock(150, 300)
        self.block4 = FeedforwardReluBlock(300, 150)
        self.block5 = FeedforwardReluBlock(150, 50)
        self.block6 = FeedforwardReluBlock (50, 24)

        self.final_linear = nn.Linear(24, 1, bias=True)

    def forward(self, x):
        x = self.input_batch(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.final_linear(x)

        return x
