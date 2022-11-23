import torch
from torch import nn

from .model_parts import FeedforwardReluBlock


class NetNiet(nn.Module):
    def __init__(self, state_dim: int = 5):
        super().__init__()

        self.input_batch = nn.BatchNorm1d(num_features=state_dim, eps=1e-05, momentum=0.1, affine=True,
                                          track_running_stats=True)

        self.block1 = FeedforwardReluBlock(state_dim, 96)
        self.block2 = FeedforwardReluBlock(96, 256)
        self.block3 = FeedforwardReluBlock(256, 64)
        self.block4 = FeedforwardReluBlock(64, 24)

        self.final_linear = nn.Linear(24, 1, bias=True)

    def forward(self, x):
        x = self.input_batch(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.final_linear(x)

        # intuition behind usage of exponential function:
        # the magnetic flux density is highly nonlinear with respect to the distance from the magnet
        # that makes it difficult to use the same neural network for sensors placed at different distances from the magnet
        # the exponential function should allow the neural network to learn a more "linear" latent representation of the flux density
        # the exponential function then allows it to be scaled
        x = torch.sign(x) * torch.exp(torch.abs(x))

        return x
