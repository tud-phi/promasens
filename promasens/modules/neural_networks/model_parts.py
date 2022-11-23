import torch
from torch import nn


class FeedforwardReluBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.01)
        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.relu = torch.nn.ReLU(inplace=False)

        self.batch = nn.BatchNorm1d(num_features=out_features, eps=1e-05, momentum=0.1, affine=True,
                                    track_running_stats=True)

    def forward(self, x):
        x = self.dropout(x)
        
        x = self.linear(x)

        x = self.relu(x)

        x = self.batch(x)

        return x


class FeedforwardLeakyReluBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.01)
        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.leaky_relu = torch.nn.LeakyReLU(inplace=False)

        self.batch = nn.BatchNorm1d(num_features=out_features, eps=1e-05, momentum=0.1, affine=True,
                                    track_running_stats=True)

    def forward(self, x):
        x = self.dropout(x)

        x = self.linear(x)

        x = self.leaky_relu(x)

        x = self.batch(x)

        return x
