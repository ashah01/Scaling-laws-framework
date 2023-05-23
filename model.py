import torch
import torch.nn as nn
import torch.nn.functional as F


def init_params(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Residual(nn.Module):
    """
    This module looks like what you find in the original resnet or IC paper
    (https://arxiv.org/pdf/1905.05928.pdf), except that it's based on MLP, not CNN.
    If you flag `only_MLP` as True, then it won't use any batch norm, dropout, or
    residual connections

    """

    def __init__(
        self,
        num_features: int,
        dropout: float,
        i: int,
    ):
        super().__init__()
        self.num_features = num_features

        self.i = i

        if not (self.i == 0):
            self.norm_layer1 = nn.LayerNorm(num_features)
            self.dropout1 = nn.Dropout(p=dropout)
            self.linear1 = nn.Linear(num_features, num_features)
        else:
            self.linear1 = nn.Linear(32*32*3, num_features)
        self.relu1 = nn.ReLU()

        self.norm_layer2 = nn.LayerNorm(num_features)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(num_features, num_features)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.Tensor:
        out = x

        if not (self.i == 0):
            identity = out
            out = self.norm_layer1(out)
            out = self.dropout1(out)

        out = self.linear1(out)
        out = self.relu1(out)

        if self.i == 0:
            identity = out

        out = self.norm_layer2(out)
        out = self.dropout2(out)
        out = self.linear2(out)

        out += identity

        out = self.relu2(out)

        return out


class DownSample(nn.Module):
    """
    This module is an MLP, where the number of output features is lower than
    that of input features. If you flag `only_MLP` as False, it'll add norm
    and dropout

    """

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        assert in_features > out_features

        self.in_features = in_features
        self.out_features = out_features

        self.norm_layer = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.Tensor:
        out = x

        out = self.norm_layer(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.relu(out)

        return out


class ResMLP(nn.Module):
    """
    MLP with optinally batch norm, dropout, and residual connections. I got
    inspiration from the original ResNet paper and https://arxiv.org/pdf/1905.05928.pdf.

    Downsampling is done after every block so that the features can be encoded
    and compressed.

    """

    def __init__(
        self,
        dropout: float,
        num_blocks: int,
        num_initial_features: int,
    ):
        super().__init__()

        blocks = []

        # input layer
        blocks.append(nn.Flatten())

        for i in range(num_blocks):
            blocks.extend(
                self._create_block(
                    num_initial_features,
                    dropout,
                    i,
                )
            )
            num_initial_features //= 2

        # last classiciation layer
        blocks.append(nn.Linear(num_initial_features, 10))

        self.blocks = nn.Sequential(*blocks)
        self.apply(init_params)

    def _create_block(
        self,
        in_features: int,
        dropout: float,
        i: int,
    ) -> list:
        block = []
        block.append(Residual(in_features, dropout, i))
        block.append(DownSample(in_features, in_features // 2, dropout))

        return block

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.blocks(x)


class TransMLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 32 * 3, width), nn.ReLU()
        )
        self.middle_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.middle_layers.append(
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU())
            )
        
        self.output_layer = nn.Linear(width, 10)
        self.apply(init_params)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.middle_layers:
            x = x + layer(x)
        x = self.output_layer(x)
        return x
