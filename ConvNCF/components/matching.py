import torch
import torch.nn as nn
from .aggregator import OuterProduct


class ConvoluationalCollaborativeFilteringLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list,
        dropout: float,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim
        self.channels = channels
        self.dropout = dropout

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        feature_map = self.agg(user_emb, item_emb)
        predictive_vec = self.cnn(feature_map)
        return predictive_vec

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        self.agg = OuterProduct()

        components = list(self._yield_conv_block())
        self.cnn = nn.Sequential(*components)

    def _yield_conv_block(self):
        IN_CHANNELS = 1
        OUT_CHANNELS = self.channels
        SPATIAL_SIZE = self.input_dim
        PADDING = 0
        KERNEL_SIZE = 2
        STRIDE = 2
        
        while SPATIAL_SIZE > 1:
            kwargs = dict(
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS, 
                kernel_size=KERNEL_SIZE, 
                stride=STRIDE,
                padding=PADDING,
            )
            yield nn.Sequential(
                nn.Conv2d(**kwargs),
                nn.BatchNorm2d(OUT_CHANNELS),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            SPATIAL_SIZE = (SPATIAL_SIZE + 2*PADDING - KERNEL_SIZE)//STRIDE + 1
            IN_CHANNELS = self.channels

        yield nn.Flatten()