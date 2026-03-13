import torch
import torch.nn as nn
from collections.abc import Iterator


class ConvoluationalCollaborativeFilteringLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        channel: int,
        dropout: float,
    ):
        super().__init__()

        kwargs = dict(
            input_dim=embedding_dim,
            channel=channel,
            dropout=dropout,
        )
        components = list(conv_block(**kwargs))
        self.cnn = nn.Sequential(*components)

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        # OUTER PRODUCT ==========
        # (B,D) -> (B,D,1)
        user_emb_exp = user_emb.unsqueeze(-1)
        # (B,D) -> (B,1,D)
        item_emb_exp = item_emb.unsqueeze(-2)
        # (B,D,1) x (B,1,D) -> (B,D,D)
        mat = torch.bmm(user_emb_exp, item_emb_exp)
        # (B,D,D) -> (B,1,D,D)
        X = mat.unsqueeze(1)
        
        # CONVOLUTION NETWORKS ==========
        return self.cnn(X)


def conv_block(
    input_dim: int, 
    channel: int,
    dropout: float,
) -> Iterator[nn.Sequential]:
    IN_CHANNELS = 1
    OUT_CHANNELS = channel
    SPATIAL_SIZE = input_dim
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
            nn.Dropout2d(dropout),
        )
        SPATIAL_SIZE = (SPATIAL_SIZE + 2*PADDING - KERNEL_SIZE)//STRIDE + 1
        IN_CHANNELS = OUT_CHANNELS

    yield nn.Flatten()