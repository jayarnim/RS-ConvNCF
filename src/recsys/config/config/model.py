from dataclasses import dataclass


@dataclass
class ConvNCFCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    channels: int
    dropout: float