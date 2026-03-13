import torch
import torch.nn as nn


class OuterProduct(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        # (B,D,1)
        user_emb_exp = user_emb.unsqueeze(-1)
        # (B,1,D)
        item_emb_exp = item_emb.unsqueeze(-2)
        # (B,D,D)
        feature_map = torch.bmm(user_emb_exp, item_emb_exp)
        # (B,1,D,D)
        feature_map_exp = feature_map.unsqueeze(1)
        return feature_map_exp