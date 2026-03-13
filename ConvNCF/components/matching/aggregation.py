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
        mat = torch.bmm(user_emb_exp, item_emb_exp)
        # (B,1,D,D)
        return mat.unsqueeze(1)