import numpy as np
import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int=32,
        channels: int=16,
        dropout: float=0.2,
    ):
        super().__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.channels = channels
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector = self.conv(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def conv(self, user_idx, item_idx):
        feature_map = self.outer_product(user_idx, item_idx)
        pred_vector = self.conv_layers(feature_map)
        return pred_vector

    def outer_product(self, user_idx, item_idx):
        user_slice_exp = self.user_embed(user_idx).unsqueeze(2)     # (B, D, 1)
        item_slice_exp = self.item_embed(item_idx).unsqueeze(1)     # (B, 1, D)
        feature_map = torch.bmm(user_slice_exp, item_slice_exp)     # (B, D, D)
        feature_map_exp = feature_map.unsqueeze(1)                  # (B, 1, D, D)
        return feature_map_exp

    def _init_layers(self):
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.user_embed = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_embed = nn.Embedding(**kwargs)

        nn.init.normal_(self.user_embed.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed.weight, mean=0.0, std=0.01)

        self.conv_layers = nn.Sequential(
            *list(self._generate_layers(self.n_factors, self.channels)),
            nn.Flatten(),
            nn.Dropout(self.dropout),
        )

        self.logit_layer = nn.Linear(
            in_features=self.channels,
            out_features=1,
        )

    def _generate_layers(self, n_factors, channels):
        hidden = n_factors
        idx = 0

        while hidden > 1:
            if idx==0:
                yield nn.Conv2d(
                    in_channels=1, 
                    out_channels=channels, 
                    kernel_size=2, 
                    stride=2,
                )

            else:
                yield nn.Conv2d(
                    in_channels=channels, 
                    out_channels=channels, 
                    kernel_size=2, 
                    stride=2,
                )

            yield nn.ReLU()

            hidden //= 2
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.n_factors & (self.n_factors - 1) == 0)
        ERROR_MESSAGE = f"embedding dim must match be power of 2: {self.n_factors}"
        assert CONDITION, ERROR_MESSAGE