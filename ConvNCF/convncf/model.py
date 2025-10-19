import numpy as np
import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int=32,
        dropout: float=0.2,
        channels: int=16,
    ):
        """
        Outer product-based neural collaborative filtering (He et al., 2018)
        -----
        Implements the base structure of Convolutional Neural Collaborative Filtering (ConvNCF),
        CNN & id embedding based latent factor model.

        Args:
            n_users (int): 
                total number of users in the dataset, U.
            n_items (int): 
                total number of items in the dataset, I.
            n_factors (int): 
                dimensionality of user and item latent representation vectors, K.
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
            channels (int): 
                number of convolutional feature maps (output channels) used in the CNN layers.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.dropout = dropout
        self.channels = channels

        # debugging args error
        self._assert_arg_error()

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        return self.score(user_idx, item_idx)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        pred_vector = self.conv(user_idx, item_idx)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def conv(self, user_idx, item_idx):
        feature_map = self.outer_product(user_idx, item_idx)
        pred_vector = self.matching_fn(feature_map)
        return pred_vector

    def outer_product(self, user_idx, item_idx):
        # (B, D, 1)
        user_embed_slice_exp = self.user_embed(user_idx).unsqueeze(2)
        # (B, 1, D)
        item_embed_slice_exp = self.item_embed(item_idx).unsqueeze(1)
        # (B, D, D)
        feature_map = torch.bmm(user_embed_slice_exp, item_embed_slice_exp)
        # (B, 1, D, D)
        feature_map_exp = feature_map.unsqueeze(1)
        return feature_map_exp

    def _set_up_components(self):
        self._create_embeddings()
        self._init_embeddings()
        self._create_layers()

    def _create_embeddings(self):
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

    def _init_embeddings(self):
        nn.init.normal_(self.user_embed.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed.weight, mean=0.0, std=0.01)

    def _create_layers(self):
        components = list(self._yield_conv_block(self.n_factors, self.channels))
        self.matching_fn = nn.Sequential(*components)

        kwargs = dict(
            in_features=self.channels,
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)

    def _yield_conv_block(self, n_factors, out_channels):
        hidden = n_factors
        idx = 0
        
        while hidden > 1:
            kwargs = dict(
                in_channels=1 if idx==0 else out_channels,
                out_channels=out_channels, 
                kernel_size=2, 
                stride=2,
            )
            yield nn.Sequential(
                nn.Conv2d(**kwargs),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(self.dropout),
            )
            hidden //= 2
            idx += 1

        yield nn.Flatten()

    def _assert_arg_error(self):
        CONDITION = (self.n_factors & (self.n_factors - 1) == 0)
        ERROR_MESSAGE = f"embedding dim must match be power of 2: {self.n_factors}"
        assert CONDITION, ERROR_MESSAGE