import torch
import torch.nn as nn
from components.base import BaseModel
from .layers.embedding import build as build_embedding_layer
from .layers.matching import ConvoluationalCollaborativeFilteringLayer
from .layers.prediction import ProjectionLayer


class ConvolutionalNeuralCollaborativeFiltering(BaseModel):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        channel: int,
        dropout: float,
    ):
        """
        Outer product-based neural collaborative filtering (He et al., 2018)
        -----
        Implements the base structure of Convolutional Neural Collaborative Filtering (ConvNCF),
        CNN & id embedding based latent factor model.

        Args:
            num_users (int): 
                total number of users in the dataset, U.
            num_items (int): 
                total number of items in the dataset, I.
            hidden_dim (int): 
                dimensionality of user and item latent representation vectors, K.
            channel (int): 
                number of convolutional feature maps (output channels) used in the CNN layers.
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__(locals())

        # IDX EMBEDDING ==========
        self.embedding = build_embedding_layer(
            name="idx",
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # MATCHING FUNCTION LEARNING ==========
        self.matching = ConvoluationalCollaborativeFilteringLayer(
            embedding_dim=embedding_dim,
            channel=channel,
            dropout=dropout,
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=channel,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # IDX EMBEDDING ==========
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        # MATCHING FUNCTION LEARNING ==========
        X_pred = self.matching(user_emb, item_emb)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit