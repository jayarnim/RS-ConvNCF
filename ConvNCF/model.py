import torch
import torch.nn as nn
from .components.embedding import IDXEmbedding
from .components.matching import ConvoluationalCollaborativeFilteringLayer
from .components.scorer import LinearProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        channels: int,
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
            channels (int): 
                number of convolutional feature maps (output channels) used in the CNN layers.
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.channels = channels
        self.dropout = dropout

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        predictive_vec = self.matching(user_emb, item_emb)
        return predictive_vec

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        predictive_vec = self.forward(user_idx, item_idx)
        logit = self.scorer(predictive_vec)
        return logit

    @property
    @torch.no_grad()
    def user_emb(self):
        user_emb = self.embedding.user_emb
        return user_emb

    @property
    @torch.no_grad()
    def item_emb(self):
        item_emb = self.embedding.item_emb
        return item_emb

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
        )
        self.embedding = IDXEmbedding(**kwargs)

        kwargs = dict(
            input_dim=self.embedding_dim,
            channels=self.channels,
            dropout=self.dropout,
        )
        self.matching = ConvoluationalCollaborativeFilteringLayer(**kwargs)

        kwargs = dict(
            input_dim=self.channels,
        )
        self.scorer = LinearProjectionLayer(**kwargs)