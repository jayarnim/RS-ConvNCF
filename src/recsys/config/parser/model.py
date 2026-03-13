from ..config.model import (
    ConvNCFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="convncf":
        return convncf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def convncf(cfg):
    return ConvNCFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        channels=cfg["model"]["channels"],
        dropout=cfg["model"]["dropout"],
    )
