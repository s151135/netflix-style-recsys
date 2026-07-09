from __future__ import annotations


class TorchUnavailableError(RuntimeError):
    pass


def require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise TorchUnavailableError(
            "Install the 'ml' extra to train the optional two-tower retriever."
        ) from exc


class TwoTowerFactory:
    """Factory placeholder so the production interface exists before Torch is installed."""

    @staticmethod
    def build(user_encoder, item_encoder):
        torch = require_torch()

        class TwoTower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.user_encoder = user_encoder
                self.item_encoder = item_encoder

            def forward(self, batch):
                return self.user_encoder(batch["user_feats"]), self.item_encoder(batch["item_feats"])

        return TwoTower()
