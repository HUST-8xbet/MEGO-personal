"""Utilities for rank-matched Max-Cut -> TSP latent mapping."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.regression.surrogate_vae import SurrogateVAE


class LatentBridge(nn.Module):
    """Map a source expert latent to the fixed TSP expert latent size."""

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        if source_dim < 1 or target_dim < 1:
            raise ValueError("latent dimensions must be positive")
        self.source_dim = int(source_dim)
        self.target_dim = int(target_dim)
        self.projection = nn.Linear(self.source_dim, self.target_dim)

    def forward(self, latent):
        if latent.ndim != 2 or latent.shape[1] != self.source_dim:
            raise ValueError("source latent has an invalid shape")
        if not torch.isfinite(latent).all():
            raise ValueError("source latent contains non-finite values")
        return self.projection(latent)


@dataclass
class RankMatchedPairs:
    source_latent: np.ndarray
    source_quality: np.ndarray
    target_tours: np.ndarray
    target_quality: np.ndarray
    source_order: np.ndarray
    target_order: np.ndarray


def rank_matched_pairs(source_latent, source_quality, target_tours, target_quality,
                       max_pairs=None):
    """Pair source and target samples by descending, higher-is-better quality.

    Max-Cut quality is its native score. TSP quality is expected to be
    ``-length``. The function deliberately does not compare raw score units.
    """
    source_latent = np.asarray(source_latent, dtype=np.float32)
    source_quality = np.asarray(source_quality, dtype=np.float64)
    target_tours = np.asarray(target_tours)
    target_quality = np.asarray(target_quality, dtype=np.float64)
    if source_latent.ndim != 2 or source_quality.shape != (len(source_latent),):
        raise ValueError("source latent and quality shapes do not match")
    if target_tours.ndim != 2 or target_quality.shape != (len(target_tours),):
        raise ValueError("target tours and quality shapes do not match")
    if len(source_latent) == 0 or len(target_tours) == 0:
        raise ValueError("cannot pair empty data")
    if not (np.isfinite(source_quality).all() and np.isfinite(target_quality).all()):
        raise ValueError("qualities must be finite")
    source_order = np.argsort(-source_quality, kind="stable")
    target_order = np.argsort(-target_quality, kind="stable")
    count = min(len(source_order), len(target_order))
    if max_pairs is not None:
        if isinstance(max_pairs, (bool, np.bool_)) or int(max_pairs) != max_pairs or max_pairs < 2:
            raise ValueError("max_pairs must be an integer >= 2")
        count = min(count, int(max_pairs))
    source_order, target_order = source_order[:count], target_order[:count]
    return RankMatchedPairs(
        source_latent=source_latent[source_order].copy(),
        source_quality=source_quality[source_order].copy(),
        target_tours=target_tours[target_order].astype(np.int64, copy=True),
        target_quality=target_quality[target_order].copy(),
        source_order=source_order.copy(), target_order=target_order.copy())


def load_maxcut_latent(checkpoint, hyperparameters, x_path, y_path, device):
    """Load a legacy MEGO Max-Cut VAE and return posterior means."""
    config = dict(hyperparameters)
    model_config = dict(config["model_params"])
    model_config.pop("name", None)
    model_config.pop("latent_dim_coefficient", None)
    model = SurrogateVAE(**model_config).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    x = np.load(x_path).astype(np.float32, copy=False)
    quality = np.load(y_path).astype(np.float64, copy=False)
    if x.ndim != 2 or len(x) != len(quality):
        raise ValueError("Max-Cut data shapes do not match")
    with torch.no_grad():
        latent, _ = model.encode(torch.as_tensor(x, device=device))
    return latent.cpu().numpy(), quality, model_config


def set_decoder_trainable(model):
    """Freeze TSP encoder/score head; keep the existing TSP decoder trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.city_embedding, model.decoder_init, model.decoder_cell,
                   model.key, model.query, model.pointer):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
