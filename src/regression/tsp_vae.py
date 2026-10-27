"""Milestone B: instance-conditioned permutation VAE; independent of binary VAE."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def canonicalize_tours(tours):
    """Choose city 0 first and the direction with second city < last city.

    This gives one sequence per undirected cycle with distinct city indices.
    It is a representation convention, not a learned symmetry-invariant model.
    """
    tours = np.asarray(tours)
    if tours.ndim != 2 or tours.shape[1] < 3 or tours.dtype.kind not in "iu":
        raise ValueError("tours must be integer permutations of shape (batch, n>=3)")
    n = tours.shape[1]
    if not np.all(np.sort(tours, axis=1) == np.arange(n)):
        raise ValueError("each tour must contain every city exactly once")
    result = np.empty(tours.shape, dtype=np.int64)
    for i, tour in enumerate(tours):
        rotated = np.roll(tour, -int(np.flatnonzero(tour == 0)[0]))
        result[i] = rotated if rotated[1] < rotated[-1] else np.r_[0, rotated[:0:-1]]
    return result


class TSPVAE(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=16):
        super().__init__()
        if hidden_dim < 1 or latent_dim < 1:
            raise ValueError("hidden_dim and latent_dim must be positive")
        self.hidden_dim, self.latent_dim = hidden_dim, latent_dim
        self.city_embedding = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.tour_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.posterior = nn.Linear(2 * hidden_dim, 2 * latent_dim)
        self.decoder_init = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(hidden_dim + latent_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim + latent_dim, hidden_dim, bias=False)
        self.pointer = nn.Linear(hidden_dim, 1, bias=False)
        # Standardized quality may be negative: no final ReLU.
        self.score_predictor = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                             nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _cities(self, coordinates):
        if coordinates.ndim != 3 or coordinates.shape[1] < 3 or coordinates.shape[2] != 2:
            raise ValueError("coordinates must have shape (batch, n>=3, 2)")
        if not coordinates.is_floating_point() or not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite floating point values")
        return self.city_embedding(coordinates)

    @staticmethod
    def _validate_tours(tours, batch, n):
        if tours.shape != (batch, n) or tours.dtype != torch.long:
            raise ValueError("tours must be int64 of shape (batch, n)")
        expected = torch.arange(n, device=tours.device).expand(batch, n)
        if not torch.equal(tours.sort(dim=1).values, expected):
            raise ValueError("tours must be valid permutations")
        if not torch.all(tours[:, 0] == 0):
            raise ValueError("decoder expects tours rotated to start at city 0")

    def encode(self, coordinates, tours):
        cities = self._cities(coordinates)
        batch, n, hidden = cities.shape
        self._validate_tours(tours, batch, n)
        ordered = cities.gather(1, tours[..., None].expand(batch, n, hidden))
        _, state = self.tour_encoder(ordered)
        mu, logvar = self.posterior(torch.cat([state[0], cities.mean(1)], -1)).chunk(2, -1)
        return mu, logvar.clamp(-10, 10)

    def decode(self, coordinates, z, teacher_tours=None, sample=False):
        """Return (tour, logits). Teacher forcing uses only the visited prefix.

        City 0 is fixed, so logits have shape (batch, n-1, n). No repair is used.
        Greedy decoding is deterministic; sample=True draws categorical actions.
        """
        cities = self._cities(coordinates)
        batch, n, _ = cities.shape
        if z.shape != (batch, self.latent_dim) or not torch.isfinite(z).all():
            raise ValueError("z has invalid shape or nonfinite values")
        if teacher_tours is not None:
            self._validate_tours(teacher_tours, batch, n)
        rows = torch.arange(batch, device=z.device)
        visited = torch.zeros(batch, n, dtype=torch.bool, device=z.device)
        current = torch.zeros(batch, dtype=torch.long, device=z.device)
        visited[:, 0] = True
        state = torch.tanh(self.decoder_init(torch.cat([cities.mean(1), z], -1)))
        keys = self.key(cities)
        choices, all_logits = [current], []
        for step in range(1, n):
            state = self.decoder_cell(torch.cat([cities[rows, current], z], -1), state)
            query = self.query(torch.cat([state, z], -1))
            logits = self.pointer(torch.tanh(keys + query[:, None])).squeeze(-1)
            # clone prevents autograd version errors when visited is updated.
            logits = logits.masked_fill(visited.clone(), float("-inf"))
            all_logits.append(logits)
            if teacher_tours is not None:
                current = teacher_tours[:, step]
            elif sample:
                current = torch.distributions.Categorical(logits=logits).sample()
            else:
                current = logits.argmax(-1)
            choices.append(current)
            visited[rows, current] = True
        return torch.stack(choices, 1), torch.stack(all_logits, 1)

    def forward(self, coordinates, tours, stochastic=True):
        mu, logvar = self.encode(coordinates, tours)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if stochastic else mu
        _, logits = self.decode(coordinates, z, teacher_tours=tours)
        return dict(logits=logits, mu=mu, logvar=logvar, z=z,
                    score=self.score_predictor(z).squeeze(-1))

    @staticmethod
    def loss(output, tours, standardized_quality, beta=0.001, score_weight=1.0):
        n = tours.shape[1]
        nll = F.cross_entropy(output["logits"].reshape(-1, n), tours[:, 1:].reshape(-1))
        kl = -0.5 * (1 + output["logvar"] - output["mu"].square()
                     - output["logvar"].exp()).sum(-1).mean()
        score_mse = F.mse_loss(output["score"], standardized_quality)
        return dict(loss=nll + beta * kl + score_weight * score_mse,
                    nll=nll, kl=kl, score_mse=score_mse)
