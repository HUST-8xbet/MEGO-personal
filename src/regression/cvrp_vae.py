"""CVRP permutation/route VAE with integer capacity masks and padded loss."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def canonicalize_solutions(problem, solutions):
    """Canonical route directions and vehicle order; not a neural invariance."""
    result = []
    for solution in solutions:
        routes = []
        for route in problem.split_routes(solution):
            customers = tuple(int(i) for i in route[1:-1])
            routes.append(min(customers, customers[::-1]))
        sequence = [0]
        for route in sorted(routes):
            sequence.extend(route)
            sequence.append(0)
        result.append(np.array(sequence, dtype=np.int64))
    return result


class CVRPVAE(nn.Module):
    def __init__(self, hidden_dim=64, latent_dim=16):
        super().__init__()
        if hidden_dim < 1 or latent_dim < 1:
            raise ValueError('hidden_dim and latent_dim must be positive')
        self.hidden_dim, self.latent_dim = hidden_dim, latent_dim
        # x, y, demand/capacity, depot indicator.
        self.city_embedding = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.route_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.posterior = nn.Linear(2 * hidden_dim, 2 * latent_dim)
        self.decoder_init = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        self.decoder_cell = nn.GRUCell(hidden_dim + latent_dim + 1, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim + latent_dim + 1, hidden_dim, bias=False)
        self.pointer = nn.Linear(hidden_dim, 1, bias=False)
        self.score_predictor = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                             nn.Linear(hidden_dim, 1))

    def _cities(self, coordinates, demands, capacity):
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2 or coordinates.shape[1] < 2:
            raise ValueError('coordinates must have shape (batch, customers+1, 2)')
        batch, vertices, _ = coordinates.shape
        if not coordinates.is_floating_point() or not torch.isfinite(coordinates).all():
            raise ValueError('coordinates must be finite floating point values')
        if demands.shape != (batch, vertices) or demands.dtype != torch.long:
            raise ValueError('demands must be int64 with shape (batch, vertices)')
        if capacity.shape != (batch,) or capacity.dtype != torch.long or not torch.all(capacity > 0):
            raise ValueError('capacity must be positive int64 with shape (batch,)')
        if not (torch.all(demands[:, 0] == 0) and torch.all(demands[:, 1:] > 0)
                and torch.all(demands[:, 1:] <= capacity[:, None])):
            raise ValueError('infeasible demand vector')
        depot = torch.zeros_like(coordinates[..., :1])
        depot[:, 0] = 1.
        normalized = demands.to(coordinates.dtype) / capacity[:, None].to(coordinates.dtype)
        return self.city_embedding(torch.cat([coordinates, normalized[..., None], depot], -1))

    @staticmethod
    def _check_sequences(sequences, lengths, batch, vertices):
        width = 2 * (vertices - 1) + 1
        if sequences.shape != (batch, width) or sequences.dtype != torch.long:
            raise ValueError('sequences must be int64 padded to width 2*customers+1')
        if lengths.shape != (batch,) or lengths.dtype != torch.long:
            raise ValueError('lengths must be int64 of shape (batch,)')
        if not torch.all((lengths >= vertices + 1) & (lengths <= width)):
            raise ValueError('invalid sequence lengths')
        active = torch.arange(width, device=sequences.device)[None] < lengths[:, None]
        if not torch.all(torch.where(active, (sequences >= 0) & (sequences < vertices), sequences == -1)):
            raise ValueError('invalid token or padding')
        if not (torch.all(sequences[:, 0] == 0)
                and torch.all(sequences.gather(1, lengths[:, None] - 1) == 0)):
            raise ValueError('sequences must start and end at depot')
        if torch.any((sequences[:, :-1] == 0) & (sequences[:, 1:] == 0) & active[:, 1:]):
            raise ValueError('empty vehicle routes are forbidden')
        customers = torch.arange(1, vertices, device=sequences.device)
        if not torch.all((sequences[..., None] == customers).sum(1) == 1):
            raise ValueError('each customer must be visited exactly once')

    def encode(self, coordinates, demands, capacity, sequences, lengths):
        cities = self._cities(coordinates, demands, capacity)
        batch, vertices, hidden = cities.shape
        self._check_sequences(sequences, lengths, batch, vertices)
        ordered = cities.gather(1, sequences.clamp_min(0)[..., None].expand(-1, -1, hidden))
        packed = nn.utils.rnn.pack_padded_sequence(ordered, lengths.cpu(), batch_first=True,
                                                  enforce_sorted=False)
        _, state = self.route_encoder(packed)
        mu, logvar = self.posterior(torch.cat([state[0], cities.mean(1)], -1)).chunk(2, -1)
        return mu, logvar.clamp(-10, 10)

    def decode(self, coordinates, demands, capacity, z, teacher_sequences=None,
               teacher_lengths=None, sample=False):
        """Return padded routes, true sizes and prefix-masked logits.

        Depot is forbidden at the depot, allowed after a customer. Completed
        rows output -1; a safe dummy depot logit prevents NaNs in ignored slots.
        Remaining load uses INTEGER arithmetic, not float comparisons.
        """
        cities = self._cities(coordinates, demands, capacity)
        batch, vertices, _ = cities.shape
        if z.shape != (batch, self.latent_dim) or not torch.isfinite(z).all():
            raise ValueError('invalid latent')
        if (teacher_sequences is None) != (teacher_lengths is None):
            raise ValueError('provide both teacher sequences and lengths, or neither')
        if teacher_sequences is not None:
            self._check_sequences(teacher_sequences, teacher_lengths, batch, vertices)
        rows = torch.arange(batch, device=coordinates.device)
        current = torch.zeros(batch, dtype=torch.long, device=coordinates.device)
        visited = torch.zeros(batch, vertices, dtype=torch.bool, device=coordinates.device)
        remaining = capacity.clone()
        finished = torch.zeros(batch, dtype=torch.bool, device=coordinates.device)
        sizes = torch.ones(batch, dtype=torch.long, device=coordinates.device)
        state = torch.tanh(self.decoder_init(torch.cat([cities.mean(1), z], -1)))
        keys = self.key(cities)
        choices, all_logits = [current], []
        for step in range(1, 2 * (vertices - 1) + 1):
            active = ~finished
            load_feature = (remaining.to(cities.dtype) / capacity.to(cities.dtype))[:, None]
            state = self.decoder_cell(torch.cat([cities[rows, current], z, load_feature], -1), state)
            query = self.query(torch.cat([state, z, load_feature], -1))
            logits = self.pointer(torch.tanh(keys + query[:, None])).squeeze(-1)
            forbidden = visited | (demands > remaining[:, None])
            forbidden[:, 0] = current == 0  # no empty route / depot-depot transition
            forbidden[finished] = True
            forbidden[finished, 0] = False  # safe dummy action, not emitted
            logits = logits.masked_fill(forbidden, float('-inf'))
            all_logits.append(logits)
            if teacher_sequences is not None:
                if not torch.equal(active, step < teacher_lengths):
                    raise ValueError('teacher termination does not match complete service and depot return')
                chosen = torch.where(active, teacher_sequences[:, step], torch.zeros_like(current))
                if torch.any(forbidden[rows, chosen] & active):
                    raise ValueError('infeasible teacher action: capacity, repeat or empty route')
            elif sample:
                chosen = torch.distributions.Categorical(logits=logits).sample()
            else:
                chosen = logits.argmax(-1)
            choices.append(torch.where(active, chosen, torch.full_like(chosen, -1)))
            sizes = sizes + active.long()
            served = active & (chosen > 0)
            visited[rows[served], chosen[served]] = True
            selected_demand = demands[rows, chosen]
            remaining = torch.where(chosen == 0, capacity, remaining - selected_demand)
            finished = finished | (visited[:, 1:].all(1) & (chosen == 0))
            current = chosen
        if not torch.all(finished):
            raise RuntimeError('decoder failed to terminate within 2*customers actions')
        return torch.stack(choices, 1), sizes, torch.stack(all_logits, 1)

    def forward(self, coordinates, demands, capacity, sequences, lengths, stochastic=True):
        mu, logvar = self.encode(coordinates, demands, capacity, sequences, lengths)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if stochastic else mu
        _, _, logits = self.decode(coordinates, demands, capacity, z, sequences, lengths)
        return dict(logits=logits, mu=mu, logvar=logvar, z=z,
                    score=self.score_predictor(z).squeeze(-1))

    @staticmethod
    def reconstruction_nll(logits, sequences):
        targets = sequences[:, 1:]
        # Each real action (including final depot return) counts once.
        return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-1, reduction='sum') / (targets != -1).sum()

    @staticmethod
    def loss(output, sequences, standardized_quality, beta=0.001, score_weight=1.):
        nll = CVRPVAE.reconstruction_nll(output['logits'], sequences)
        kl = -0.5 * (1 + output['logvar'] - output['mu'].square() - output['logvar'].exp()).sum(-1).mean()
        score_mse = F.mse_loss(output['score'], standardized_quality)
        return dict(loss=nll + beta * kl + score_weight * score_mse,
                    nll=nll, kl=kl, score_mse=score_mse)
