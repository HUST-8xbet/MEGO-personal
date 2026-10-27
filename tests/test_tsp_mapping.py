import tempfile
import unittest

import numpy as np
import torch

from src.regression.tsp_vae import TSPVAE, canonicalize_tours
from src.tsp_mapping import LatentBridge, rank_matched_pairs, set_decoder_trainable
from src.problem.tsp_problem import TSPProblem


class TSPMappingTests(unittest.TestCase):
    def test_quality_pairing_uses_higher_is_better_for_both_problems(self):
        source_latent = np.arange(20, dtype=np.float32).reshape(5, 4)
        source_quality = np.array([3., 10., 7., 1., 5.])
        tours = np.array([np.roll(np.arange(4), i) for i in range(5)])
        target_quality = np.array([-4., -1., -3., -2., -5.])
        pairs = rank_matched_pairs(source_latent, source_quality, tours, target_quality)
        np.testing.assert_array_equal(pairs.source_quality, [10., 7., 5., 3., 1.])
        np.testing.assert_array_equal(pairs.target_quality, [-1., -2., -3., -4., -5.])
        np.testing.assert_array_equal(pairs.source_latent[0], source_latent[1])
        np.testing.assert_array_equal(pairs.target_tours[0], tours[1])

    def test_pairing_rejects_bad_shapes_and_nonfinite_quality(self):
        with self.assertRaises(ValueError):
            rank_matched_pairs(np.zeros((2, 3)), [1.], np.zeros((2, 4), dtype=int), [1., 2.])
        with self.assertRaises(ValueError):
            rank_matched_pairs(np.zeros((2, 3)), [1., np.nan], np.zeros((2, 4), dtype=int), [1., 2.])

    def test_bridge_gradient_and_checkpoint_round_trip(self):
        bridge = LatentBridge(6, 4)
        source = torch.randn(8, 6)
        loss = bridge(source).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(bridge.projection.weight.grad).all())
        self.assertGreater(bridge.projection.weight.grad.abs().sum().item(), 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/bridge.pt"
            torch.save(bridge.state_dict(), path)
            other = LatentBridge(6, 4)
            other.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(bridge(source), other(source))

    def test_decoder_without_source_latent_still_generates_valid_tours(self):
        problem = TSPProblem(seed=2)
        coords = torch.tensor(problem.coordinates, dtype=torch.float32)[None].expand(4, -1, -1)
        model = TSPVAE(hidden_dim=16, latent_dim=4)
        trainable = set_decoder_trainable(model)
        self.assertTrue(trainable)
        tours, _ = model.decode(coords, torch.zeros(4, 4))
        for tour in tours.numpy():
            self.assertTrue(np.isfinite(problem.evaluate(tour)))
            np.testing.assert_array_equal(canonicalize_tours(tour[None])[0, 0], 0)


if __name__ == "__main__":
    unittest.main()
