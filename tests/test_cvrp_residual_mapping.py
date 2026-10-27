import unittest

import numpy as np
import torch

from src.cvrp_residual_mapping import (
    ResidualBridge, assert_loss_partition, residual_preprocess,
    shuffle_residual_within_bins,
)


class ResidualMappingTests(unittest.TestCase):
    def test_preprocessing_is_train_only_and_zeroes_inactive_dimensions(self):
        latent = np.array([[1., 2.], [3., 4.], [5., 100.], [7., 200.]], dtype=np.float32)
        quality = np.array([4., 3., 2., 1.])
        rank, residual, bins, _, audit = residual_preprocess(latent, quality, [0, 1], [2, 3], 2, 1e-8)
        self.assertEqual(audit['bin_counts_train'], [1, 1])
        self.assertEqual(audit['fallback_bins'], [])
        self.assertEqual(rank.shape, (4, 3)); self.assertEqual(residual.shape, latent.shape)
        self.assertEqual(audit['active_dimension_count'], 0)
        np.testing.assert_array_equal(residual, np.zeros_like(residual))

    def test_zero_residual_branch_has_no_residual_contribution(self):
        bridge = ResidualBridge(4, 3)
        rank = torch.randn(5, 3)
        zero = torch.zeros(5, 4)
        np.testing.assert_allclose(bridge(rank, zero).detach().numpy(), bridge.rank(rank).detach().numpy())

    def test_shuffle_stays_in_partition_and_bin(self):
        residual = np.arange(24, dtype=np.float32).reshape(6, 4)
        bins = np.array([0, 0, 1, 1, 2, 2])
        shuffled, permutation, audit = shuffle_residual_within_bins(residual, bins, [0, 1, 2, 3], [4, 5], 9)
        np.testing.assert_array_equal(np.sort(permutation[[0, 1, 2, 3]]), [0, 1, 2, 3])
        np.testing.assert_array_equal(np.sort(permutation[[4, 5]]), [4, 5])
        np.testing.assert_array_equal(bins, bins[permutation])
        self.assertTrue(audit['rank_bins_unchanged'])
        self.assertEqual(shuffled.shape, residual.shape)

    def test_loss_partition_audit_rejects_overlap(self):
        audit = assert_loss_partition(np.array([10, 11, 12, 13]), [0, 1], [2, 3])
        self.assertTrue(audit['disjoint'])
        with self.assertRaises(AssertionError):
            assert_loss_partition(np.array([10, 11, 12, 13]), [0, 1], [1, 2])


if __name__ == '__main__':
    unittest.main()
