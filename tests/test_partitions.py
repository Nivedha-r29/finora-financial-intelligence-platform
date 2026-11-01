import unittest

import numpy as np

from data.dataset import partition_iid, partition_non_iid


class TinyDataset:
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = TinyDataset(np.repeat(np.arange(4), 100))

    def test_iid_partition_is_reproducible_and_complete(self):
        first = partition_iid(self.dataset, num_clients=5, seed=19)
        second = partition_iid(self.dataset, num_clients=5, seed=19)

        self.assertEqual(first, second)
        self.assertEqual(sorted(index for shard in first for index in shard), list(range(400)))
        self.assertTrue(all(len(shard) == 80 for shard in first))

    def test_dirichlet_partition_is_reproducible_and_complete(self):
        first = partition_non_iid(self.dataset, num_clients=5, alpha=0.3, seed=19)
        second = partition_non_iid(self.dataset, num_clients=5, alpha=0.3, seed=19)

        self.assertEqual(first, second)
        self.assertEqual(sorted(index for shard in first for index in shard), list(range(400)))
        client_class_counts = [
            np.bincount(self.dataset.targets[shard], minlength=4)
            for shard in first
        ]
        self.assertTrue(any(not np.array_equal(client_class_counts[0], counts) for counts in client_class_counts[1:]))


if __name__ == "__main__":
    unittest.main()