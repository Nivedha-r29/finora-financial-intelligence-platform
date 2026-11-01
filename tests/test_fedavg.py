import unittest

import torch

from server.aggregator import fedavg


class FedAvgTests(unittest.TestCase):
    def test_equal_weight_average(self):
        result = fedavg([
            ({"weight": torch.tensor([2.0])}, 4),
            ({"weight": torch.tensor([8.0])}, 4),
        ])

        torch.testing.assert_close(result["weight"], torch.tensor([5.0]))

    def test_sample_weighted_average(self):
        first = {"weight": torch.tensor([1.0, 3.0])}
        second = {"weight": torch.tensor([5.0, 7.0])}

        result = fedavg([(first, 1), (second, 3)])

        torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))
        torch.testing.assert_close(first["weight"], torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(second["weight"], torch.tensor([5.0, 7.0]))

    def test_integer_buffers_are_not_averaged(self):
        result = fedavg([
            ({"counter": torch.tensor(2, dtype=torch.long)}, 1),
            ({"counter": torch.tensor(5, dtype=torch.long)}, 3),
        ])

        self.assertEqual(result["counter"].item(), 5)

    def test_rejects_zero_sample_count(self):
        with self.assertRaisesRegex(ValueError, "sample count"):
            fedavg([({"weight": torch.tensor([1.0])}, 0)])

    def test_rejects_empty_updates(self):
        with self.assertRaisesRegex(ValueError, "No client updates"):
            fedavg([])

    def test_rejects_invalid_sample_counts(self):
        for count in (-1, 1.5, True):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "sample count"):
                    fedavg([({"weight": torch.tensor([1.0])}, count)])

    def test_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            fedavg([
                ({"weight": torch.zeros(2)}, 1),
                ({"weight": torch.zeros(3)}, 1),
            ])

    def test_rejects_missing_or_unexpected_keys(self):
        for other_state in ({"weight": torch.ones(1)}, {"bias": torch.ones(1)}):
            with self.subTest(keys=set(other_state)):
                with self.assertRaisesRegex(ValueError, "keys incompatible"):
                    fedavg([
                        ({"weight": torch.zeros(1), "bias": torch.zeros(1)}, 1),
                        (other_state, 1),
                    ])


if __name__ == "__main__":
    unittest.main()