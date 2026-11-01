import secrets
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from client.client import FederatedClient
from config import FLConfig
from model.mlp import MLP
from privacy.config import DPConfig
from server.aggregator import fedavg
from server.secure_aggregation import (
    MaskedClientUpdate,
    SecureAggregationConfig,
    SecureAggregationParticipant,
    SecureAggregationProtocol,
)
from server.server import FederatedServer


def create_round(client_ids, config):
    protocol = SecureAggregationProtocol(config, client_ids=client_ids)
    participants = {
        client_id: SecureAggregationParticipant(client_id, config)
        for client_id in client_ids
    }
    round_id = secrets.token_hex(16)
    public_keys = {
        client_id: participant.begin_round(round_id, client_ids)
        for client_id, participant in participants.items()
    }
    for participant in participants.values():
        participant.accept_public_keys(round_id, public_keys)
    for participant in participants.values():
        participant.distribute_recovery_shares(
            participants, int(config.recovery_threshold)
        )
    protocol.begin_round(round_id, public_keys)
    return protocol, participants, round_id


class SecureAggregationTests(unittest.TestCase):
    def setUp(self):
        self.config = SecureAggregationConfig(
            enabled=True,
            quantization_scale=1_000.0,
            modulus=2**61 - 1,
            min_clients=2,
            max_dropout_fraction=0.5,
            recovery_threshold=2,
            max_abs_update=1_000.0,
        )
        self.client_ids = (1, 2, 3)
        self.protocol, self.participants, self.round_id = create_round(
            self.client_ids, self.config
        )

    def mask(self, client_id, state, sample_count=1):
        return self.participants[client_id].mask_client_update(state, sample_count)

    def test_pairwise_mask_generation_uses_opposite_masks(self):
        first = self.participants[1].pairwise_mask(2, "weight", torch.Size([4]))
        second = self.participants[2].pairwise_mask(1, "weight", torch.Size([4]))
        self.assertTrue(torch.any(first != 0))
        torch.testing.assert_close(
            torch.remainder(first + second, self.config.modulus),
            torch.zeros_like(first),
        )

    def test_round_keys_are_fresh_and_round_ids_cannot_be_replayed(self):
        participant = SecureAggregationParticipant(1, self.config)
        first_public_key = participant.begin_round("fresh-round-1", (1, 2))
        second_public_key = participant.begin_round("fresh-round-2", (1, 2))
        self.assertNotEqual(first_public_key, second_public_key)
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            participant.begin_round("fresh-round-2", (1, 2))

    def test_masked_sum_cancels_pairwise_masks(self):
        states = {
            1: {"weight": torch.tensor([1.0, 2.0])},
            2: {"weight": torch.tensor([3.0, 4.0])},
            3: {"weight": torch.tensor([5.0, 6.0])},
        }
        counts = {1: 1, 2: 2, 3: 3}
        masked = [
            self.mask(client_id, state, counts[client_id])
            for client_id, state in states.items()
        ]
        result = self.protocol.aggregate(masked, sample_counts=counts)
        expected = fedavg([(states[cid], counts[cid]) for cid in self.client_ids])
        torch.testing.assert_close(result["weight"], expected["weight"])

    def test_secure_aggregation_matches_ordinary_fedavg(self):
        states = {
            1: {"weight": torch.tensor([2.0, 4.0])},
            2: {"weight": torch.tensor([6.0, 8.0])},
        }
        counts = {1: 2, 2: 4}
        protocol, participants, round_id = create_round((1, 2), self.config)
        masked = [
            participants[cid].mask_client_update(states[cid], counts[cid])
            for cid in (1, 2)
        ]
        secure_result = protocol.aggregate(masked, sample_counts=counts)
        fedavg_result = fedavg([(states[cid], counts[cid]) for cid in (1, 2)])
        torch.testing.assert_close(
            secure_result["weight"], fedavg_result["weight"], atol=0.0005, rtol=0
        )
        self.assertTrue(round_id)

    def test_quantization_error_is_bounded_by_half_a_step(self):
        values = torch.tensor([0.1234, -0.4567, 2.5002])
        restored = self.protocol.dequantize_tensor(
            self.protocol.quantize_tensor(values)
        )
        self.assertLessEqual((restored - values.to(torch.float64)).abs().max().item(), 0.0005)

    def test_update_outside_configured_bound_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_abs_update"):
            self.mask(1, {"weight": torch.tensor([1_001.0])})

    def test_modular_sum_overflow_bound_is_rejected(self):
        config = SecureAggregationConfig(
            enabled=True,
            quantization_scale=1.0,
            modulus=101,
            min_clients=2,
            max_dropout_fraction=0.0,
            max_abs_update=20.0,
        )
        protocol, participants, _ = create_round((1, 2), config)
        masked = [
            participants[cid].mask_client_update({"weight": torch.tensor([1.0])}, 2)
            for cid in (1, 2)
        ]
        with self.assertRaisesRegex(OverflowError, "wrap"):
            protocol.aggregate(masked)

    def test_dropout_private_key_recovery_removes_unmatched_masks(self):
        ids = (1, 2, 3, 4)
        config = SecureAggregationConfig(
            enabled=True,
            min_clients=2,
            max_dropout_fraction=0.5,
            recovery_threshold=2,
            max_abs_update=1_000.0,
        )
        protocol, participants, round_id = create_round(ids, config)
        states = {
            1: {"weight": torch.tensor([1.0, 10.0])},
            2: {"weight": torch.tensor([99.0, 99.0])},
            3: {"weight": torch.tensor([3.0, 30.0])},
            4: {"weight": torch.tensor([5.0, 50.0])},
        }
        submitted_ids = (1, 3, 4)
        masked = [participants[cid].mask_client_update(states[cid], 1) for cid in submitted_ids]
        shares = {
            2: [
                participants[cid].recovery_share_for(2, round_id)
                for cid in submitted_ids
            ]
        }
        result = protocol.aggregate(masked, recovery_shares=shares)
        expected = fedavg([(states[cid], 1) for cid in submitted_ids])
        torch.testing.assert_close(result["weight"], expected["weight"])

    def test_dropout_aborts_when_recovery_is_disabled(self):
        config = SecureAggregationConfig(
            enabled=True,
            min_clients=2,
            max_dropout_fraction=0.5,
            allow_recovery=False,
        )
        protocol, participants, _ = create_round((1, 2, 3), config)
        masked = [
            participants[cid].mask_client_update({"weight": torch.tensor([1.0])}, 1)
            for cid in (1, 2)
        ]
        with self.assertRaisesRegex(ValueError, "recovery is disabled"):
            protocol.aggregate(masked)

    def test_dropout_budget_is_enforced(self):
        config = SecureAggregationConfig(
            enabled=True,
            min_clients=2,
            max_dropout_fraction=0.2,
        )
        protocol, participants, round_id = create_round((1, 2, 3), config)
        masked = [
            participants[cid].mask_client_update({"weight": torch.tensor([1.0])}, 1)
            for cid in (1, 2)
        ]
        shares = {
            3: [
                participants[cid].recovery_share_for(3, round_id)
                for cid in (1, 2)
            ]
        }
        with self.assertRaisesRegex(ValueError, "dropout budget"):
            protocol.aggregate(masked, recovery_shares=shares)

    def test_duplicate_and_malformed_submissions_are_rejected(self):
        state = {"weight": torch.tensor([1.0])}
        masked = self.mask(1, state)
        with self.assertRaisesRegex(ValueError, "Duplicate client submission"):
            self.protocol.aggregate([masked, masked])

        malformed = self.mask(2, state)
        malformed.state_dict["weight"] = torch.tensor([float("nan")])
        with self.assertRaisesRegex(ValueError, "incompatible masked tensor"):
            self.protocol.aggregate([masked, malformed])

    def test_non_finite_client_update_is_rejected_before_masking(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.mask(1, {"weight": torch.tensor([float("nan")])})

    def test_minimum_uploading_client_count_is_enforced(self):
        masked = self.mask(1, {"weight": torch.tensor([1.0])})
        with self.assertRaisesRegex(ValueError, "at least 2 uploading"):
            self.protocol.aggregate([masked])

    def test_recovery_requires_threshold_and_distinct_survivors(self):
        ids = (1, 2, 3)
        protocol, participants, round_id = create_round(ids, self.config)
        masked = [
            participants[1].mask_client_update({"weight": torch.tensor([1.0])}, 1),
            participants[2].mask_client_update({"weight": torch.tensor([3.0])}, 1),
        ]
        one_share = {
            3: [participants[1].recovery_share_for(3, round_id)]
        }
        with self.assertRaisesRegex(ValueError, "at least 2 valid recovery shares"):
            protocol.aggregate(masked, recovery_shares=one_share)

    def test_sample_count_mismatch_and_wrong_round_are_rejected(self):
        masked = [
            self.mask(cid, {"weight": torch.tensor([float(cid)])}, 2)
            for cid in self.client_ids
        ]
        with self.assertRaisesRegex(ValueError, "invalid sample count"):
            self.protocol.aggregate(masked, sample_counts={1: 3, 2: 2, 3: 2})

        masked[0].round_id = "another-round"
        with self.assertRaisesRegex(ValueError, "wrong round"):
            self.protocol.aggregate(masked)

    def test_explicit_aggregation_modes_enable_the_requested_path(self):
        cases = (
            ("fedavg", False, False),
            ("secure", True, False),
            ("dp", False, True),
            ("secure_dp", True, True),
        )
        for mode, secure, dp in cases:
            with self.subTest(mode=mode):
                config = FLConfig(
                    aggregation_mode=mode,
                    num_clients=3,
                    results_dir="results",
                    data_dir="data_storage",
                )
                self.assertEqual(config.secure_aggregation, secure)
                self.assertEqual(config.enable_dp, dp)

    def test_server_receives_only_masked_uploads_in_secure_mode(self):
        class RecordingSecureClient(FederatedClient):
            def train_and_mask_secure(self, model_class, model_kwargs):
                result = super().train_and_mask_secure(model_class, model_kwargs)
                self.server_visible_result = result
                return result

        torch.manual_seed(11)
        clients = []
        for client_id in (1, 2):
            inputs = torch.randn(8, 1, 28, 28)
            labels = torch.randint(0, 10, (8,))
            loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
            clients.append(
                RecordingSecureClient(
                    client_id=client_id,
                    dataloader=loader,
                    device=torch.device("cpu"),
                    local_epochs=1,
                    learning_rate=0.001,
                    optimizer_name="Adam",
                )
            )

        test_inputs = torch.randn(6, 1, 28, 28)
        test_labels = torch.randint(0, 10, (6,))
        test_loader = DataLoader(TensorDataset(test_inputs, test_labels), batch_size=3)
        server = FederatedServer(
            global_model=MLP(dropout_p=0.0),
            clients=clients,
            test_loader=test_loader,
            device=torch.device("cpu"),
            num_rounds=1,
            secure_config=SecureAggregationConfig(
                enabled=True,
                min_clients=2,
                max_dropout_fraction=0.0,
            ),
        )

        with patch.object(
            server.aggregator,
            "aggregate",
            side_effect=AssertionError("Plain FedAvg must not run in secure mode"),
        ):
            metrics = server.run(MLP, {"dropout_p": 0.0})

        for client in clients:
            upload, loss, accuracy = client.server_visible_result
            self.assertIsInstance(upload, MaskedClientUpdate)
            self.assertTrue(all(tensor.dtype == torch.int64 for tensor in upload.state_dict.values()))
            self.assertTrue(torch.isfinite(torch.tensor([loss, accuracy])).all())
        self.assertEqual(len(metrics["rounds"]), 2)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["global_test_loss"][0])))

    def test_secure_plus_dp_runs_client_noise_before_masking(self):
        class TinyBinaryModel(torch.nn.Module):
            def __init__(self, input_dim=4, num_classes=1):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim, num_classes)

            def forward(self, inputs):
                return self.linear(inputs)

        dp_config = DPConfig(
            enable_dp=True,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            sampling_probability=1.0,
            local_epochs=1,
            privacy_rounds=1,
        )
        clients = []
        for client_id in (1, 2):
            inputs = torch.randn(4, 4)
            labels = torch.randint(0, 2, (4,))
            loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
            clients.append(
                FederatedClient(
                    client_id=client_id,
                    dataloader=loader,
                    device=torch.device("cpu"),
                    local_epochs=1,
                    learning_rate=0.01,
                    dp_config=dp_config,
                )
            )
        test_loader = DataLoader(
            TensorDataset(torch.randn(4, 4), torch.randint(0, 2, (4,))),
            batch_size=4,
        )
        server = FederatedServer(
            global_model=TinyBinaryModel(),
            clients=clients,
            test_loader=test_loader,
            device=torch.device("cpu"),
            num_rounds=1,
            num_classes=1,
            dp_config=dp_config,
            secure_config=SecureAggregationConfig(
                enabled=True,
                min_clients=2,
                max_dropout_fraction=0.0,
            ),
        )

        metrics = server.run(
            TinyBinaryModel, {"input_dim": 4, "num_classes": 1}
        )

        self.assertEqual(len(metrics["rounds"]), 2)
        self.assertTrue(
            all(client._trainer.privacy_accountant.total_steps > 0 for client in clients)
        )


if __name__ == "__main__":
    unittest.main()
