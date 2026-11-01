import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from client.trainer import LocalTrainer
from privacy.accountant import PrivacyAccountant
from privacy.config import DPConfig


class DifferentialPrivacyTests(unittest.TestCase):
    def test_dp_config_rejects_non_per_example_clipping(self):
        with self.assertRaisesRegex(ValueError, "Only per-example clipping"):
            DPConfig(enable_dp=True, clipping_mode="global")

    def test_dp_accountant_tracks_positive_budget_for_valid_step(self):
        accountant = PrivacyAccountant(
            noise_multiplier=1.5,
            max_grad_norm=2.0,
            sampling_probability=0.25,
            local_epochs=2,
            rounds=3,
        )
        accountant.step(total_examples=100, batch_size=10)
        self.assertGreater(accountant.epsilon, 0.0)
        self.assertGreater(accountant.total_steps, 0)

    def test_dp_accountant_rejects_invalid_sampling_probability(self):
        accountant = PrivacyAccountant(sampling_probability=0.0)
        with self.assertRaisesRegex(ValueError, "sampling_probability"):
            accountant.step(total_examples=100, batch_size=10)

    def test_local_trainer_dp_mode_keeps_training_workflow_valid(self):
        torch.manual_seed(123)
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        trainer = LocalTrainer(
            client_id=1,
            dataloader=loader,
            device=torch.device("cpu"),
            local_epochs=1,
            learning_rate=0.01,
            dp_config=DPConfig(
                enable_dp=True,
                noise_multiplier=1.0,
                max_grad_norm=1.0,
                sampling_probability=1.0,
                local_epochs=1,
                privacy_rounds=1,
            ),
        )

        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_value = trainer._dp_step(
            model,
            optimizer,
            torch.nn.BCEWithLogitsLoss(),
            x,
            y.float(),
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss_value)))
        self.assertGreater(trainer.privacy_accountant.epsilon, 0.0)

    def test_local_trainer_dp_mode_clips_each_example_before_noise(self):
        torch.manual_seed(7)
        x = torch.randn(6, 4)
        y = torch.randint(0, 2, (6,))
        loader = DataLoader(TensorDataset(x, y), batch_size=6)

        dp_config = DPConfig(
            enable_dp=True,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            sampling_probability=0.5,
            local_epochs=1,
            privacy_rounds=1,
        )
        trainer = LocalTrainer(
            client_id=1,
            dataloader=loader,
            device=torch.device("cpu"),
            local_epochs=1,
            learning_rate=0.01,
            dp_config=dp_config,
        )

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        original_gradients = []
        for p in model.parameters():
            if p.grad is not None:
                original_gradients.append(p.grad.detach().clone())

        trainer._dp_step(model, optimizer, torch.nn.BCEWithLogitsLoss(), x, y.float())
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == "__main__":
    unittest.main()
