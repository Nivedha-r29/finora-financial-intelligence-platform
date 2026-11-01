import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from client.client import FederatedClient
from model.mlp import MLP
from server.aggregator import fedavg
from server.server import FederatedServer


class CountingTensorDataset(TensorDataset):
    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.read_count = 0

    def __getitem__(self, index):
        self.read_count += 1
        return super().__getitem__(index)


class RecordingClient(FederatedClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received_history = []
        self.returned_history = []

    def receive_global_model(self, global_parameters):
        snapshot = {
            key: value.clone() for key, value in global_parameters.items()
        }
        self.received_history.append(snapshot)
        super().receive_global_model(global_parameters)

    def train_locally(self, model_class, model_kwargs):
        result = super().train_locally(model_class, model_kwargs)
        self.returned_history.append(result[0])
        return result


class ServerSmokeTests(unittest.TestCase):
    def test_two_client_single_round_federated_workflow(self):
        torch.manual_seed(7)
        device = torch.device("cpu")
        model = MLP(dropout_p=0.0)
        initial_parameters = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        clients = []
        for client_id in (1, 2):
            inputs = torch.randn(8, 1, 28, 28)
            labels = torch.randint(0, 10, (8,))
            loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
            clients.append(RecordingClient(
                client_id=client_id,
                dataloader=loader,
                device=device,
                local_epochs=1,
                learning_rate=0.001,
                optimizer_name="Adam",
            ))

        test_inputs = torch.randn(6, 1, 28, 28)
        test_labels = torch.randint(0, 10, (6,))
        test_dataset = CountingTensorDataset(test_inputs, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=3)
        server = FederatedServer(
            global_model=model,
            clients=clients,
            test_loader=test_loader,
            device=device,
            num_rounds=2,
        )

        metrics = server.run(MLP, {"dropout_p": 0.0})

        for client in clients:
            for key in initial_parameters:
                torch.testing.assert_close(client.received_history[0][key], initial_parameters[key])
                self.assertFalse(torch.equal(client.returned_history[0][key], initial_parameters[key]))
        self.assertEqual(metrics["rounds"], [0, 1, 2])
        self.assertEqual(len(metrics["global_test_accuracy"]), 3)
        self.assertEqual(len(metrics["client_losses"][1]), 2)
        self.assertEqual(len(metrics["round_duration_seconds"]), 2)
        self.assertEqual(test_dataset.read_count, 3 * len(test_dataset))
        first_round_state = fedavg([
            (client.returned_history[0], 8) for client in clients
        ])
        for client in clients:
            for key in initial_parameters:
                torch.testing.assert_close(client.received_history[1][key], first_round_state[key])
        expected_state = fedavg([(client.returned_history[1], 8) for client in clients])
        for key, value in server.global_model.state_dict().items():
            torch.testing.assert_close(value.cpu(), expected_state[key])
        self.assertTrue(any(not torch.equal(expected_state[key], initial_parameters[key]) for key in initial_parameters))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["global_test_loss"])).all())


if __name__ == "__main__":
    unittest.main()