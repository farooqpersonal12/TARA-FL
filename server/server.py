from model.model import MNISTModel
from clients.client import Client
from data.dataset import load_mnist, create_clients
import torch


class Server:

    def __init__(self):
        self.global_model = MNISTModel()

    def aggregate(self, client_parameters, client_sizes):

        total_samples = sum(client_sizes)

        new_parameters = {}

        for name in client_parameters[0]:

            new_parameters[name] = torch.zeros_like(
                client_parameters[0][name]
            )

            for parameters, size in zip(
                    client_parameters,
                    client_sizes
            ):

                weight = size / total_samples

                new_parameters[name] += (
                        weight * parameters[name]
                )

        return new_parameters

    def evaluate(self, test_dataset):

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )

        self.global_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                outputs = self.global_model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                total += labels.size(0)

                correct += (
                        predictions == labels
                ).sum().item()

        accuracy = correct / total

        return accuracy


if __name__ == "__main__":

    server = Server()

    print("Server created.")


    train_dataset, test_dataset = load_mnist()

    print("MNIST loaded.")

    client_datasets = create_clients(
        train_dataset,
        num_clients=3
    )

    clients = []

    for i in range(3):

        client = Client(
            client_id=i + 1,
            dataset=client_datasets[i]
        )

        clients.append(client)

        print(
            "Client",
            client.client_id,
            "created with",
            len(client.dataset),
            "samples."
        )

    num_rounds = 5


    for round_number in range(
            1,
            num_rounds + 1
    ):

        print()
        print("==============================")
        print(
            "Federated Round",
            round_number
        )
        print("==============================")


        global_parameters = (
            server.global_model.state_dict()
        )

        for client in clients:

            client.set_model(
                global_parameters
            )


        client_parameters = []
        client_sizes = []

        for client in clients:

            print(
                "Training Client",
                client.client_id
            )

            client.train(
                epochs=1
            )

            parameters = client.get_parameters()

            client_parameters.append(
                parameters
            )

            client_sizes.append(
                len(client.dataset)
            )

        print("Performing FedAvg...")

        new_parameters = server.aggregate(
            client_parameters,
            client_sizes
        )


        server.global_model.load_state_dict(
            new_parameters
        )

        accuracy = server.evaluate(
            test_dataset
        )

        print(
            "Round",
            round_number,
            "accuracy:",
            accuracy * 100,
            "%"
        )


    print()
    print("==============================")
    print("Federated training completed.")
    print("==============================")

    print(
        "Number of clients:",
        len(clients)
    )

    print(
        "Client sample sizes:",
        client_sizes
    )


#OUTPUTS
"""
    Baseline: FedAvg
    Clients: 3
    Samples/client: 20,000
    Local epochs: 1
    Federated rounds: 5
    
    Round 1: 82.29%
    Round 2: 91.29%
    Round 3: 93.74%
    Round 4: 95.31%
    Round 5: 96.27%
"""