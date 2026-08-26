from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients

from server.server import Server
from clients.client import Client
from data.dataset import load_mnist, create_clients


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