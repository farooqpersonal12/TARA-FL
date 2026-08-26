import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model import (MNISTModel)


class Client:

    def __init__(self, client_id, dataset):

        self.client_id = client_id
        self.dataset = dataset
        self.model = MNISTModel()

    def set_model(self, parameters):
        self.model.load_state_dict(parameters)

    def train(self, epochs=1):

        train_loader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True
        )

        loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01
        )

        self.model.train()

        for epoch in range(epochs):

            for images, labels in train_loader:

                optimizer.zero_grad()

                outputs = self.model(images)

                loss = loss_function(outputs, labels)

                loss.backward()

                optimizer.step()

        return self.model

    def get_parameters(self):
        return self.model.state_dict()
