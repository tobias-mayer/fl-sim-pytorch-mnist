import argparse
import random
from functools import reduce
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from model import Net, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
x = model.get_params_numpy()
model.assign_params(x)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST('../data', train=True, download=True, transform=transform)
server_test_dataset = datasets.MNIST('../data', train=False, transform=transform)

class Client:
    def __init__(self, train_x, train_y, args):
        ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32), train_y)
        self._train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
        self._args = args
        self._model = Net().to(device)

    def num_samples(self) -> int: return len(self.train_x)
    def model(self) -> Net: return self._model
    def train(self) -> None:
        optimizer = optim.Adadelta(self._model.parameters(), lr=self._args.learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        for epoch in range(1, self._args.epochs + 1):
            train(self._model, device, self._train_loader, optimizer, epoch)
            scheduler.step()

def start_fl(args) -> None:
    # shuffle the training data and equally split it across all clients -> IID
    p = np.random.permutation(len(train_ds.data))
    train_x, train_y = train_ds.data[p], train_ds.targets[p]
    train_parts_x, train_parts_y = np.array_split(train_x, args.num_clients), np.array_split(train_y, args.num_clients)
    clients = [Client(train_parts_x[i], train_parts_y[i], args) for i in range(args.num_clients)]
    
    server_model = Net().to(device)
    
    for round in range(1, 1 + args.num_rounds):
        print(f"starting round {round}")

        # step 1: select clients to participate in the FL round
        selected_clients = select_clients(clients, args.percentage_available_per_round)

        # step 2: distribute the server model to all selected clients
        for c in selected_clients: c.model().assign_params(server_model.get_params_numpy())
        
        # step 3: each client trains it's model locally
        for c in selected_clients: c.train()
        
        # step 4: the server collects and aggregates all client models and updates the server model
        new_params = fed_avg(selected_clients)
        server_model.assign_params(new_params)


def select_clients(clients: list[Client], percentage_available: float) -> list[Client]:
    k = len(clients) * percentage_available
    return random.sample(clients, int(k))

def fed_avg(clients: list[Client]) -> np.ndarray:
    total_samples = sum(map(lambda c: c.num_samples(), clients))

    weighted_params = [
        [layer * c.num_samples() for layer in c.model().get_weights()] for c in clients
    ]
    
    average_params = [
        reduce(np.add, layer_updates) / total_samples for layer_updates in zip(*weighted_params)
    ]
    
    return average_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federate Learning simulation for MNIST')
    parser.add_argument('--num-rounds', help='number of rounds (default: 10)', type=int, default=10)
    parser.add_argument('--num-clients', help='number of clients (default: 10)', type=int, default=10)
    parser.add_argument('--percentage-available-per-round', help='percentage of clients that participate in training each round (default: 0.2)', type=float, default=0.2)
    parser.add_argument('--batch-size', help='batch size used for training by each client (default: 64)', type=int, default=64)
    parser.add_argument('--epochs', help='number of epochs to train on each client (default: 8)', type=int, default=8)
    parser.add_argument('--learning-rate', help='learing rate used by each client (default: 1.0)', type=float, default=1.0)
    args = parser.parse_args()
    start_fl(args)
