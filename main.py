import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from model import Net, train, test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST('../data', train=True, download=True, transform=transform)
server_test_dataset = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(server_test_dataset, batch_size=64)

class Client:
    def __init__(self, client_id: int, train_x: np.ndarray, train_y: np.ndarray, args):
        self.client_id = client_id
        self._ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32).to(device).unsqueeze(1), torch.tensor(train_y).to(device))
        self._train_loader = torch.utils.data.DataLoader(self._ds, batch_size=args.batch_size)
        self._args = args
        self._model = Net().to(device)

    def num_samples(self) -> int: return len(self._ds)
    def model(self) -> Net: return self._model
    
    def train(self) -> None:
        print(f'starting training on client {self.client_id}')
        # optimizer = optim.Adam(self._model.parameters(), lr=self._args.learning_rate)
        optimizer = optim.Adam(self._model.parameters(), lr=self._args.learning_rate)
        for epoch in range(1, self._args.epochs + 1):
            train(self._model, device, self._train_loader, optimizer, epoch)

def start_fl(args) -> None:
    # shuffle the training data and equally split it across all clients -> IID
    p = np.random.permutation(len(train_ds.data))
    train_x, train_y = train_ds.data[p], train_ds.targets[p]
    train_parts_x, train_parts_y = np.array_split(train_x, args.num_clients), np.array_split(train_y, args.num_clients)
    clients = [Client(i, train_parts_x[i], train_parts_y[i], args) for i in range(args.num_clients)]
    
    server_model = Net().to(device)
    
    for round in range(1, 1 + args.num_rounds):
        print(f'starting round {round}')
        # step 1: select clients to participate in the FL round
        selected_clients = select_clients(clients, args.percentage_available_per_round)
        # step 2: distribute the server model to all selected clients
        for c in selected_clients: c.model().load_state_dict(server_model.state_dict())
        # step 3: each client trains it's model locally
        for c in selected_clients: c.train()
        # step 4: the server collects and aggregates all client models and updates the server model
        new_params = fed_avg(selected_clients)
        server_model.load_state_dict(new_params)
        # step 5: test the model on the server dataset
        test(server_model, device, test_loader)

    if args.save_model:
        torch.save(server_model.state_dict(), 'mnist_cnn.pt')

def select_clients(clients: list[Client], percentage_available: float) -> list[Client]:
    k = int(len(clients) * percentage_available)
    print(f'selecting {k} random clients')
    return random.sample(clients, k)

def fed_avg(clients: list[Client]) -> dict:
    total_samples = sum(client.num_samples() for client in clients)
    state_dicts = [client.model().state_dict() for client in clients]
    aggregated_state_dict = {}

    for key in state_dicts[0].keys():
        aggregated_state_dict[key] = sum(client.model().state_dict()[key] * client.num_samples() for client in clients) / total_samples

    return aggregated_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federate Learning simulation for MNIST')
    parser.add_argument('--num-rounds', help='number of rounds (default: 10)', type=int, default=10)
    parser.add_argument('--num-clients', help='number of clients (default: 100)', type=int, default=100)
    parser.add_argument('--percentage-available-per-round', help='percentage of clients that participate in training each round (default: 0.1)', type=float, default=0.1)
    parser.add_argument('--batch-size', help='batch size used for training by each client (default: 16)', type=int, default=16)
    parser.add_argument('--epochs', help='number of epochs to train on each client (default: 10)', type=int, default=10)
    parser.add_argument('--learning-rate', help='learing rate used by each client (default: 0.0001)', type=float, default=0.0001)
    parser.add_argument('--save-model', help='save the trained server model', action='store_true', default=False)
    args = parser.parse_args()
    start_fl(args)
