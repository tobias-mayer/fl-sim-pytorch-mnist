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
test_ds = datasets.MNIST('../data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
test_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)


def start_central_learning(args):
  model = Net().to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  for epoch in range(1, args.epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    
  test(model, device, test_loader)
      
  if args.save_model:
    torch.save(model.state_dict(), 'mnist_cnn_central.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federate Learning simulation for MNIST')
    parser.add_argument('--epochs', help='number of epochs to train on each client (default: 10)', type=int, default=10)
    parser.add_argument('--learning-rate', help='learing rate used by each client (default: 0.0001)', type=float, default=0.0001)
    parser.add_argument('--save-model', help='save the trained server model', action='store_true', default=False)
    args = parser.parse_args()
    start_central_learning(args)