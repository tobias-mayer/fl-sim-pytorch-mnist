# A Minimalistic Federated Learning Simulation with PyTorch and MNIST

This project is a small federated learning simulation implemented in PyTorch, utilizing the popular MNIST dataset. It aims to demonstrate the power of federated learning by training a deep learning model on distributed devices while preserving data privacy. The simulation showcases the process of aggregating model updates from multiple clients in a decentralized manner, fostering collaborative learning without sharing raw data.


## Installation
1. Clone this repository:
```sh
git clone https://github.com/tobias-mayer/fl-sim-pytorch-mnist.git
cd ...
```
2. Setup environment:
```sh
conda env create --file environment.yaml
conda activate torch-fl
```
3. Run:
```sh
python3 main.py
```

The simulation will execute several rounds of federated learning, where each round consists of multiple local training iterations performed by multiple clients (simulated edge devices). After each round, the global model's accuracy and loss will be evaluated on the test dataset.

## Customization
If you want to customize the simulation, you can modify the following parameters in the `main.py` script:

```sh
usage: main.py [-h] [--num-rounds NUM_ROUNDS] [--num-clients NUM_CLIENTS]
               [--percentage-available-per-round PERCENTAGE_AVAILABLE_PER_ROUND] [--batch-size BATCH_SIZE]
               [--epochs EPOCHS] [--learning-rate LEARNING_RATE] [--save-model]

Federate Learning simulation for MNIST

optional arguments:
  -h, --help            show this help message and exit
  --num-rounds NUM_ROUNDS
                        number of rounds (default: 10)
  --num-clients NUM_CLIENTS
                        number of clients (default: 100)
  --percentage-available-per-round PERCENTAGE_AVAILABLE_PER_ROUND
                        percentage of clients that participate in training each round (default: 0.1)
  --batch-size BATCH_SIZE
                        batch size used for training by each client (default: 16)
  --epochs EPOCHS       number of epochs to train on each client (default: 10)
  --learning-rate LEARNING_RATE
                        learing rate used by each client (default: 0.0001)
  --save-model          save the trained server model
```


## References
- Communication-Efficient Learning of Deep Networks from Decentralized Data: https://arxiv.org/abs/1602.05629
- PyTorch: https://pytorch.org/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
