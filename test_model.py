import torch
from torchvision import datasets, transforms
from model import Net
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Please provide the path to the model")
    exit()

model_path = sys.argv[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST('../data', download=True, train=False, transform=transform)
ds_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

test_x, test_y = next(iter(ds_loader))

model = Net().to(device)
model.load_state_dict(torch.load(model_path))
output = model(test_x.to(device))


num_row = 8
num_col = 8

fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i, (x, y, y_output) in enumerate(zip(test_x, test_y, output)):
    pred = y_output.argmax(dim=0, keepdim=True)
    ax = axes[i//num_col, i%num_col]
    ax.imshow(x.numpy().squeeze(), cmap='gray')
    ax.set_title(f'y: {y.cpu().numpy()}, pred: {pred.cpu().numpy()}')

plt.tight_layout()
plt.show()

