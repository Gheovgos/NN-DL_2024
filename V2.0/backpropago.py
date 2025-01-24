from torch import nn
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Lambda, Compose, Normalize
from torchvision.transforms import RandomRotation, RandomAffine


# DATASETS

transform = Compose([
    RandomRotation(10),
    RandomAffine(0, shear=10, scale=(0.8,1.2)),
    ToTensor(),
    Normalize((0.5,), (0.5,))
])


full_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Define the size of the test set (e.g., 10% of the dataset)
test_size = int(0.2 * len(full_data))
train_size = len(full_data) - test_size

# Split the dataset
train_data, test_data = random_split(full_data, [train_size, test_size])

# DATALOADERS

batch_size = 1024

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# DEVICE

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# NEURAL NETWORK


class NeuralNetwork(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn_train, optimizer_train):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn_train(pred, y)

        loss.backward()
        optimizer_train.step()
        optimizer_train.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn_test):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn_test(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


learning_rate = 1e-4
epochs = 10
loss_fn = nn.CrossEntropyLoss()

node_numbers = [128, 256, 512, 768, 1024]

for nodes in node_numbers:

    net = NeuralNetwork(nodes).to(device)
    print(net)

    optimizer = torch.optim.Rprop(net.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, net, loss_fn, optimizer)
        test_loop(test_dataloader, net, loss_fn)

    print("Done!")

