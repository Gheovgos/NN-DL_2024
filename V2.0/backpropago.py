from datetime import datetime
import time
from torch import nn
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Compose, Normalize
from torchvision.transforms import RandomRotation, RandomAffine

# VARIABLES

batch_size = 128
learning_rate = 1e-4
epochs = 10
node_numbers = [256]
loss_fn = nn.CrossEntropyLoss()

# DEVICE
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


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
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


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
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def train_loop(dataloader, model, loss_fn_train, optimizer_train):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        optimizer_train.zero_grad()

        pred = model(X)
        loss = loss_fn_train(pred, y)

        loss.backward()
        optimizer_train.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


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
    return test_loss

for nodes in node_numbers:

    net = NeuralNetwork(nodes).to(device)
    print(net)

    optimizer = torch.optim.Rprop(net.parameters(), lr=learning_rate)

    early_stopping = EarlyStopping(patience=5, delta=0.01)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, net, loss_fn, optimizer)
        test_loss = test_loop(test_dataloader, net, loss_fn)
        
        early_stopping(test_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Done!")

early_stopping.load_best_model(net)

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_dataloader:
        data = data.to(device)
        target = target.to(device)
        outputs = net(data)

        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to(device)

        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"model_{current_time}.pth"
torch.save(net.state_dict(), filename)


