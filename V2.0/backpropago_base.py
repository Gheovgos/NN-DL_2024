import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.5,), (0.5,))
])

full_data = torchvision.datasets.MNIST(root='../data', train=True, transform=transform, download=True)

train_val_data, test_data = train_test_split(full_data, test_size=0.2, shuffle=True)

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=2)

class ShallowNet(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, n_nodes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_nodes, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

n_nodes = 256
net = ShallowNet(n_nodes=n_nodes).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Rprop(net.parameters(), lr=0.01)

def train_loop(running_loss, train_loader):
    for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            inputs = inputs.view(-1, 28*28)
            outputs = net(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step() 
        
            running_loss += loss.item()
    return running_loss

def test_loop(test_loader):
    correct = 0
    total = 0
    test_loss = 0.0

    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            images = images.view(-1, 28*28)
            outputs = net(images)
            _, prediction = torch.max(outputs, 1)
            total += labels.size(0)

            loss = loss_function(outputs, labels)
            test_loss += loss.item() * images.size(0)

            correct += (prediction == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss /= len(test_loader)
    return test_loss, accuracy

best_accuracy = 0.0
all_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_data)))):
    print(f'Fold {fold+1}/{k_folds}')

    train_subset = torch.utils.data.Subset(full_data, train_idx)
    val_subset = torch.utils.data.Subset(full_data, val_idx)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1000, shuffle=True, num_workers=2)

    for epoch in range(20):
        print(f'Training epoch {epoch+1}...')
        
        running_loss = train_loop(0.0, train_loader)
        val_loss, accuracy = test_loop(val_loader)

        print(f'Loss: {running_loss/len(train_loader):.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')

        if (best_accuracy < accuracy):
            best_accuracy = accuracy

    print(f'Best accuracy for fold {fold+1} was: {accuracy:.2f}%')

    all_accuracies.append(accuracy)

average_accuracy = np.mean(all_accuracies)
std_accuracy = np.std(all_accuracies)

print(f'Cross-Validation Accuracy: {average_accuracy:.2f}% ± {std_accuracy:.2f}%')

test_loss, test_accuracy = test_loop(test_loader)
print(f'Test Accuracy: {test_accuracy}%')


