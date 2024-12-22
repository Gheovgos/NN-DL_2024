import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_nodes):
        super(Net, self, ).__init__()
        self.hidden_layer = nn.Linear(28 * 28, n_nodes)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(n_nodes, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
    