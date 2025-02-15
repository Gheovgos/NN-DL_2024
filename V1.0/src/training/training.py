
class Trainer():

    def __init__(self, net, training_data, optimizer, loss_fn):
        self.net = net
        self.training_data = training_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, device):
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(self.training_data, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_fn(outputs, labels) # di default usa soft-max | loss = nn.CrossEntropyLoss()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 200 == 199:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                    running_loss = 0.0
        print('Finished training')
