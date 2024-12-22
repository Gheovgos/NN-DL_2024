from models import net

class Trainer():

    def __init__(self, training_data, optimizer, criterion):
        self.training_data = training_data
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self):
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(self.training_data, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outpus = net(inputs)
                loss = self.criterion(outpus, labels) # di default usa soft-max | loss = nn.CrossEntropyLoss()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 200 == 199:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                    running_loss = 0.0
        print('Finished training')
