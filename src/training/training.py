from models import net

class Trainer():
    def train(self, training_data, optimizer, criterion):
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(training_data, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outpus = net(inputs)
                loss = criterion(outpus, labels) # di default usa soft-max
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 200 == 199:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                    running_loss = 0.0
        print('Finished training')
