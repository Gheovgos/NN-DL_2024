import torch

class Tester:
    def __init__(self, net, test_data):
        self.net = net
        self.test_data = test_data

    def evaluate_accuracy(self, device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_data:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy

    def test_loop(self, model, loss_fn, device):
        self.net.eval()
        size = len(self.test_data)
        num_batches = len(self.test_data)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_data:
                X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")