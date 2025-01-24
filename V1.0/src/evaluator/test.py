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
