import torch
import torchvision
import torchvision.transforms as transforms

class Dataset():
    def __init__(self, batch_size, shuffle_train):
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

    def transform():
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 

    def prepare_dataset(self):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle_train)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        return trainloader, testloader
        
