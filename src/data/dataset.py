import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils

class Dataset():
    def __init__(self, batch_size, shuffle_train, train_set_size, test_set_size):
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.trainset_size = train_set_size
        self.testset_size = test_set_size

    def transform(self, img):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])(img)

    def prepare_dataset(self):
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        train_indices = torch.arange(self.trainset_size)
        trainset = data_utils.Subset(trainset, train_indices)
        
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        test_indices = torch.arange(self.testset_size)
        testset = data_utils.Subset(testset, test_indices)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle_train)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        return trainloader, testloader
        
