import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils


class Dataset():
    def __init__(self, batch_size, shuffle_train, train_set_size, test_set_size):
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size

    def transform(self, img):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])(img)

    def prepare_dataset(self):
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        train_indices = torch.arange(self.train_set_size)
        train_set = data_utils.Subset(train_set, train_indices)

        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        test_indices = torch.arange(self.test_set_size)
        test_set = data_utils.Subset(test_set, test_indices)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=self.shuffle_train)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader
