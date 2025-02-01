from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.transforms.v2 import Compose, Normalize, RandomRotation, RandomAffine, ToTensor


class Dataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def prepare_dataset(self):

        transform = Compose([
            RandomRotation(10),
            RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])

        full_data =  torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        test_size = int(0.2 * len(full_data))
        train_size = len(full_data) - test_size

        train_data, test_data = random_split(full_data, [train_size, test_size])

        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        return train_dataloader, test_dataloader
        
