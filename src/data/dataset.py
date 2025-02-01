from torch.utils.data import DataLoader, random_split
import torch
import torchvision
from torchvision.transforms.v2 import Compose, Normalize, RandomRotation, RandomAffine, ToImage, ToDtype


class Dataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def prepare_dataset(self):
        
        transform = Compose([
            RandomRotation(10),  # Random rotation of up to 10 degrees
            RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Random affine transformation
            ToImage(),  # Convert PIL or NumPy to tensor
            ToDtype(torch.float32, scale=True),  # Scale to [0, 1] and convert to float32
            Normalize((0.5,), (0.5,))  # Normalize with mean=0.5, std=0.5
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
        
