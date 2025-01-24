import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.net import Net
from evaluator.test import Tester
from training.training import Trainer
from data.dataset import Dataset


def main():
    try:

        device = pick_device()

        for n_nodes in range(12000, 60001, 12000):
            print(f'Test with n_nodes = {n_nodes}')

            start_time = time.time()

            model = Net(n_nodes=n_nodes).to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)

            dataset = Dataset(batch_size=256, shuffle_train=True, train_set_size=60000, test_set_size=10000)
            train_loader, test_loader = dataset.prepare_dataset()

            trainer = Trainer(net=model, training_data=train_loader, optimizer=optimizer, loss_fn=loss_fn)
            trainer.train(device=device)
            tester = Tester(net=model, test_data=test_loader)
            tester.evaluate_accuracy(device=device)

            print("--- %s seconds ---\n" % (time.time() - start_time))

    except KeyboardInterrupt:
        print("Interrupted from keyboard")


def pick_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "hip"
        if torch.version.hip is not None
        else "cpu"
    )
    print(f"Using {device} device")

    return device


if __name__ == "__main__":
    main()

