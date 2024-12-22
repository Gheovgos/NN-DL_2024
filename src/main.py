import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.net import Net
from evaluator.test import Tester
from training.training import Trainer
from data.dataset import Dataset

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print(f'Using device: {device}\n')

    for n_nodes in range(12000, 60001, 12000):
        print(f'Test with n_nodes = {n_nodes}')

        start_time = time.time()

        net = Net(n_nodes=n_nodes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        dataset = Dataset(batch_size=64, shuffle_train=True, train_set_size=30000, test_set_size=7500)
        train_set, test_set = dataset.prepare_dataset()

        trainer = Trainer(net=net, training_data=train_set, optimizer=optimizer, criterion=criterion)
        trainer.train()

        tester = Tester(net=net, test_data=test_set)
        tester.evaluate_accuracy()

        print("--- %s seconds ---\n" % (time.time() - start_time))

except KeyboardInterrupt:
    print("Interrupted from keyboard")