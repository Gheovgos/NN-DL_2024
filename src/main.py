import torch.nn as nn
import torch.optim as optim

from data import dataset
from models.net import Net
from evaluator.test import Tester
from training.training import Trainer

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

dataset = dataset.Dataset(batch_size=64, shuffle_train=True)
train, test = dataset.prepare_dataset()

trainer = Trainer(training_data=train, optimizer=optimizer, criterion=criterion)
trainer.train()

tester = Tester(net=net, test_data=test)
tester.evaluate_accuracy()