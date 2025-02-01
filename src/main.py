import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.net import Net
from evaluator.test import Tester
from training.training import Trainer
from data.dataset import Dataset

try:
    # CHECK GPU NVIDIA CUDA
    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
    # CHECK GPU AMD ROCM                  
    elif torch.version.hip is not None:         # Non c'è una funzione per vedere se rocm è disponibile, quindi bisogna fare il check della versione
        device = torch.device('hip')
    else:
        device = torch.device('cpu')
    
    print(f'Using device: {device}\n')


    nodes = [128, 256, 512, 1024, 2048]

    for n_nodes in nodes:
        print(f'Test with n_nodes = {n_nodes}')

        start_time = time.time()

        net = Net(n_nodes=n_nodes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Rprop(net.parameters(), lr=0.01)

        dataset = Dataset(batch_size=64)
        train_dataloader, test_dataloader = dataset.prepare_dataset()

        trainer = Trainer(net=net, training_data=train_dataloader, optimizer=optimizer, criterion=criterion)
        trainer.train(device)

        tester = Tester(net=net, test_data=test_dataloader)
        tester.evaluate_accuracy(device)

        print("--- %s seconds ---\n" % (time.time() - start_time))

except KeyboardInterrupt:
    print("Interrupted from keyboard")