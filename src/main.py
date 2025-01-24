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

    for n_nodes in range(12000, 60001, 12000):
        print(f'Test with n_nodes = {n_nodes}')

        start_time = time.time()

        net = Net(n_nodes=n_nodes)
        model = net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

        dataset = Dataset(batch_size=256, shuffle_train=True, train_set_size=30000, test_set_size=7500)
        train_set, test_set = dataset.prepare_dataset()

        trainer = Trainer(net=net, training_data=train_set, optimizer=optimizer, criterion=criterion)
        trainer.train()

        tester = Tester(net=net, test_data=test_set)
        tester.evaluate_accuracy()

        print("--- %s seconds ---\n" % (time.time() - start_time))

        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

except KeyboardInterrupt:
    print("Interrupted from keyboard")