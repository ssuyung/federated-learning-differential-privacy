# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
def load_cnn_mnist(num_users):
    train = datasets.MNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    train_data = train.data.float().unsqueeze(1)
    train_label = train.targets

    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std

    test = datasets.MNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    test_data = test.data.float().unsqueeze(1)
    test_label = test.targets
    test_data = (test_data - mean) / std

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(train_label, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = train_data[idx]
        targets = train_label[idx].float()
        non_iid.append((d, targets))
    non_iid.append((test_data.float(), test_label.float()))
    return non_iid

"""
1. load_data
2. generate clients (step 3)
3. generate aggregator
4. training
"""
client_num = 4
d = load_cnn_mnist(client_num)
"""
FL model parameters.
"""
import warnings
warnings.filterwarnings("ignore")

lr = 0.15

fl_param = {
    'output_size': 10,
    'client_num': client_num,
    'model': MNIST_CNN,
    'data': d,
    'lr': lr,
    'E': 30,
    'C': 1,
    'eps': 4.0,
    'delta': 1e-5,
    'q': 0.01,
    'clip': 0.1,
    'tot_T': 70,
    'batch_size': 128,
    'device': device,
    'noise_level': 1,
    'noise_gamma': 0.9999,   # noise_level = noise_level * noise_gamma for every local epoch
    'fixed_sigma': True,
    'sigma' : 3,
    'noise_type': 'laplacian',
    'epsilon':1.0,
}

fl_entity = FLServer(fl_param).to(device)

import time

acc = []
start_time = time.time()
for t in range(fl_param['tot_T']):
    acc += [fl_entity.global_update()]
    print("global epochs = {:d}, acc = {:.4f}".format(t+1, acc[-1]), " Time taken: %.2fs" % (time.time() - start_time))

# Generate x values (indices of the lists)
x = list(range(len(acc)))

dictt = {'accuracy': acc}
df = pd.DataFrame(dictt)
df.to_csv(f'./results/csv/laplace_gamma_{fl_param["noise_gamma"]}_{fl_param["epsilon"]*100}.csv')

# Plot both lines
plt.figure(figsize=(10, 6))  # Set the figure size
# plt.plot(x, self.grad, label='Gradient Line', color='blue', marker='o')
plt.plot(x, acc, label='accuracy', color='red', linestyle='--')

# Add titles, labels, and legend
plt.title('Line Chart of Gradient and Noise', fontsize=14)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(fontsize=12)

# Display the grid and plot
plt.grid(True)
# plt.show()
plt.savefig('results/plots/accuracy.png', format='png', dpi=300)
