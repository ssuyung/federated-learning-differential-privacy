# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import gaussian_noise, laplacian_noise
from utils import clip_grad_l1
from rdp_analysis import calibrating_sampled_gaussian
import matplotlib.pyplot as plt


from MLModel import *

import numpy as np
import copy


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, batch_size, q, clip, sigma, noise_level=1, noise_gamma=1, fixed_sigma=0, epsilon=None, device=None, noise_type="gaussian"):
    # def __init__(self, model, output_size, data, lr, E, batch_size, q, clip, sigma=None,  device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.sigma = sigma    # DP noise level
        self.epsilon = epsilon
        self.lr = lr
        self.E = E
        self.clip = clip
        self.q = q
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.noise_gamma = noise_gamma
        self.fixed_sigma = fixed_sigma

        if model == 'scatter':
            self.model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.model = model(data[0].shape[1], output_size).to(self.device)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.model.parameters())
        
        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]

            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )
            
            optimizer.zero_grad()

            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad 
                    self.model.zero_grad()

                
                """
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    clip_grad_l1(self.model.parameters(), self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad
                    self.model.zero_grad()
                """

            # print(np.linalg.norm(clipped_grads, 2))
            
            # add Gaussian noise
            # add laplacian noise
            noise_ls = []
            for name, param in self.model.named_parameters():
                if self.noise_type == 'gaussian': noise = self.noise_level * gaussian_noise(clipped_grads[name], self.clip, self.sigma, self.fixed_sigma, device=self.device)
                else: noise = self.noise_level * laplacian_noise(clipped_grads[name].shape, self.clip, self.epsilon, device=self.device)
                clipped_grads[name] += noise

            #Change        
            #self.grad.append(np.array(list(clipped_grads.values().cpu())).mean())
            #elf.noise.append(sum(noise_ls)/len(noise_ls))
            #self.grad.append(np.mean([g.cpu().numpy() for g in clipped_grads.values()]))
            self.grad.append(np.mean([g.norm().cpu().numpy() for g in clipped_grads.values()]))
            #self.noise.append(np.mean(noise_ls))
            self.noise.append(np.mean([n.norm().cpu().numpy() for n in noise_ls]))
            

            # scale back
            for name, param in self.model.named_parameters():
                clipped_grads[name] /= (self.data_size*self.q)
            
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            
            self.noise_level *= self.noise_gamma
            # update local model
            optimizer.step()
        # # Generate x values (indices of the lists)
        # x = list(range(len(self.noise)))
        # print(x)
        # # Plot both lines
        # plt.figure(figsize=(10, 6))  # Set the figure size
        # # plt.plot(x, self.noise, label='Gradient Line', color='blue', marker='o')
        # plt.plot(x, self.noise, label='Grad Line', color='red', linestyle='--')

        # # Add titles, labels, and legend
        # # plt.title('Gradient', fontsize=14)
        # plt.xlabel('Index', fontsize=12)
        # plt.ylabel('Value', fontsize=12)
        # plt.legend(fontsize=12)

        # # Display the grid and plot
        # plt.grid(True)
        # # plt.show()
        # plt.savefig('results/plots/histogram.png', format='png', dpi=300)



class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']      # (float) C in [0, 1]
        self.clip = fl_param['clip']
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)
        self.noise_type = fl_param['noise_type']

        self.data = []
        self.target = []
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]    # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']
        
        # compute noise using moments accountant
        # self.sigma = compute_noise(1, fl_param['q'], fl_param['eps'], fl_param['E']*fl_param['tot_T'], fl_param['delta'], 1e-5)
        
        self.fixed_sigma = fl_param['fixed_sigma']
        # calibration with subsampeld Gaussian mechanism under composition 
        if fl_param['noise_type'] == 'gaussian':
            self.sigma = fl_param['sigma'] if self.fixed_sigma else calibrating_sampled_gaussian(fl_param['q'], fl_param['eps'], fl_param['delta'], iters=fl_param['E']*fl_param['tot_T'], err=1e-3)
            print("noise scale =", self.sigma)
        elif fl_param['noise_type'] == 'laplacian':
            self.epsilon = fl_param['epsilon']
            print("Laplacian noise epsilon =", self.epsilon)

        
        
        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 self.sigma,
                                 fl_param['noise_level'],
                                 fl_param['noise_gamma'],
                                 fl_param['fixed_sigma'],
                                 fl_param['epsilon'],
                                 self.device)
                        for i in range(self.client_num)]
        
        if fl_param['model'] == 'scatter':
            self.global_model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        else:
            self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self):
        # idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        acc = self.test_acc()
        torch.cuda.empty_cache()
        return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr

