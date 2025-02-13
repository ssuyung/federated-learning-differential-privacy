"""
Useful tools
"""
import numpy as np
import random
import torch


def mnist_noniid(labels, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # num_shards, num_imgs = 30, 2000
    num_shards = int(num_users*3)
    num_imgs = int(60000 / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def gaussian_noise(data, s, sigma, fixed_sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma if fixed_sigma else data.reshape(-1).float().std(), data.shape).to(device)

def laplace_noise(data, s, sigma, device=None):
    """
    Laplace noise
    """
    return torch.normal(0, data.reshape(-1).float().std(), data.shape).to(device)

def laplacian_noise(data_shape, clip, epsilon, device=None):
    """
    Generate Laplacian noise for differential privacy.
    :param data_shape: The shape of the gradient tensor to which noise will be added.
    :param clip: Gradient clipping bound (L1 sensitivity).
    :param epsilon: Privacy budget for the Laplacian mechanism.
    :param device: Device to place the noise tensor (CPU or GPU).
    :return: Tensor of Laplacian noise.
    """
    scale = clip/epsilon  # Scale parameter for Laplacian distribution
    noise = torch.tensor(np.random.laplace(loc=0, scale=scale, size=data_shape), device=device)
    return noise

def clip_grad_l1(parameters, max_norm): 
    """
    Perform L1 gradient clipping.
    :param parameters: Model parameters (e.g., model.parameters()).
    :param max_norm: Maximum allowed L1 norm for gradients.
    """
    for p in parameters:
        if p.grad is not None:
            grad = p.grad
            grad_norm = torch.sum(torch.abs(grad))
            if grad_norm > max_norm:
                p.grad *= max_norm / grad_norm
